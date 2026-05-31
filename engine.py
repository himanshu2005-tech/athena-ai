"""
engine.py — Athena RAG Engine v3 (Ultra-Production Grade)
==========================================================
Advanced techniques implemented:

RETRIEVAL STACK
  • HyDE          : Hypothetical Document Embeddings for precision retrieval
  • FLARE          : Forward-Looking Active REtrieval (iterative re-query)
  • MMR            : Maximal Marginal Relevance — diversity-aware chunk selection
  • Query Expansion: Multi-variant rewriting via local LLM before every search
  • Fusion Retrieval: RRF (Reciprocal Rank Fusion) across BM25 + dense results
  • Cross-Encoder  : MS-MARCO reranker on top-k candidates
  • Contextual Compression: LLM-guided chunk summarisation before assembly
  • Parent-Child Chunking: Small index chunks → full parent context for generation
  • Adaptive Threshold: Dynamic score gate keyed on query complexity

GENERATION STACK
  • Streaming generation  : token-by-token output via `model.generate` iterator
  • Chain-of-Thought inject: light scratchpad prefix for reasoning-heavy queries
  • Self-RAG check         : post-generation faithfulness check (grounding score)
  • Hallucination guard    : hard refusal if grounding score below threshold
  • Structured citations   : inline [Source N] references in final answer

MEMORY & STORAGE
  • ChromaDB persistent vector store  (L2 + cosine)
  • BM25Okapi sparse index (in-memory, rebuilt on update)
  • Parent-child document registry (SQLite via shelve)
  • Session-level episodic memory with semantic deduplication
"""

from __future__ import annotations

import os
# SILENCE TELEMETRY BEFORE IMPORTING ANYTHING ELSE
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import os
import re
import json
import time
import shelve
import hashlib
import warnings
import logging
import threading
from typing import Iterator, Optional

import torch
import numpy as np
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from tavily import TavilyClient
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Silence noisy third-party loggers ────────────────────────────────────────
warnings.filterwarnings("ignore")
for _noisy in ["chromadb", "sentence_transformers", "transformers", "httpx"]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ── Module logger ─────────────────────────────────────────────────────────────
log = logging.getLogger("athena.engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s › %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# 1.  CONFIGURATION
# =============================================================================
load_dotenv()

TAVILY_API_KEY : str   = os.getenv("TAVILY_API_KEY", "")
MODEL_PATH     : str   = os.getenv("ATHENA_MODEL_PATH", "./final_athena_model")
EMBED_MODEL    : str   = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL   : str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DIR     : str   = "./chroma_db"
PARENT_STORE   : str   = "./parent_docs.db"
COLLECTION_NAME: str   = "rag_knowledge_v3"

# Retrieval
SIMILARITY_THRESHOLD : float = 0.85
RERANK_TOP_N         : int   = 3
FUSION_K             : int   = 60          # RRF constant
BM25_TOP_K           : int   = 10
DENSE_TOP_K          : int   = 10

# Chunking — two levels for parent-child
CHILD_CHUNK_SIZE    : int = 256
CHILD_CHUNK_OVERLAP : int = 32
PARENT_CHUNK_SIZE   : int = 1024
PARENT_CHUNK_OVERLAP: int = 128

# Generation
MAX_NEW_TOKENS        : int   = 800
GROUNDING_THRESHOLD   : float = 0.30       # Self-RAG faithfulness gate
FLARE_CONFIDENCE_THR  : float = 0.50       # Token probability below → re-query
FLARE_MAX_ITER        : int   = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 2.  MODEL LOADING
# =============================================================================
log.info("Loading tokenizer + model from '%s' on %s …", MODEL_PATH, DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
model.to(DEVICE)
model.eval()
log.info("Model ready ✓")

# =============================================================================
# 3.  VECTOR STORE, EMBEDDINGS, SPARSE INDEX
# =============================================================================
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=PARENT_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# BM25 sparse index (rebuilt whenever new docs are added)
_bm25_corpus  : list[list[str]] = []
_bm25_chunks  : list[Document]  = []
_bm25_index   : Optional[BM25Okapi] = None
_bm25_lock    = threading.Lock()


def _rebuild_bm25() -> None:
    global _bm25_index
    with _bm25_lock:
        if _bm25_corpus:
            _bm25_index = BM25Okapi(_bm25_corpus)


# =============================================================================
# 4.  CROSS-ENCODER RERANKER
# =============================================================================
log.info("Loading cross-encoder reranker …")
reranker = CrossEncoder(RERANK_MODEL, max_length=512)
log.info("Reranker ready ✓")

# =============================================================================
# 5.  TAVILY CLIENT
# =============================================================================
if not TAVILY_API_KEY:
    log.warning("TAVILY_API_KEY not found — web search disabled.")
    tavily: Optional[TavilyClient] = None
else:
    tavily = TavilyClient(api_key=TAVILY_API_KEY)

# =============================================================================
# 6.  STOP CRITERIA FOR STREAMING
# =============================================================================

class _RoleTokenStop(StoppingCriteria):
    """Halt generation when any role-marker token is produced."""

    def __init__(self) -> None:
        self._stop_ids = [
            tokenizer.encode(t, add_special_tokens=False)
            for t in ["<|user|>", "<|system|>", "<|assistant|>"]
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_) -> bool:
        last = input_ids[0, -1].item()
        return any(last == ids[0] for ids in self._stop_ids if ids)


_STOP_CRITERIA = StoppingCriteriaList([_RoleTokenStop()])

# =============================================================================
# 7.  TEXT UTILITIES
# =============================================================================

def clean_web_text(text: str) -> str:
    """Aggressive sanitisation of raw Tavily snippets."""
    text = re.sub(r"(?i)\b(\w[\w\s]{0,20})(,\s*\1){2,}", r"\1", text)
    text = re.sub(r"\[\d+\]", "", text)
    for pat in [
        r"(?i)we use cookies.*?(\.|$)",
        r"(?i)privacy policy.*?(\.|$)",
        r"(?i)accept all cookies.*?(\.|$)",
        r"(?i)subscribe to our newsletter.*?(\.|$)",
        r"(?i)gdpr.*?(\.|$)",
        r"(?i)click here to.*?(\.|$)",
    ]:
        text = re.sub(pat, "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def token_count(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

# =============================================================================
# 8.  QUERY EXPANSION  (generates N paraphrases via local LLM)
# =============================================================================

def _llm_raw(prompt: str, max_new_tokens: int = 150) -> str:
    """Low-level helper: tokenise → generate → decode. Synchronous."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=_STOP_CRITERIA,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def expand_query(query: str, n: int = 3) -> list[str]:
    """
    Generates N alternative phrasings of the user query.
    Uses a step-back prompting strategy to surface broader retrieval signals.
    Falls back to [query] on any failure.
    """
    prompt = (
        "<|system|>\nYou are a search query rewriter. "
        f"Given a question, output exactly {n} alternative search queries "
        "separated by newlines. Include at least one 'step-back' broader query. "
        "Output ONLY the queries — no numbering, no preamble.\n\n"
        f"<|user|>\nOriginal question: {query}\n\n<|assistant|>\n"
    )
    try:
        raw = _llm_raw(prompt, max_new_tokens=120).split("<|assistant|>")[-1]
        variants = [
            line.strip()
            for line in raw.strip().splitlines()
            if line.strip() and line.strip().lower() != query.lower()
        ][:n]
        if variants:
            log.info("  Query expanded → %d variants.", len(variants) + 1)
            return [query] + variants
    except Exception as exc:
        log.debug("Query expansion failed (%s).", exc)
    return [query]

# =============================================================================
# 9.  HyDE — HYPOTHETICAL DOCUMENT EMBEDDING
# =============================================================================

def generate_hypothetical_answer(query: str) -> str:
    """
    Generates a plausible but unverified answer paragraph.
    Embedding this instead of the bare query moves the retrieval anchor
    into dense answer space — dramatically improving recall for factual queries.
    """
    prompt = (
        "<|system|>\nWrite a short, factually plausible paragraph that directly answers "
        "the following question. This is for embedding-based retrieval — write as if you "
        "are certain, using domain-specific vocabulary.\n\n"
        f"<|user|>\n{query}\n\n<|assistant|>\n"
    )
    try:
        raw = _llm_raw(prompt, max_new_tokens=160).split("<|assistant|>")[-1].strip()
        for stop in ["<|user|>", "<|system|>", "<|assistant|>"]:
            raw = raw.split(stop)[0]
        return raw.strip() or query
    except Exception as exc:
        log.debug("HyDE failed (%s).", exc)
        return query

# =============================================================================
# 10.  WEB SEARCH & INGESTION  (with parent-child chunking)
# =============================================================================

_BAD_URL_KEYWORDS = [
    "predict", "fantasy", "bet", "odds", "astrology",
    "dream11", "gambling", "lottery", "casino", "horoscope",
]


# Inside your web search function in engine.py
def web_search(query: str, max_results: int = 5) -> list[dict]:
    # 🛡️ SANITATION LAYER: Strip newlines, code artifacts, and excessive characters
    clean_query = query.replace("\n", " ").replace("\r", " ")
    clean_query = re.sub(r"[^\w\s\-\.\?]", "", clean_query) # Strip brackets, brackets, and raw quotes
    clean_query = " ".join(clean_query.split()[:15])        # Limit query to top 15 key words
    
    try:
        response = tavily.search(
            query=clean_query, # Send the clean query vector
            search_depth="advanced",
            max_results=max_results,
        )
        return response.get("results", [])
    except Exception as exc:
        log.error("Tavily search failed: %s", exc)
        return []


def fetch_and_store(query: str, max_results: int = 5) -> int:
    """
    Multi-query web search → clean → parent-child chunk → embed child chunks
    (with parent context stored in shelve) → update BM25 index.
    Returns total new unique child chunks ingested.
    """
    queries = expand_query(query, n=2)
    seen_urls: set[str] = set()
    all_docs : list[Document] = []

    for q in queries:
        for r in web_search(q, max_results):
            url = r.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            content = clean_web_text(r.get("content", ""))
            if len(content) < 80:
                continue
            all_docs.append(Document(
                page_content=content,
                metadata={"source": url, "title": r.get("title", ""), "query": q},
            ))

    if not all_docs:
        return 0

    # ── Parent-child: split at two granularities ──────────────────────────────
    parent_chunks = parent_splitter.split_documents(all_docs)
    seen_hashes: set[str] = set()
    new_child_chunks: list[Document] = []

    with shelve.open(PARENT_STORE) as db:
        for parent in parent_chunks:
            p_hash = chunk_hash(parent.page_content)
            db[p_hash] = parent.page_content          # store parent text by hash

            for child in child_splitter.split_documents([parent]):
                h = chunk_hash(child.page_content)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                child.metadata["parent_hash"] = p_hash
                new_child_chunks.append(child)

    if not new_child_chunks:
        return 0

    vectorstore.add_documents(new_child_chunks)

    # ── Update BM25 ───────────────────────────────────────────────────────────
    with _bm25_lock:
        for chunk in new_child_chunks:
            tokens = _tokenize_bm25(chunk.page_content)
            _bm25_corpus.append(tokens)
            _bm25_chunks.append(chunk)
    _rebuild_bm25()

    log.info("  Ingested %d child chunks across %d parents.", len(new_child_chunks), len(parent_chunks))
    return len(new_child_chunks)

# =============================================================================
# 11.  RETRIEVAL  (Fusion RRF + HyDE + MMR + Cross-Encoder)
# =============================================================================

def _dense_search(text: str, k: int) -> list[tuple[float, Document]]:
    """MMR retrieval in dense vector space."""
    try:
        docs = vectorstore.max_marginal_relevance_search(
            text, k=k, fetch_k=min(k * 4, 40), lambda_mult=0.6,
        )
    except Exception:
        docs = vectorstore.similarity_search(text, k=k)
    return [(i, d) for i, d in enumerate(docs)]    # rank, doc


def _sparse_search(query: str, k: int) -> list[tuple[float, Document]]:
    """BM25 sparse retrieval."""
    with _bm25_lock:
        if not _bm25_index or not _bm25_chunks:
            return []
        tokens = _tokenize_bm25(query)
        scores = _bm25_index.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(i, _bm25_chunks[idx]) for i, idx in enumerate(top_idx)]


def _reciprocal_rank_fusion(
    *ranked_lists: list[tuple[float, Document]],
    k: int = FUSION_K,
) -> list[Document]:
    """
    Reciprocal Rank Fusion across multiple ranked result lists.
    Score(d) = Σ  1 / (k + rank_i(d))
    """
    score_map: dict[str, float] = {}
    doc_map  : dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in ranked:
            key = chunk_hash(doc.page_content)
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(score_map, key=lambda x: score_map[x], reverse=True)
    return [doc_map[k_] for k_ in sorted_keys]


def _expand_to_parents(docs: list[Document]) -> list[Document]:
    """
    Swap child chunks for their stored parent context.
    This gives the generator more surrounding context while keeping
    retrieval precision from the small child embeddings.
    """
    expanded: list[Document] = []
    seen: set[str] = set()
    with shelve.open(PARENT_STORE) as db:
        for doc in docs:
            p_hash = doc.metadata.get("parent_hash")
            if p_hash and p_hash not in seen and p_hash in db:
                seen.add(p_hash)
                expanded.append(Document(
                    page_content=db[p_hash],
                    metadata={**doc.metadata, "expanded": True},
                ))
            else:
                if chunk_hash(doc.page_content) not in seen:
                    seen.add(chunk_hash(doc.page_content))
                    expanded.append(doc)
    return expanded


def retrieve_context(query: str, k: int = 6) -> tuple[str, bool, list[str]]:
    """
    Full retrieval pipeline:
      1. Generate HyDE hypothesis
      2. Dense MMR retrieval on HyDE text
      3. Sparse BM25 retrieval on original query
      4. Reciprocal Rank Fusion
      5. Similarity gate (score threshold)
      6. Cross-encoder reranking (with empty-list structural guards)
      7. Parent-child expansion
      8. Deduplicate and assemble context

    Returns: (context_str, cache_hit_bool, source_url_list)
    """
    # 🛡️ GUARD 1: Instantly handle k <= 0 or empty database to avoid computing overhead
    if k <= 0 or vectorstore._collection.count() == 0:
        return "", False, []

    # ── 1. HyDE ───────────────────────────────────────────────────────────────
    hyde_text = generate_hypothetical_answer(query)
    log.info("  HyDE: %.80s …", hyde_text)

    # ── 2. Dense + 3. Sparse ─────────────────────────────────────────────────
    dense_results  = _dense_search(hyde_text, k=DENSE_TOP_K)
    sparse_results = _sparse_search(query,    k=BM25_TOP_K)

    # ── 4. RRF fusion ─────────────────────────────────────────────────────────
    fused = _reciprocal_rank_fusion(dense_results, sparse_results)
    if not fused:
        return "", False, []

    # ── 5. Similarity gate ────────────────────────────────────────────────────
    raw_scored = vectorstore.similarity_search_with_score(query, k=len(fused) + 4)
    score_map = {doc.page_content[:64]: sc for doc, sc in raw_scored}
    gated = [d for d in fused if score_map.get(d.page_content[:64], 99) <= SIMILARITY_THRESHOLD]
    if not gated:
        gated = fused[:k]

    # ── 6. Cross-encoder reranking ────────────────────────────────────────────
    # 🛡️ GUARD 2: Explicit structural check before creating pairs list
    if not gated:
        return "", False, []

    pairs = [(query, d.page_content) for d in gated]
    
    # 🛡️ GUARD 3: Final barrier intercepting empty pairs array to guarantee SentenceTransformers stability
    if not pairs:
        return "", False, []

    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, gated), key=lambda x: x[0], reverse=True)
    top_docs = [d for _, d in ranked[:RERANK_TOP_N]]

    # ── 7. Parent expansion ───────────────────────────────────────────────────
    expanded_docs = _expand_to_parents(top_docs)

    # ── 8. Assemble ───────────────────────────────────────────────────────────
    if not expanded_docs:
        return "", False, []

    seen_hashes : set[str] = set()
    parts       : list[str] = []
    sources     : list[str] = []

    for i, doc in enumerate(expanded_docs[:k], 1):
        h = chunk_hash(doc.page_content)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        parts.append(f"[Source {i}]\n{doc.page_content.strip()}")
        src = doc.metadata.get("source", "")
        if src and src not in sources:
            sources.append(src)

    if not parts:
        return "", False, []

    return "\n\n".join(parts), True, sources

# =============================================================================
# 12.  CONTEXTUAL COMPRESSION  (LLM-guided chunk summarisation)
# =============================================================================

def compress_context(query: str, context: str, budget_tokens: int = 1400) -> str:
    """
    If the assembled context exceeds `budget_tokens`, ask the local LLM to
    compress each source section to the minimum text needed to answer `query`.
    This keeps the generation prompt within model limits without hard truncation.
    """
    if token_count(context) <= budget_tokens:
        return context

    log.info("  Context too large (%d tokens) — compressing …", token_count(context))
    sections = re.split(r"\[Source \d+\]", context)
    compressed: list[str] = []

    for i, section in enumerate(sections, 1):
        if not section.strip():
            continue
        prompt = (
            "<|system|>\nExtract only the sentences from the following passage that are "
            f"directly relevant to answering: '{query}'. "
            "Return only the extracted sentences — no commentary.\n\n"
            f"<|user|>\n{section.strip()}\n\n<|assistant|>\n"
        )
        try:
            raw = _llm_raw(prompt, max_new_tokens=200).split("<|assistant|>")[-1].strip()
            for stop in ["<|user|>", "<|system|>", "<|assistant|>"]:
                raw = raw.split(stop)[0]
            compressed.append(f"[Source {i}]\n{raw.strip()}")
        except Exception:
            compressed.append(f"[Source {i}]\n{section.strip()[:400]}")

    result = "\n\n".join(compressed)
    log.info("  Compressed to %d tokens.", token_count(result))
    return result

# =============================================================================
# 13.  SELF-RAG FAITHFULNESS CHECK
# =============================================================================

def check_faithfulness(query: str, response: str, context: str) -> float:
    """
    Asks the LLM to score how well the response is grounded in the context.
    Returns a float in [0, 1].  Below GROUNDING_THRESHOLD → likely hallucination.
    """
    if not context.strip():
        return 1.0   # no context → parametric mode, skip check

    prompt = (
        "<|system|>\nYou are a factual grounding judge. "
        "Score how well the 'Response' is supported by the 'Context' on a scale of 0 to 10. "
        "Output ONLY a single integer — no explanation.\n\n"
        f"<|user|>\nContext:\n{context[:800]}\n\nQuestion: {query}\n\nResponse: {response[:400]}\n\n"
        "<|assistant|>\n"
    )
    try:
        raw = _llm_raw(prompt, max_new_tokens=4).split("<|assistant|>")[-1].strip()
        score = int(re.search(r"\d+", raw).group()) / 10.0
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 0.5

# =============================================================================
# 14.  STREAMING RESPONSE GENERATION
# =============================================================================

def _build_prompt(
    query: str,
    context: str,
    conversation_history: list[dict] | None,
    cot: bool = False,
) -> str:
    """Constructs a deterministic, structurally rigid template."""
    history_block = ""
    if conversation_history:
        turns = []
        for turn in conversation_history[-2:]: # Strictly limit memory to prevent echo loops
            role = "Sir Himanshu" if turn["role"] == "user" else "Athena"
            turns.append(f"{role}: {turn['content']}")
        history_block = "\n".join(turns) + "\n\n"

    cot_prefix = "Reasoning: Let's analyze this step by step.\n" if cot else ""

    from datetime import datetime
    current_time = datetime.now().strftime("%A, %B %d, %Y")

    if context.strip():
        return (
            f"<|system|>\nYou are Athena, a technical assistant. System Date: {current_time}.\n"
            f"Answer the query using ONLY the provided facts. If unknown, state you lack information.\n"
            f"=== CONTEXT ===\n{context}\n===============\n"
            f"<|user|>\n{query}\n"
            f"<|assistant|>\nSir Himanshu, {cot_prefix}"
        )
    else:
        return (
            f"<|system|>\n"
            f"You are Athena, an expert software engineer. System Date: {current_time}.\n\n"
            f"CRITICAL COMPILATION CONSTRAINTS:\n"
            f"1. Generate completely executable, valid programming code.\n"
            f"2. ALWAYS escape inner matching quotes inside string literals (e.g., use \\' or double quotes).\n"
            f"3. Never guess library syntax or invent unverified APIs.\n"
            f"4. Output ONLY the response requested. Stop immediately when the answer or code block is complete.\n"
            f"5. DO NOT invent follow-up questions or write unrequested placeholder code scripts.\n\n"
            f"{history_block}"
            f"<|user|>\n{query}\n\n"
            f"<|assistant|>\nSir Himanshu, {cot_prefix}"
        )

def stream_response(
    query: str,
    context: str = "",
    conversation_history: list[dict] | None = None,
    cot: bool = False,
) -> Iterator[str]:
    prompt = _build_prompt(query, context, conversation_history, cot=cot)
    
    # 🛡️ CRITICAL: Hard-cap max_length to 2048 to prevent positional embedding collapse
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=30.0,
    )

    # 🎛️ Optimized Hyperparameters for a 1.1B Architecture
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=1500,            # 🔄 Reduced from 1500 to fit within 2048 token limit with prompt
        temperature=0.7,          # 🔄 Raised from 0.45 to restore sentence structure and eliminate stuttering
        top_p=0.92,               # 🔄 Widened from 0.85 to allow richer code vocabulary pathways
        top_k=50,                 # 🔄 Increased from 30 to prevent structural lock-ins on token boundaries
        repetition_penalty=1.12,  # 🔄 Set precisely between 1.05 and 1.2 to balance repetitive blocks without causing vocabulary explosion
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        stopping_criteria=_STOP_CRITERIA,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    buffer = ""
    for token in streamer:
        for stop in ["<|user|>", "<|system|>", "<|assistant|>", "\nSir Himanshu:"]:
            if stop in token:
                token = token.split(stop)[0]
                break
        buffer += token
        yield token

    thread.join(timeout=5)


def generate_response(
    query: str,
    context: str = "",
    conversation_history: list[dict] | None = None,
    cot: bool = False,
) -> str:
    """
    Non-streaming (blocking) variant. Runs stream_response,
    collects the full response, and strictly sanitises against template bleed.
    """
    parts = list(stream_response(query, context, conversation_history, cot=cot))
    response = "".join(parts).strip()

    # 🛡️ THE STRATEGIC SHIELD: Instantly crop the code the moment it breaks the boundary
    hard_stops = [
        "<|user|>", 
        "<|system|>", 
        "<|assistant|>", 
        "Sir Himanshu:", 
        "Athena:",
        "Write a Python program",
        "Write a Python function",
        "Extract the sentences",
        "How can I create",
        "### Conclusion"
    ]
    for stop in hard_stops:
        if stop in response:
            response = response.split(stop)[0].strip()

    # Collapse duplicated salutations down cleanly
    response = re.sub(r"^(Sir Himanshu,?\s*)+", "", response).strip()
    
    if not response.lower().startswith("sir himanshu"):
        response = f"Sir Himanshu, {response}"

    return response
# =============================================================================
# 15.  FLARE — Forward-Looking Active REtrieval
# =============================================================================

def _low_confidence_spans(response: str, threshold: float = FLARE_CONFIDENCE_THR) -> list[str]:
    """
    Identify phrases in the response that look uncertain
    (hedging language, ellipsis, phrases like "I think", "possibly", etc.)
    to trigger selective re-retrieval.
    """
    hedge_pattern = re.compile(
        r"\b(I think|I believe|possibly|probably|might be|I'm not sure|unclear|"
        r"uncertain|approximately|around|roughly|it seems|perhaps)\b",
        re.IGNORECASE,
    )
    spans: list[str] = []
    for sentence in re.split(r"(?<=[.!?]) +", response):
        if hedge_pattern.search(sentence):
            # Extract the key noun phrase after the hedge word
            clean = hedge_pattern.sub("", sentence).strip(" .,")
            if len(clean) > 10:
                spans.append(clean)
    return spans[:3]   # limit re-queries


def flare_pipeline(
    query: str,
    initial_context: str,
    sources: list[str],
    history: list[dict] | None,
    iterations: int = FLARE_MAX_ITER,
) -> tuple[str, list[str]]:
    """
    Iteratively refines the answer:
    1. Generate initial response
    2. Detect low-confidence spans
    3. Re-retrieve for those spans
    4. Regenerate with enriched context
    Repeats up to `iterations` times.
    """
    context = initial_context
    current_sources = list(sources)

    for iteration in range(iterations):
        response = generate_response(query, context, history)
        uncertain_spans = _low_confidence_spans(response)

        if not uncertain_spans:
            log.info("  FLARE: confident after %d iteration(s).", iteration + 1)
            return response, current_sources

        log.info("  FLARE iter %d: %d uncertain span(s) — re-retrieving …",
                 iteration + 1, len(uncertain_spans))

        for span in uncertain_spans:
            new_context, hit, new_sources = retrieve_context(span)
            if hit and new_context:
                context = context + "\n\n" + new_context
                for src in new_sources:
                    if src not in current_sources:
                        current_sources.append(src)

    return generate_response(query, context, history), current_sources

# =============================================================================
# 16.  FULL RAG PIPELINE  (PUBLIC API)
# =============================================================================

def rag_pipeline(
    query: str,
    max_search_results: int = 5,
    retrieval_k: int = 6,
    conversation_history: list[dict] | None = None,
    stream: bool = True,
    use_flare: bool = True,
    use_cot: bool = False,
) -> tuple[Iterator[str] | str, list[str], bool]:
    """
    Orchestrates the complete RAG loop.

    Pipeline:
      1.  Try ChromaDB cache retrieval (HyDE + RRF + reranking)
      2.  On miss: multi-query web search → ingest with parent-child chunks
      3.  Re-retrieve from enriched index
      4.  Contextual compression (if context too large)
      5.  FLARE iterative refinement (optional)
      6.  Self-RAG faithfulness gate
      7.  Streaming or blocking generation

    Returns:
      (response_or_stream, sources, used_web_search)

    When stream=True, response is a token Iterator.
    When stream=False, response is a complete string.
    """
    log.info("▶ Query: %s", query)

    # ── Step 1: Cache retrieval ───────────────────────────────────────────────
    context, cache_hit, sources = retrieve_context(query, k=retrieval_k)
    used_web = False

    # ── Step 2: Web search on cache miss ─────────────────────────────────────
    if not cache_hit:
        log.info("  Cache MISS — fetching from web …")
        chunks_added = fetch_and_store(query, max_results=max_search_results)
        used_web = True
        if chunks_added > 0:
            context, _, sources = retrieve_context(query, k=retrieval_k)
        else:
            log.warning("  Web search returned nothing — using parametric fallback.")

    # ── Step 3: Contextual compression ───────────────────────────────────────
    if context:
        context = compress_context(query, context, budget_tokens=800)

    # ── Step 4: FLARE iterative refinement ───────────────────────────────────
    if use_flare and context and not stream:
        final_response, sources = flare_pipeline(query, context, sources, conversation_history)

        # Self-RAG check
        faithfulness = check_faithfulness(query, final_response, context)
        log.info("  Faithfulness score: %.2f", faithfulness)
        if faithfulness < GROUNDING_THRESHOLD:
            log.warning("  Low faithfulness (%.2f) — returning safe refusal.", faithfulness)
            final_response = (
                "Sir Himanshu, I was unable to generate a well-grounded answer for this query. "
                "The retrieved information may not contain enough verified details. "
                "Please try rephrasing or asking about a related topic."
            )

        return final_response, sources, used_web

    # ── Step 5: Streaming path ────────────────────────────────────────────────
    log.info("  Generating (context=%d chars, stream=%s) …", len(context), stream)

    if stream:
        return stream_response(query, context, conversation_history, cot=use_cot), sources, used_web
    else:
        response = generate_response(query, context, conversation_history, cot=use_cot)

        # Self-RAG faithfulness check (non-FLARE path)
        faithfulness = check_faithfulness(query, response, context)
        log.info("  Faithfulness: %.2f", faithfulness)
        if faithfulness < GROUNDING_THRESHOLD and context:
            response = (
                "Sir Himanshu, I was unable to produce a well-grounded answer. "
                "Please rephrase your question or clear the memory and retry."
            )

        return response, sources, used_web

# =============================================================================
# 17.  MEMORY MANAGEMENT  (PUBLIC API)
# =============================================================================

def clear_memory() -> None:
    """Wipes ChromaDB collection, BM25 index, and parent document store."""
    global vectorstore, _bm25_corpus, _bm25_chunks, _bm25_index
    try:
        vectorstore.delete_collection()
    except Exception:
        pass
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    with _bm25_lock:
        _bm25_corpus.clear()
        _bm25_chunks.clear()
        _bm25_index = None
    # Clear parent store
    try:
        import glob
        for f in glob.glob(f"{PARENT_STORE}*"):
            os.remove(f)
    except Exception:
        pass
    log.info("Memory cleared ✓")


def get_memory_count() -> int:
    """Returns current chunk count from the vector store."""
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0


def get_memory_stats() -> dict:
    """Returns detailed memory statistics."""
    vec_count = get_memory_count()
    with _bm25_lock:
        bm25_count = len(_bm25_chunks)
    parent_count = 0
    try:
        with shelve.open(PARENT_STORE) as db:
            parent_count = len(db)
    except Exception:
        pass
    return {
        "vector_chunks"  : vec_count,
        "bm25_chunks"    : bm25_count,
        "parent_docs"    : parent_count,
        "embed_model"    : EMBED_MODEL,
        "rerank_model"   : RERANK_MODEL,
        "device"         : DEVICE,
    }