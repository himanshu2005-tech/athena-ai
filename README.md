# Athena AI

Athena is a production-oriented AI assistant built on a fine-tuned TinyLlama model and an advanced Retrieval-Augmented Generation (RAG) pipeline.

The system combines supervised fine-tuning, hybrid retrieval, web search integration, reranking, and grounding validation to generate context-aware responses while reducing hallucinations.

---

## Key Features

- Fine-tuned TinyLlama-1.1B using LoRA (PEFT)
- Trained on 20,000+ instruction-tuning samples
- Hybrid Retrieval-Augmented Generation (RAG)
- Tavily-powered real-time web search
- ChromaDB vector database
- Dense + Sparse retrieval fusion
- Cross-Encoder reranking
- Parent-child document chunking
- HyDE (Hypothetical Document Embeddings)
- Grounding validation and safe-response mechanism
- React frontend with Python backend

---

## Architecture

```text
User Query
    │
    ▼
Query Expansion + HyDE
    │
    ▼
Dense Retrieval (Embeddings)
    │
    ├──────────────┐
    ▼              ▼
ChromaDB      BM25 Retrieval
    │              │
    └──────┬───────┘
           ▼
   Reciprocal Rank Fusion
           ▼
 Cross Encoder Reranking
           ▼
 Parent Context Recovery
           ▼
 Fine-Tuned TinyLlama
           ▼
 Grounding Validation
           ▼
      Final Answer
```

---

## Tech Stack

### LLM

- TinyLlama-1.1B
- Hugging Face Transformers
- PEFT
- LoRA
- TRL
- PyTorch

### Retrieval

- ChromaDB
- Sentence Transformers
- BM25
- HyDE
- Reciprocal Rank Fusion (RRF)
- Cross Encoder Reranking

### Search

- Tavily Search API

### Frontend

- React
- JavaScript
- HTML/CSS

### Backend

- Python
- FastAPI / ASGI

---

## Fine-Tuning Details

| Parameter | Value |
|------------|---------|
| Base Model | TinyLlama-1.1B |
| Dataset Size | 20,000+ Samples |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 2e-4 |
| Training Method | PEFT + LoRA |
| Training Time | 15+ Hours |

### Dataset Composition

- 10,000 coding instruction samples
- 2,000 conversational samples
- Additional instruction tuning data
- Focus areas:
  - Python
  - React
  - JavaScript
  - Firebase
  - REST APIs
  - Full Stack Development

---

## Retrieval Pipeline

1. User query received
2. Tavily fetches relevant web sources
3. Documents are cleaned and chunked
4. Chunks are embedded and stored in ChromaDB
5. Dense and sparse retrieval executed
6. Results fused using RRF
7. Cross-Encoder reranking applied
8. Relevant context sent to the model
9. Grounding validation performed
10. Final response generated

---

## Installation

```bash
git clone https://github.com/yourusername/athena-ai.git

cd athena-ai

pip install -r requirements.txt
```

---

## Environment Variables

```env
ATHENA_MODEL_PATH=./final_athena_model

TAVILY_API_KEY=your_api_key_here
```

---

## Future Improvements

- Multi-agent workflows
- Hybrid BM25 + Vector Search optimization
- Larger foundation models
- Tool calling capabilities
- Memory-enhanced reasoning
- Autonomous research agents

---

## Author

Built to explore modern LLM engineering techniques including:

- Fine-Tuning
- Retrieval-Augmented Generation
- Vector Databases
- Semantic Search
- Hybrid Retrieval
- Real-Time Web Search
- AI System Design

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.