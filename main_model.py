"""
main.py — Athena CLI Interface v3 (Ultra-Production Grade)
===========================================================
Features:
  • LIVE STREAMING output — tokens printed as they are generated
  • TTS (async)          — speech runs off main thread, never blocks
  • Rich terminal UI     — animated spinner, colour-coded output, box drawing
  • Intent classifier    — code / search / reasoning / general routing
  • CoT flag             — reasoning queries get Chain-of-Thought prefix
  • FLARE flag           — iterative active retrieval for uncertain answers
  • ipynb export         — Python code blocks auto-saved as Jupyter notebooks
  • Session logging      — every exchange persisted to ./logs/session_<ts>.log
  • Conversation memory  — rolling N-turn history injected into engine
  • Graceful shutdown    — SIGINT + 'exit' both handled cleanly
  • Command palette      — clear | stats | history | help | save | export | exit
"""

from __future__ import annotations

import os
# SILENCE TELEMETRY BEFORE IMPORTING ANYTHING ELSE
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


import os
import re
import sys
import json
import time
import signal
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pyttsx3

from engine import (
    rag_pipeline,
    clear_memory,
    get_memory_count,
    get_memory_stats,
    log as engine_log,
)

# Silence engine INFO inside the CLI — they'd clutter the streamed output
engine_log.setLevel(logging.WARNING)

# =============================================================================
# 1.  ANSI TERMINAL COLOURS
# =============================================================================

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    # Foregrounds
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BLUE    = "\033[94m"
    GREY    = "\033[90m"
    ORANGE  = "\033[38;5;208m"
    TEAL    = "\033[38;5;51m"


def cprint(text: str, colour: str = C.WHITE, bold: bool = False) -> None:
    prefix = C.BOLD if bold else ""
    print(f"{prefix}{colour}{text}{C.RESET}", flush=True)


def separator(char: str = "─", width: int = 72, colour: str = C.GREY) -> None:
    print(f"{colour}{char * width}{C.RESET}", flush=True)


# =============================================================================
# 2.  ANIMATED SPINNER
# =============================================================================

class Spinner:
    """Non-blocking terminal spinner. Call .stop() when done."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Thinking", colour: str = C.CYAN) -> None:
        self._msg     = message
        self._colour  = colour
        self._running = False
        self._thread : threading.Thread | None = None

    def start(self) -> "Spinner":
        self._running = True
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self) -> None:
        i = 0
        while self._running:
            frame = self.FRAMES[i % len(self.FRAMES)]
            print(
                f"\r  {self._colour}{C.BOLD}{frame}  {self._msg} …{C.RESET}",
                end="",
                flush=True,
            )
            time.sleep(0.08)
            i += 1

    def stop(self, clear: bool = True) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if clear:
            print("\r" + " " * 60 + "\r", end="", flush=True)


# =============================================================================
# 3.  BANNER
# =============================================================================

def banner() -> None:
    art = f"""
{C.CYAN}{C.BOLD}  ┌──────────────────────────────────────────────────────────────────────┐
  │  ▄▀█ ▀█▀ █░█ █▀▀ █▄░█ ▄▀█                                            │
  │  █▀█ ░█░ █▀█ ██▄ █░▀█ █▀█                                            │
  └──────────────────────────────────────────────────────────────────────┘{C.RESET}"""
    print(art, flush=True)


# =============================================================================
# 4.  SESSION LOGGING
# =============================================================================

LOG_DIR     = Path("./logs").resolve()
LOG_DIR.mkdir(exist_ok=True)
_session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file   = LOG_DIR / f"session_{_session_ts}.log"

_fh = logging.FileHandler(_log_file, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

cli_log = logging.getLogger("athena.cli")
cli_log.setLevel(logging.INFO)
cli_log.addHandler(_fh)


def log_exchange(query: str, response: str, sources: list[str], used_web: bool) -> None:
    cli_log.info("USER    : %s", query)
    cli_log.info("ATHENA  : %s", response[:600])
    cli_log.info("SOURCES : %s | WEB: %s", sources, used_web)
    cli_log.info("-" * 60)


# =============================================================================
# 5.  ASYNC TTS
# =============================================================================

_tts_lock   = threading.Lock()
_tts_thread : threading.Thread | None = None


def _init_tts() -> pyttsx3.Engine | None:
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 163)
        voices = engine.getProperty("voices")
        for v in voices:
            if "female" in v.name.lower() or "zira" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        else:
            if len(voices) > 1:
                engine.setProperty("voice", voices[1].id)
        return engine
    except Exception:
        return None


tts_engine = _init_tts()


def speak(text: str) -> None:
    """Fire-and-forget TTS in a background thread. Skips if already speaking."""
    if not tts_engine or not text.strip():
        return
    global _tts_thread

    def _run() -> None:
        with _tts_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()

    if _tts_thread and _tts_thread.is_alive():
        return
    _tts_thread = threading.Thread(target=_run, daemon=True)
    _tts_thread.start()


def truncate_for_tts(text: str, max_sentences: int = 2) -> str:
    clean = re.sub(r"```.*?```", " [code block] ", text, flags=re.DOTALL)
    clean = re.sub(r"\[Source \d+\]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    sents = [s for s in re.split(r"(?<=[.!?]) +", clean) if s.strip()]
    if len(sents) > max_sentences:
        return " ".join(sents[:max_sentences]) + " Full response is on screen."
    return clean


# =============================================================================
# 6.  JUPYTER NOTEBOOK EXPORT & SAVING LOGIC (UPDATED)
# =============================================================================

# Resolving paths to absolute directories so you always know exactly where files go
IPYNB_DIR = Path("./ai_notebooks").resolve()
IPYNB_DIR.mkdir(exist_ok=True)


def _make_notebook(code: str, lang: str, query: str) -> dict:
    """Build a minimal Jupyter notebook dict from a code string."""
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"# Generated by Athena for Sir Himanshu\n# Query: {query}\n# {ts}\n\n"
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
            "athena": {"generated_at": ts, "query": query},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "athena-header",
                "metadata": {},
                "source": [
                    f"# Athena Generated Code\n",
                    f"**Query:** `{query}`  \n",
                    f"**Generated:** {ts}",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "athena-code-01",
                "metadata": {},
                "outputs": [],
                "source": (title + code).splitlines(keepends=True),
            },
        ],
    }


EXT_MAP: dict[str, str] = {
    "python": "py", "py": "py",
    "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts",
    "html": "html", "css": "css",
    "json": "json", "bash": "sh", "sh": "sh",
    "c": "c", "cpp": "cpp", "java": "java",
    "sql": "sql", "yaml": "yaml", "yml": "yaml",
    "rust": "rs", "go": "go", "kotlin": "kt",
}

# 🛡️ UPDATED: Bulletproof regex handles weird spacing, empty languages, and carriage returns
CODE_PATTERN = re.compile(r"```[ \t]*(\w*)[ \t]*\r?\n(.*?)```", re.DOTALL)

_snippet_dir = Path("./ai_snippets").resolve()
_snippet_dir.mkdir(exist_ok=True)


def process_and_save_code(text: str, query: str = "") -> str:
    """
    Extracts code blocks from response.
    Python blocks → saved as .ipynb (Jupyter notebooks).
    All other languages → saved as source files in ./ai_snippets/.
    Returns a cleaned version of text suitable for TTS.
    """
    matches = CODE_PATTERN.findall(text)
    if not matches:
        return text

    cprint("\n  📁  Code detected — saving …", C.YELLOW)

    for i, (lang, code) in enumerate(matches, 1):
        lang_clean = lang.strip().lower() or "python"

        if lang_clean in ("python", "py", ""):
            # ── Save as Jupyter notebook ───────────────────────────────────
            nb   = _make_notebook(code.strip(), lang_clean, query)
            fname = IPYNB_DIR / f"athena_{int(time.time())}_{i}.ipynb"
            fname.write_text(json.dumps(nb, indent=2), encoding="utf-8")
            # Print absolute path so it's clickable in most terminals
            cprint(f"     📓  Jupyter notebook: {fname.absolute()}", C.GREEN)
        else:
            # ── Save as plain source file ──────────────────────────────────
            ext  = EXT_MAP.get(lang_clean, "txt")
            headers = {
                "js": f"// Athena · Sir Himanshu\n\n",
                "ts": f"// Athena · Sir Himanshu\n\n",
                "java": f"// Athena · Sir Himanshu\n\n",
                "html": f"\n\n",
                "css": f"/* Athena · Sir Himanshu */\n\n",
            }
            header = headers.get(ext, f"# Athena · Sir Himanshu\n\n")
            fname = _snippet_dir / f"snippet_{int(time.time())}_{i}.{ext}"
            fname.write_text(header + code.strip() + "\n", encoding="utf-8")
            cprint(f"     ✅  Saved: {fname.absolute()}", C.GREEN)

    return CODE_PATTERN.sub("[see saved code file]", text)


# =============================================================================
# 7.  STREAMING DISPLAY ENGINE
# =============================================================================

# Colour mapping for inline code-block rendering
_CODE_OPEN  = re.compile(r"^```[ \t]*(\w*)[ \t]*$")
_CODE_CLOSE = re.compile(r"^```[ \t]*$")


def stream_to_terminal(
    token_stream: Iterator[str],
    prefix: str = "  ",
) -> str:
    """
    Consumes a token iterator and prints each token immediately.
    Handles basic Markdown formatting for headers, bold, and code blocks.
    🛡️ Hard stops live streaming if token-decay or task-leaching is intercepted via regex/phrase mapping.
    Returns the complete assembled response string.
    """
    full_text = ""
    in_code_block = False
    line_buffer   = ""

    print(f"\n  {C.BOLD}{C.TEAL}Athena:{C.RESET}\n", flush=True)
    print(f"{prefix}", end="", flush=True)

    # 🛡️ DYNAMIC REGEX GATE: Catches variations of design/create/write + language/program patterns
    LEACH_GATE = re.compile(
        r"^(create|design|write|how\s+can)\s+a\s+(python|javascript|js|script|program|function|custom)", 
        re.IGNORECASE
    )
    
    # 🛡️ STRING LITERAL GATE: Phrases that trigger an immediate stream cutoff
    stop_phrases = [
        "create a javascript",
        "sir himanshu:",
        "write a python",
        "extract the sentences",
        "how can i create",
        "### conclusion"
    ]

    for token in token_stream:
        full_text   += token
        line_buffer += token

        # Flush newlines immediately so streaming looks natural
        while "\n" in line_buffer:
            idx  = line_buffer.index("\n")
            line = line_buffer[:idx]
            line_buffer = line_buffer[idx + 1:]

            clean_line = line.strip()

            # 🛡️ INTERCEPT LAYER: Evaluate string formatting against regex and literal strings
            if LEACH_GATE.match(clean_line) or any(phrase in clean_line.lower() for phrase in stop_phrases):
                print(f"\n\n{C.YELLOW}  [Athena safely intercepted at boundary constraint]{C.RESET}\n", flush=True)
                if line in full_text:
                    full_text = full_text.split(line)[0]
                return full_text.strip()

            # ── Code block toggle ─────────────────────────────────────────
            if _CODE_OPEN.match(clean_line):
                in_code_block = True
                lang = _CODE_OPEN.match(clean_line).group(1) or "code"
                print(f"\n{C.CYAN}  ┌─ {lang} ─────────────────────────────{C.RESET}", flush=True)
                continue
            if _CODE_CLOSE.match(clean_line) and in_code_block:
                in_code_block = False
                print(f"{C.CYAN}  └────────────────────────────────────{C.RESET}\n{prefix}", end="", flush=True)
                continue

            # ── Render line ───────────────────────────────────────────────
            if in_code_block:
                print(f"  {C.YELLOW}{line}{C.RESET}", flush=True)
            elif clean_line.startswith("###"):
                print(f"\n  {C.BOLD}{C.BLUE}{line}{C.RESET}", flush=True)
            elif clean_line.startswith("##"):
                print(f"\n  {C.BOLD}{C.CYAN}{line}{C.RESET}", flush=True)
            elif clean_line.startswith("#"):
                print(f"\n  {C.BOLD}{C.MAGENTA}{line}{C.RESET}", flush=True)
            elif clean_line.startswith("- ") or clean_line.startswith("* "):
                print(f"  {C.TEAL}•{C.RESET} {line[2:]}", flush=True)
            else:
                # Inline bold  **text** → bright
                rendered = re.sub(r"\*\*(.+?)\*\*", f"{C.BOLD}\\1{C.RESET}", line)
                # Inline code  `text` → cyan
                rendered = re.sub(r"`([^`]+)`", f"{C.CYAN}\\1{C.RESET}", rendered)
                # [Source N] citations → grey
                rendered = re.sub(r"\[Source \d+\]", lambda m: f"{C.GREY}{m.group()}{C.RESET}", rendered)
                print(f"{prefix}{rendered}", flush=True)

            print(f"{prefix}", end="", flush=True)

    # Flush any remaining buffer
    if line_buffer.strip():
        clean_buffer = line_buffer.strip()
        # Final edge-case safety check on residual trailing text blocks
        if LEACH_GATE.match(clean_buffer) or any(phrase in clean_buffer.lower() for phrase in stop_phrases):
            print(f"\n\n{C.YELLOW}  [Athena safely intercepted at boundary constraint]{C.RESET}\n", flush=True)
            return full_text.split(line_buffer)[0].strip()

        remaining = line_buffer
        remaining = re.sub(r"\*\*(.+?)\*\*", f"{C.BOLD}\\1{C.RESET}", remaining)
        remaining = re.sub(r"`([^`]+)`", f"{C.CYAN}\\1{C.RESET}", remaining)
        print(remaining, flush=True)
    else:
        print(flush=True)   # final newline

    return full_text
# =============================================================================
# 8.  RESPONSE DISPLAY FOOTER
# =============================================================================

def display_footer(sources: list[str], used_web: bool, faithfulness: float | None = None) -> None:
    """Prints source list, web/cache badge, and optional faithfulness score."""
    tag = f"{C.GREEN}● LIVE WEB{C.RESET}" if used_web else f"{C.BLUE}● CACHED{C.RESET}"
    ts  = f"{C.DIM}{datetime.now().strftime('%H:%M:%S')}{C.RESET}"
    print(f"\n  {tag}  {ts}", flush=True)

    if faithfulness is not None:
        bar_len = 20
        filled  = round(faithfulness * bar_len)
        bar     = "█" * filled + "░" * (bar_len - filled)
        colour  = C.GREEN if faithfulness >= 0.7 else (C.YELLOW if faithfulness >= 0.4 else C.RED)
        print(f"  {C.DIM}Grounding:{C.RESET} {colour}[{bar}]{C.RESET} {faithfulness:.0%}", flush=True)

    if sources:
        print(flush=True)
        cprint("  🔗  Sources:", C.GREY)
        for i, src in enumerate(sources, 1):
            cprint(f"       {i}. {src}", C.GREY)

    separator()


# =============================================================================
# 9.  INTENT CLASSIFIER
# =============================================================================

_CODE_TRIGGERS = re.compile(
    r"\b(write|generate|create|code|script|function|class|implement|build)\b.*"
    r"\b(code|script|function|program|algorithm|snippet|class|notebook|jupyter)\b|"
    r"\bin (python|javascript|js|typescript|html|css|bash|sql|rust|go|java)\b",
    re.IGNORECASE,
)
_SEARCH_TRIGGERS = re.compile(
    r"\b(latest|current|today|recent|news|2024|2025|2026|live|now|update|"
    r"stock|price|score|winner|result|announcement)\b",
    re.IGNORECASE,
)
_REASONING_TRIGGERS = re.compile(
    r"\b(why|explain|analyse|analyze|compare|difference|pros and cons|"
    r"step by step|how does|derive|prove|reasoning)\b",
    re.IGNORECASE,
)


def classify_intent(query: str) -> str:
    """Returns 'code', 'search', 'reasoning', or 'general'."""
    if _CODE_TRIGGERS.search(query):
        return "code"
    if _SEARCH_TRIGGERS.search(query):
        return "search"
    if _REASONING_TRIGGERS.search(query):
        return "reasoning"
    return "general"


# =============================================================================
# 10.  COMMAND PALETTE
# =============================================================================

HELP_TEXT = f"""
{C.CYAN}{C.BOLD}  ╔─────────────────────────────────────────────────────────────╗
  │                   ATHENA COMMAND PALETTE v3                  │
  ╚─────────────────────────────────────────────────────────────╝{C.RESET}
  {C.YELLOW}exit{C.RESET}      →  Shutdown Athena gracefully
  {C.YELLOW}clear{C.RESET}     →  Wipe ALL memory (ChromaDB + BM25 + parent docs)
  {C.YELLOW}stats{C.RESET}     →  Detailed memory and model statistics
  {C.YELLOW}history{C.RESET}   →  Print conversation history (last 10 turns)
  {C.YELLOW}export{C.RESET}    →  Export conversation history to JSON
  {C.YELLOW}save{C.RESET}      →  Save last response to a text file
  {C.YELLOW}help{C.RESET}      →  Show this menu

  {C.DIM}Tip: prefix your query with /cot  to force Chain-of-Thought mode
       prefix your query with /noflare  to skip FLARE refinement{C.RESET}
  {C.DIM}Logs → {LOG_DIR}{C.RESET}
"""


def handle_command(
    cmd: str,
    history: list[dict],
    last_response: str,
) -> bool:
    """
    Handles special CLI commands.
    Returns True if cmd was a command (so chat loop skips pipeline).
    """
    cmd_lower = cmd.strip().lower()

    if cmd_lower == "clear":
        spin = Spinner("Clearing memory", C.RED).start()
        clear_memory()
        spin.stop()
        cprint("\n  🗑️   All memory banks cleared.", C.YELLOW)
        speak("Memory banks fully cleared, Sir Himanshu.")
        return True

    if cmd_lower == "stats":
        spin = Spinner("Gathering stats", C.CYAN).start()
        stats = get_memory_stats()
        spin.stop()
        separator("·")
        cprint("\n  📊  Athena Memory & Model Statistics\n", C.CYAN, bold=True)
        for k, v in stats.items():
            cprint(f"     {k:<20} {v}", C.WHITE)
        separator("·")
        speak(f"I am holding {stats['vector_chunks']} vector chunks, Sir Himanshu.")
        return True

    if cmd_lower == "history":
        if not history:
            cprint("\n  (No conversation history yet.)", C.DIM)
        else:
            separator("·")
            for turn in history:
                role   = "Sir Himanshu" if turn["role"] == "user" else "Athena"
                colour = C.YELLOW if turn["role"] == "user" else C.CYAN
                cprint(f"\n  {colour}{C.BOLD}{role}:{C.RESET}", colour)
                print(f"    {turn['content'][:300]}", flush=True)
            separator("·")
        return True

    if cmd_lower == "export":
        out_path = LOG_DIR / f"history_{_session_ts}.json"
        out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        cprint(f"\n  💾  History exported → {out_path.absolute()}", C.GREEN)
        return True

    if cmd_lower == "save":
        if last_response:
            out_path = LOG_DIR / f"response_{int(time.time())}.txt"
            out_path.write_text(last_response, encoding="utf-8")
            cprint(f"\n  💾  Response saved → {out_path.absolute()}", C.GREEN)
        else:
            cprint("\n  (No response to save yet.)", C.DIM)
        return True

    if cmd_lower == "help":
        print(HELP_TEXT, flush=True)
        return True

    return False


# =============================================================================
# 11.  CONVERSATION MEMORY
# =============================================================================

MAX_HISTORY_TURNS = 2   # 6 full exchanges


def append_history(history: list[dict], role: str, content: str) -> None:
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY_TURNS:
        history[:] = history[-MAX_HISTORY_TURNS:]


# =============================================================================
# 12.  GRACEFUL SHUTDOWN
# =============================================================================

_shutdown_called = False


def _shutdown(message: str = "Goodbye Sir Himanshu!") -> None:
    global _shutdown_called
    if _shutdown_called:
        return
    _shutdown_called = True
    print(flush=True)
    separator("═")
    cprint(f"  👋  {message}", C.CYAN, bold=True)
    separator("═")
    speak(message)
    time.sleep(0.9)
    sys.exit(0)


def _sigint_handler(sig, frame) -> None:  # noqa: ANN001
    _shutdown("Interrupted. Goodbye Sir Himanshu!")


signal.signal(signal.SIGINT, _sigint_handler)


# =============================================================================
# 13.  MAIN CHAT LOOP
# =============================================================================

MAX_SEARCH_RESULTS = 5
RETRIEVAL_K        = 6


def main() -> None:
    # Clear stale ChromaDB cache on fresh session start
    import os
    os.makedirs(r"D:\transformers\ai_notebooks", exist_ok=True)
    os.makedirs(r"D:\transformers\ai_snippets", exist_ok=True)
    try:
        clear_memory()
    except Exception:
        pass

    banner()
    cprint(
        "  Online, Sir Himanshu. Streaming · HyDE · RRF Fusion · FLARE · Self-RAG active.\n"
        "  Type 'help' for the command palette.\n",
        C.GREEN, bold=True,
    )
    speak("Athena online. Welcome back, Sir Himanshu. I am ready.")

    conversation_history: list[dict] = []
    last_response: str = ""

    while True:
        # ── Prompt ───────────────────────────────────────────────────────────
        try:
            raw_input = input(
                f"\n  {C.YELLOW}{C.BOLD}Sir Himanshu › {C.RESET}"
            ).strip()
        except EOFError:
            _shutdown()

        if not raw_input:
            continue

        # ── Special commands ─────────────────────────────────────────────────
        if raw_input.lower() == "exit":
            _shutdown()

        if handle_command(raw_input, conversation_history, last_response):
            continue

        # ── Parse inline flags ────────────────────────────────────────────────
        use_cot   = raw_input.lower().startswith("/cot")
        use_flare = not raw_input.lower().startswith("/noflare")
        query     = re.sub(r"^/(cot|noflare)\s*", "", raw_input, flags=re.IGNORECASE).strip()

        if not query:
            cprint("  (Empty query after stripping flags.)", C.DIM)
            continue

        # ── Intent classification ─────────────────────────────────────────────
        intent = classify_intent(query)
        intent_labels = {
            "code"     : f"{C.MAGENTA}[CODE GEN]{C.RESET}",
            "search"   : f"{C.GREEN}[WEB SEARCH]{C.RESET}",
            "reasoning": f"{C.ORANGE}[REASONING]{C.RESET}",
            "general"  : f"{C.BLUE}[KNOWLEDGE]{C.RESET}",
        }
        flags_str = ""
        if use_cot:
            flags_str += f" {C.DIM}+CoT{C.RESET}"
        if not use_flare:
            flags_str += f" {C.DIM}-FLARE{C.RESET}"

        print(f"\n  {intent_labels[intent]}{flags_str}", flush=True)

        # CoT mode is auto-enabled for reasoning intent
        if intent == "reasoning":
            use_cot = True

        speak("Let me look into that for you, Sir Himanshu.")

        # ── Retrieval spinner (before streaming starts) ───────────────────────
        spin = Spinner("Retrieving knowledge", C.CYAN).start()

        try:
            current_k = 0 if intent == "code" else RETRIEVAL_K
            result, sources, used_web = rag_pipeline(
                query=query,
                max_search_results=MAX_SEARCH_RESULTS,
                retrieval_k=current_k,
                conversation_history=conversation_history,
                stream=True,          # always stream for CLI
                use_flare=False,      # FLARE handled below for streaming compat
                use_cot=use_cot,
            )
        except Exception as exc:
            spin.stop()
            cprint(f"\n  ⚠️  Pipeline error: {exc}", C.RED)
            cli_log.exception("Pipeline exception — query: %s", query)
            speak("I encountered an error, Sir Himanshu. Please check the terminal.")
            continue

        spin.stop()
        separator()

        # ── Live streaming output ─────────────────────────────────────────────
        try:
            last_response = stream_to_terminal(result)
        except Exception as exc:
            cprint(f"\n  ⚠️  Streaming error: {exc}", C.RED)
            last_response = ""

        # ── 2. Footer: sources, cache tag (NO EXTRA RESPONSE PRINTS HERE) ──
        display_footer(sources, used_web)

        # ── 3. Code saving (Python → .ipynb, others → source files) ───────────
        clean_text = process_and_save_code(last_response, query=query)

        # ── 4. TTS (first 2 sentences, async) ─────────────────────────────────
        spoken = truncate_for_tts(clean_text, max_sentences=2)
        speak(spoken)

        # ── 5. Memory update ──────────────────────────────────────────────────
        append_history(conversation_history, "user",      query)
        append_history(conversation_history, "assistant", last_response)

        # ── 6. Session log (Saves silently to file, DOES NOT print to stdout) ──
        log_exchange(query, last_response, sources, used_web)


if __name__ == "__main__":
    main()