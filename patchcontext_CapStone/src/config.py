"""
PatchContext configuration — all tunables in one place.

Adjust scoping limits, cleaning rules, model choices, and retriever
parameters here. Every other module imports from this file.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional in non-Streamlit contexts
    st = None


def _get_setting(name: str, default: str = "") -> str:
    """Resolve a setting from environment variables, Streamlit secrets, or .env."""
    value = os.getenv(name, "")
    if value:
        return value

    if st is not None:
        try:
            secret_value = st.secrets.get(name, "")
            if secret_value:
                return str(secret_value)
        except Exception:
            pass

    return default

# ---------------------------------------------------------------------------
# Fix Windows console encoding (cp1252 can't handle emoji in print())
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass  # Fallback: some environments don't support reconfigure

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GITHUB_TOKEN = _get_setting("GITHUB_TOKEN", "")
OPENAI_API_KEY = _get_setting("OPENAI_API_KEY", "")  # kept for optional OpenAI use
GOOGLE_API_KEY = _get_setting("GOOGLE_API_KEY", "")
GOOGLE_API_KEY_APP = _get_setting("GOOGLE_API_KEY_APP", _get_setting("GOOGLE_API_KEY", ""))
GOOGLE_API_KEY_EVAL = _get_setting("GOOGLE_API_KEY_EVAL", _get_setting("GOOGLE_API_KEY", ""))
GROQ_API_KEY = _get_setting("GROQ_API_KEY", "")
GROQ_API_KEY_EVAL = _get_setting("GROQ_API_KEY_EVAL", _get_setting("GROQ_API_KEY", ""))

# ---------------------------------------------------------------------------
# Target repository
# ---------------------------------------------------------------------------
REPO_OWNER = "fastapi"
REPO_NAME = "fastapi"
REPO_FULL = f"{REPO_OWNER}/{REPO_NAME}"
REPO_URL = f"https://github.com/{REPO_FULL}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"

RAW_PRS_PATH = DATA_DIR / "raw_prs.json"
RAW_ISSUES_PATH = DATA_DIR / "raw_issues.json"
RAW_COMMITS_PATH = DATA_DIR / "raw_commits.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
BENCHMARK_TEMPLATE_PATH = DATA_DIR / "benchmark_questions_template.json"
BENCHMARK_PATH = DATA_DIR / "benchmark_questions.json"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.json"

# ---------------------------------------------------------------------------
# Scoping limits  (§11 — set to None = unlimited; re-run extract.py overnight)
# ---------------------------------------------------------------------------
# FastAPI has ~16,000+ PRs, ~10,000+ issues, ~5,000+ commits going back to 2018.
# None means fetch everything the API returns (pagination follows all Link headers).
# Re-extraction with None limits takes 4-8h depending on GitHub rate limits.
MAX_PRS = None       # Was 400 — None = fetch all merged PRs (newest + oldest)
MAX_ISSUES = None    # Was 300 — None = fetch all closed issues
MAX_COMMITS = None   # Was 500 — None = fetch all non-merge commits

# Set True to re-fetch even if raw_*.json files already exist
EXTRACT_FORCE_REFRESH = False

# ---------------------------------------------------------------------------
# Bot authors to exclude
# ---------------------------------------------------------------------------
BOT_AUTHORS = frozenset({
    "dependabot[bot]",
    "github-actions[bot]",
    "codecov[bot]",
    "pre-commit-ci[bot]",
})

# ---------------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------------
MIN_CHUNK_LENGTH = 30  # Characters after cleaning; shorter → dropped

# HTML comment pattern (PR-template boilerplate)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Collapse triple-plus blank lines to one
MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Low-signal patterns — acknowledgement-only content
LOW_SIGNAL_PATTERNS = [
    re.compile(r"^\s*(\+1|👍|lgtm|LGTM|thanks!?|thank you!?|🎉|🚀|❤️|💯)\s*$", re.IGNORECASE),
    re.compile(r"^\s*:[\w+-]+:\s*$"),            # emoji-only  :thumbsup:
    re.compile(r"^\s*(great|awesome|nice)!?\s*$", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Retrieval (§6.1, §6.3, §6.4)
# ---------------------------------------------------------------------------
RETRIEVER_SEARCH_TYPE = "mmr"
RETRIEVER_K = 6
RETRIEVER_FETCH_K = 15
RETRIEVER_LAMBDA_MULT = 0.7  # 0=diverse, 1=most similar (standard=0.5)

# Chunks that are structurally valid but semantically too generic to be
# useful for design-rationale retrieval. Add any chunk id that appears
# repeatedly as off-topic filler across unrelated queries.
RETRIEVAL_BLOCKLIST = frozenset({
    "PR#16012",   # "Release version 0.139.1 - Prepare release." — no content
    "PR#16011",   # Empty PR template body, dotted-path bug fix
})

# ---------------------------------------------------------------------------
# Generation  (§7)
# ---------------------------------------------------------------------------
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0                           # Deterministic
HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"      # Local HuggingFace embeddings (free)

# ---------------------------------------------------------------------------
# Corpus knowledge bounds
# ---------------------------------------------------------------------------
# These constants describe the actual date range of the currently-indexed data.
# They are injected into the generation prompt so the LLM never claims to know
# about items outside the indexed window.
#
# After running extract.py with MAX_PRS/MAX_ISSUES/MAX_COMMITS = None,
# update these values to reflect the newly built corpus.
CORPUS_NEWEST_DATE = "2026-07-16"  # Latest non-bot activity indexed
CORPUS_OLDEST_DATE = "2018-12-05"  # Earliest item date present in any raw JSON
CORPUS_OLDEST_COMMIT = "2018-12-05"  # Oldest COMMIT in the index
CORPUS_COMMITS_PARTIAL = False  # Full history is now present

# Backwards-compat alias used by the sidebar
CORPUS_CUTOFF_DATE = CORPUS_NEWEST_DATE

# ---------------------------------------------------------------------------
# Hallucination guard  (§9)
# ---------------------------------------------------------------------------
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_THRESHOLD = 0.35        # Entailment score above this → VERIFIED
MAX_TOTAL_CONTEXT = 5000  # Character limit to prevent NLI silent truncation

# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------
GITHUB_API_BASE = "https://api.github.com"
GITHUB_PER_PAGE = 100
GITHUB_RATE_LIMIT_BUFFER = 50  # Sleep when remaining < this

COMMENT_BATCH_SIZE = 4
MAX_CHUNKS_PER_THREAD = 10
