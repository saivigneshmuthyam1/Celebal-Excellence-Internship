"""
PatchContext — Streamlit UI

A polished interface for querying FastAPI design rationale with
RAG-grounded answers, clickable citations, and per-claim NLI
hallucination verification.

Launch:
    cd patchcontext
    streamlit run ui/app.py
"""

import json
import re
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — add src/ to sys.path so we can import project modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    FAISS_INDEX_DIR,
    CHUNKS_PATH,
    CORPUS_COMMITS_PARTIAL,
    CORPUS_CUTOFF_DATE,
    CORPUS_NEWEST_DATE,
    CORPUS_OLDEST_COMMIT,
    CORPUS_OLDEST_DATE,
    LLM_MODEL,
    RETRIEVER_K,
    RETRIEVER_SEARCH_TYPE,
    NLI_MODEL,
    NLI_THRESHOLD,
    REPO_FULL,
    REPO_URL,
)


# ---------------------------------------------------------------------------
# Helper functions (must be defined before use in Streamlit's linear flow)
# ---------------------------------------------------------------------------

def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _format_answer_html(answer: str) -> str:
    """
    Convert markdown-style citation links [tag](url) to HTML <a> tags,
    and handle basic markdown formatting for the answer card.

    Process order: extract links first → escape remaining text → reassemble.
    This avoids HTML-escaping URLs or breaking link regex patterns.
    """
    # Step 1: Extract markdown links and replace with placeholders
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    links = []

    def _capture_link(match):
        idx = len(links)
        links.append((match.group(1), match.group(2)))
        return f"__LINK_PLACEHOLDER_{idx}__"

    text = link_pattern.sub(_capture_link, answer)

    # Step 2: Escape HTML on the non-link text
    text = _escape_html(text)

    # Step 3: Restore links as HTML <a> tags (safe — URLs are from our own resolver)
    for i, (label, url) in enumerate(links):
        safe_label = _escape_html(label)
        text = text.replace(
            f"__LINK_PLACEHOLDER_{i}__",
            f'<a href="{url}" target="_blank" rel="noopener">{safe_label}</a>',
        )

    # Step 4: Convert remaining [tag] references (unresolved) to styled spans
    text = re.sub(
        r'\[(PR#\d+|issue#\d+|commit:[a-f0-9]{7,}|README)\]',
        r'<span style="color: #f59e0b; font-weight: 500;">[\1]</span>',
        text,
    )

    # Paragraphs
    text = text.replace("\n\n", "</p><p>")
    text = text.replace("\n", "<br>")
    text = f"<p>{text}</p>"

    return text


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
LOGO_PATH = str(Path(__file__).parent / "logo.png")

st.set_page_config(
    page_title="PatchContext — FastAPI Design Rationale",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set the sidebar logo
try:
    st.logo(LOGO_PATH)
except AttributeError:
    # st.logo is available in Streamlit >= 1.35.0
    pass

# ---------------------------------------------------------------------------
# Custom CSS for a premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Header ---------- */
    .hero-container {
        background: #2f2f2f; /* ChatGPT secondary background */
        border-radius: 12px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid #444;
    }
    @keyframes shimmer {
        0% { transform: translateX(-5%) translateY(-5%); }
        100% { transform: translateX(5%) translateY(5%); }
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: rgba(255, 255, 255, 0.7);
        margin: 0;
        position: relative;
        z-index: 1;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
    }

    /* ---------- Answer Card ---------- */
    .answer-card {
        background: #2f2f2f;
        border: 1px solid #444;
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1rem 0;
        line-height: 1.75;
        color: #ececec;
        font-size: 0.95rem;
    }
    .answer-card a {
        color: #10A37F;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .answer-card a:hover {
        color: #1a7f64;
        text-decoration: underline;
    }

    /* ---------- Verification badges ---------- */
    .claim-card {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        border-left: 4px solid;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .claim-verified {
        background: rgba(16, 185, 129, 0.08);
        border-left-color: #10b981;
        color: #d1fae5;
    }
    .claim-contradiction {
        background: rgba(239, 68, 68, 0.08);
        border-left-color: #ef4444;
        color: #fecaca;
    }
    .claim-unsupported {
        background: rgba(245, 158, 11, 0.08);
        border-left-color: #f59e0b;
        color: #fde68a;
    }
    .claim-unverified {
        background: rgba(107, 114, 128, 0.08);
        border-left-color: #6b7280;
        color: #d1d5db;
    }
    .claim-status {
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }
    .claim-text {
        font-size: 0.88rem;
        opacity: 0.9;
    }
    .claim-scores {
        font-size: 0.78rem;
        opacity: 0.6;
        margin-top: 0.5rem;
        font-family: 'Courier New', monospace;
    }

    /* ---------- Source cards ---------- */
    .source-card {
        background: rgba(30, 30, 50, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        transition: border-color 0.2s ease;
    }
    .source-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }
    .source-id {
        font-weight: 600;
        color: #818cf8;
        font-size: 0.9rem;
    }
    .source-meta {
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }
    .source-preview {
        color: rgba(255, 255, 255, 0.65);
        font-size: 0.83rem;
        margin-top: 0.5rem;
        line-height: 1.5;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        padding-top: 0.5rem;
    }

    /* ---------- Stats row ---------- */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .stat-pill {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.82rem;
        color: #a5b4fc;
    }
    .stat-pill strong {
        color: #c7d2fe;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1;
    }

    /* ---------- Misc ---------- */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.5rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Load models (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_rag():
    """
    Load RAG chain (cached across reruns).
    NOTE: Because `.streamlit/config.toml` sets `fileWatcherType = "none"`, 
    changes to `rag_chain.py` or other backend files will not take effect 
    until the Streamlit server process is fully killed and restarted!
    """
    from rag_chain import PatchContextRAG
    return PatchContextRAG()


@st.cache_resource(show_spinner=False)
def load_guard():
    """Load NLI hallucination guard (cached across reruns)."""
    from hallucination_guard import HallucinationGuard
    return HallucinationGuard()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    commit_coverage = (
        f'<span style="color: #f87171;">{CORPUS_OLDEST_COMMIT} – {CORPUS_NEWEST_DATE}</span> '
        f'<span style="color: #f59e0b; font-size:0.75rem;">(incomplete — re-run extract.py)</span>'
    ) if CORPUS_COMMITS_PARTIAL else (
        f'<span style="color: #34d399;">{CORPUS_OLDEST_DATE} – {CORPUS_NEWEST_DATE}</span>'
    )
    st.markdown(f"""
    <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.8;">
        <strong>Repository:</strong> <a href="{REPO_URL}" target="_blank" style="color: #818cf8;">{REPO_FULL}</a><br>
        <strong>LLM:</strong> {LLM_MODEL}<br>
        <strong>Retrieval:</strong> {RETRIEVER_SEARCH_TYPE.upper()}, k={RETRIEVER_K}<br>
        <strong>NLI Model:</strong> {NLI_MODEL.split('/')[-1]}<br>
        <strong>NLI Threshold:</strong> {NLI_THRESHOLD}<br>
        <strong>Issues/PRs:</strong> <span style="color: #fbbf24;">{CORPUS_OLDEST_DATE} – {CORPUS_NEWEST_DATE}</span><br>
        <strong>Commits:</strong> {commit_coverage}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    run_guard = st.toggle("🔬 Enable hallucination guard", value=True,
                          help="Run NLI verification on each claim. Adds ~5s per query.")

    st.markdown("---")

    # Pipeline readiness check
    st.markdown("### 📋 Pipeline Status")
    faiss_ready = FAISS_INDEX_DIR.exists() and any(FAISS_INDEX_DIR.iterdir()) if FAISS_INDEX_DIR.exists() else False
    chunks_ready = CHUNKS_PATH.exists()

    if faiss_ready and chunks_ready:
        st.success("Index & chunks loaded ✓", icon="✅")
    else:
        if not chunks_ready:
            st.error("chunks.json missing — run chunk.py first", icon="❌")
        if not faiss_ready:
            st.error("FAISS index missing — run build_index.py first", icon="❌")
        st.info("Run Phases 1–3 before querying.", icon="💡")

    st.markdown("---")

    # History
    if st.session_state.history:
        st.markdown("### 📜 Recent Questions")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            if st.button(f"↩ {h['question'][:45]}...", key=f"hist_{i}", use_container_width=True):
                st.session_state.replay_question = h["question"]
                st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #475569; text-align: center; padding: 0.5rem;">
        PatchContext v1.0<br>
        Built for Celebal Capstone
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔍 PatchContext</div>
    <div class="hero-subtitle">
        Ask <em>"why was this designed this way?"</em> about FastAPI — get answers
        grounded in real commits, PRs, and issue discussions with clickable citations
        and automated hallucination verification.
    </div>
    <span class="hero-badge">RAG + NLI · FastAPI Repo</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------
example_questions = [
    "Why does FastAPI use Depends() for dependency injection?",
    "Why is FastAPI built on top of Starlette?",
    "Why does FastAPI return 422 instead of 400 for validation errors?",
    "Why does FastAPI use Pydantic for validation?",
    "What was the motivation for the lifespan context manager?",
]

# Handle replay from sidebar history
default_value = st.session_state.pop("replay_question", "")

with st.form("search_form", border=False):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "Ask a design-rationale question about FastAPI",
            value=default_value,
            placeholder="e.g., Why does FastAPI use Depends() for dependency injection?",
            label_visibility="collapsed",
        )
    with col_btn:
        ask_clicked = st.form_submit_button("🚀 Ask", use_container_width=True, type="primary")

# Example question chips
st.markdown('<div style="margin: -0.5rem 0 1rem 0;">', unsafe_allow_html=True)
chip_cols = st.columns(len(example_questions))
for i, eq in enumerate(example_questions):
    with chip_cols[i]:
        if st.button(eq[:35] + "…", key=f"ex_{i}", use_container_width=True):
            st.session_state.replay_question = eq
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main query logic
# ---------------------------------------------------------------------------
if (ask_clicked or default_value) and question.strip():
    # Check pipeline readiness
    if not (FAISS_INDEX_DIR.exists() and CHUNKS_PATH.exists()):
        st.error(
            "⚠️ Pipeline data not found. Please run Phases 1–3 first:\n\n"
            "```bash\ncd src\npython extract.py\npython chunk.py\npython build_index.py\n```"
        )
        st.stop()

    # ----- Run RAG chain -----
    with st.status("🔗 Running RAG pipeline...", expanded=True) as status:
        st.write("Loading FAISS index & embedding model...")
        t0 = time.time()

        try:
            rag = load_rag()
        except Exception as e:
            st.error(f"Failed to load RAG chain: {e}")
            st.stop()

        st.write(f"Retrieving relevant chunks ({RETRIEVER_SEARCH_TYPE.upper()}, k={RETRIEVER_K})...")
        st.write(f"Generating answer with {LLM_MODEL}...")

        try:
            result = rag.query(question)
        except Exception as e:
            error_text = str(e)
            if "429" in error_text or "quota" in error_text.lower() or "ResourceExhausted" in error_text:
                st.warning(
                    "⏳ PatchContext has hit its daily API request limit on the free tier. "
                    "This resets at midnight Pacific Time. Please try again later."
                )
            else:
                st.error("Something went wrong answering this question. Please try again.")
            st.stop()

        rag_time = time.time() - t0
        st.write(f"✅ Answer generated in {rag_time:.1f}s")

        # ----- Run hallucination guard -----
        verifications = None
        guard_time = 0
        if run_guard:
            if result.get("skip_guard"):
                st.write("🔬 Hallucination guard skipped (direct lookup / no generation).")
            else:
                st.write("🔬 Running NLI hallucination guard...")
                t1 = time.time()
                try:
                    guard = load_guard()
                    verifications = guard.check_from_rag_result(result)
                except Exception as e:
                    error_text = str(e)
                    if "429" in error_text or "quota" in error_text.lower() or "ResourceExhausted" in error_text:
                        st.warning("⏳ Hallucination guard skipped: API daily quota reached.")
                    else:
                        import traceback
                        traceback.print_exc()
                        st.warning("Hallucination guard encountered an error and was skipped.")
                    verifications = None
                guard_time = time.time() - t1
                if verifications:
                    st.write(f"✅ {len(verifications)} claims verified in {guard_time:.1f}s")

        total_time = rag_time + guard_time
        status.update(label=f"✅ Complete in {total_time:.1f}s", state="complete")

    # Save to history
    st.session_state.history.append({
        "question": question,
        "result": result,
        "verifications": verifications,
        "rag_time": rag_time,
        "guard_time": guard_time,
    })

    # ----- Stats pills -----
    stats_html = '<div class="stats-container">'
    stats_html += f'<div class="stat-pill">⏱ <strong>{total_time:.1f}s</strong> total</div>'
    stats_html += f'<div class="stat-pill">📚 <strong>{len(result["sources"])}</strong> sources retrieved</div>'
    stats_html += f'<div class="stat-pill">🏷 <strong>{len(result["citations"])}</strong> citations</div>'
    if verifications:
        v_count = sum(1 for v in verifications if v["status"] == "VERIFIED")
        stats_html += f'<div class="stat-pill">✅ <strong>{v_count}/{len(verifications)}</strong> claims verified</div>'
    stats_html += '</div>'
    st.markdown(stats_html, unsafe_allow_html=True)

    # ----- Answer display -----
    if result.get("confidence_gate_triggered"):
        st.info(
            f"🔍 Retrieval confidence too low (avg distance: "
            f"{result.get('avg_relevance_score', '?')}). "
            f"The corpus likely does not contain relevant information for this query."
        )

    st.markdown('<div class="section-header">📝 Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-card">{_format_answer_html(result["answer"])}</div>',
                unsafe_allow_html=True)

    # ----- Verification results -----
    if result.get("forbidden_citation_format"):
        st.warning(
            "⚠️ This answer contains citations in an unsupported format "
            "(e.g. [Source: PR#35] or documentation links). These cannot be "
            "verified by the hallucination guard. Treat this answer with extra caution."
        )

    if verifications:
        st.markdown('<div class="section-header">🔬 Per-Claim Verification</div>', unsafe_allow_html=True)

        status_config = {
            "VERIFIED": ("✅ Verified", "claim-verified"),
            "FLAGGED_CONTRADICTION": ("❌ Contradiction", "claim-contradiction"),
            "FLAGGED_UNSUPPORTED": ("⚠️ Unsupported", "claim-unsupported"),
            "UNVERIFIED": ("❓ Unverified", "claim-unverified"),
        }

        for v in verifications:
            label, css_class = status_config.get(v["status"], ("? Unknown", "claim-unverified"))

            scores_html = ""
            if v.get("scores"):
                for tag, s in v["scores"].items():
                    scores_html += (
                        f'<div class="claim-scores">'
                        f'{tag}: entailment={s["entailment"]:.3f} · '
                        f'contradiction={s["contradiction"]:.3f} · '
                        f'neutral={s["neutral"]:.3f}'
                        f'</div>'
                    )

            citations_html = ""
            if v.get("citations"):
                citations_html = f'<div class="claim-scores">Citations: {", ".join(v["citations"])}</div>'

            st.markdown(f"""
            <div class="claim-card {css_class}">
                <div class="claim-status">{label}</div>
                <div class="claim-text">{_escape_html(v["sentence"][:300])}</div>
                {citations_html}
                {scores_html}
            </div>
            """, unsafe_allow_html=True)

        # Verification summary bar
        counts = {"VERIFIED": 0, "FLAGGED_CONTRADICTION": 0, "FLAGGED_UNSUPPORTED": 0, "UNVERIFIED": 0}
        for v in verifications:
            counts[v["status"]] = counts.get(v["status"], 0) + 1

        total_claims = len(verifications)
        if total_claims > 0:
            cols = st.columns(4)
            emoji_map = {"VERIFIED": "✅", "FLAGGED_CONTRADICTION": "❌",
                         "FLAGGED_UNSUPPORTED": "⚠️", "UNVERIFIED": "❓"}
            for col, (status_key, count) in zip(cols, counts.items()):
                with col:
                    pct = (count / total_claims) * 100
                    st.metric(
                        label=f"{emoji_map[status_key]} {status_key.replace('FLAGGED_', '').title()}",
                        value=f"{count}",
                        delta=f"{pct:.0f}%",
                    )

    # ----- Retrieved sources -----
    if result["sources"]:
        st.markdown('<div class="section-header">📚 Retrieved Sources</div>', unsafe_allow_html=True)

        for src in result["sources"]:
            with st.expander(f"**{src['id']}** — {src['source_type']}  ·  {src['date']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Author:** {src['author']}")
                    if src.get("labels"):
                        labels_str = " ".join(f"`{lbl}`" for lbl in src["labels"])
                        st.markdown(f"**Labels:** {labels_str}")
                    if "similarity_score" in src:
                        # L2 distance (lower is better)
                        if src["similarity_score"] is None:
                            st.markdown("**Relevance Distance:** `N/A (keyword match)`")
                        else:
                            st.markdown(f"**Relevance Distance:** `{src['similarity_score']:.3f}`")
                with col2:
                    st.link_button("Open on GitHub ↗", src["url"], use_container_width=True)

                st.markdown("---")
                st.markdown(f"```\n{src['text_preview']}\n```")

    # ----- Raw JSON (debug) -----
    with st.expander("🔧 Raw JSON Response"):
        st.json(json.loads(json.dumps(result, default=str)))


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
elif not ask_clicked:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #64748b;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">💡</div>
        <div style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">
            Ask a design-rationale question about FastAPI
        </div>
        <div style="font-size: 0.9rem; max-width: 500px; margin: 0 auto; line-height: 1.6;">
            PatchContext retrieves real commits, PRs, and issues from the FastAPI
            repository to explain <em>why</em> things were designed the way they are.
            Every claim is cited and optionally verified with NLI.
        </div>
    </div>
    """, unsafe_allow_html=True)
