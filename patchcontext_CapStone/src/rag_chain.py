"""
Phase 4 — RAG Chain (MMR Retrieval + LCEL Generation + Citation Resolution)

Loads the FAISS index, retrieves with MMR, generates answers with
Gemini 2.5 Flash, and resolves citation tags to clickable GitHub URLs.

Usage (CLI sanity check):
    python rag_chain.py "Why does FastAPI use Depends() for dependency injection?"
"""

import json
import re
import pickle
import sys

import numpy as np
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import (
    CHUNKS_PATH,
    CORPUS_COMMITS_PARTIAL,
    CORPUS_CUTOFF_DATE,
    CORPUS_NEWEST_DATE,
    CORPUS_OLDEST_COMMIT,
    CORPUS_OLDEST_DATE,
    HF_EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    BM25_INDEX_PATH,
    GROQ_API_KEY_EVAL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    RETRIEVER_LAMBDA_MULT,
    RETRIEVER_SEARCH_TYPE,
    MAX_TOTAL_CONTEXT,
)



# ---------------------------------------------------------------------------
# Canonical no-answer string — used by both the confidence-gate early return
# and the generation prompt, so both refusal paths produce identical text.
# ---------------------------------------------------------------------------
NO_ANSWER_TEXT = "The provided corpus has no grounded answer for this."

# ---------------------------------------------------------------------------
# Citation regex — matches [PR#123], [issue#456], [commit:abc1234]
# ---------------------------------------------------------------------------
CITATION_PATTERN = re.compile(
    r'\[(PR#\d+|issue#\d+|commit:[a-f0-9]{7,}|README)\]'
)

# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------
def _make_prompt(
    corpus_newest: str,
    corpus_oldest: str,
    corpus_oldest_commit: str,
    commits_partial: bool,
) -> ChatPromptTemplate:
    """Build the generation prompt."""
    return ChatPromptTemplate.from_template(
        """You are PatchContext, an expert on the FastAPI framework's design history.
Your job is to answer questions strictly using the context chunks provided below.

WHEN TO REFUSE (ABSOLUTE CRITICAL PRIORITY):
If the provided context chunks do NOT contain the specific factual rationale or explanation required to answer the user's specific question, you MUST refuse.
Do NOT use your prior knowledge to answer the question under any circumstances.
If the chunks merely mention the topic (e.g., they are bug reports, edge cases, or unrelated PRs about a feature) but do not actually explain the design rationale or mechanism asked by the question, you must output EXACTLY and ONLY this phrase: "The provided corpus has no grounded answer for this."
If the user asks WHY a design decision was made, and the retrieved documents only describe WHAT changed (like a commit message or changelog) without explaining the underlying rationale, you MUST reply: 'The provided corpus has no grounded answer for this.' DO NOT summarize the changes.
Never invent a rationale or explanation. Never attach a barely-related citation to a hallucinated fact.

CORE RULE: Your primary goal is to extract facts from the context. Do NOT combine facts if they are from completely unrelated topics.

CITATION FORMAT — YOU MUST FOLLOW THIS EXACTLY:
Every factual claim must end with a citation in one of these four formats ONLY:
  [PR#123]       — for pull requests
  [issue#456]    — for issues
  [commit:abc1234] — for commits (7+ hex chars)
  [README]       — for documentation from the repository README

FORBIDDEN — never use these formats:
  commit:abc1234          ← forbidden (missing brackets)
  [Source: PR#35 | ...]   ← forbidden
  [Source: commit:abc]    ← forbidden
  (https://...)           ← forbidden as a citation
  "Sources:" section      ← forbidden
  Documentation links     ← forbidden

Only cite source IDs that appear in the context headers above.
If a claim cannot be cited with one of the four formats above, do not make it.
You do NOT need a single chunk that states the complete answer — synthesis is required.

TONE & STYLE RULES (CRITICAL):
- Be strictly extractive and highly factual.
- Do NOT act like a storyteller. Do NOT use transitional or narrative fluff (e.g., "The evolution of this feature...", "Overall, while the initial implementation...").
- Every single sentence you write MUST be directly traceable to a fact in the context.
- Match the terminology of the source text where possible, avoiding overly creative paraphrasing.
- Do not attribute quotes, thoughts, or feelings to a user or maintainer unless explicitly stated in the context.
- Never use global comparative framing or meta-phrasing (e.g., do NOT say "The last commit was...", "The initial design was...").
- State the raw fact directly as anchored to the citation identifier. 
- Example: Instead of "The last commit was commit 7fe315c by tiangolo on [date]", write "Commit 7fe315c was made by tiangolo on [date]."

EXPLOIT & GROUNDING RULES:
- Do not interpret or extrapolate structural associations. If an issue number appears in parentheses next to a commit message (e.g., "Message text (#1234)"), state it exactly as it appears. 
- Example: Write "Commit 7fe315c includes the text (#16013)" instead of "The commit was made as part of issue #16013."
- Every sentence containing extracted information MUST terminate with exactly one clean bracket identifier, containing only the ID string (e.g., [commit:7fe315c]). Do not append raw text strings like "commit:7fe315c" without brackets.
- When synthesizing across multiple sources, each individual factual claim must be traceable to something explicitly present in the specific source(s) you cite for that claim — not merely thematically related. Do not attach a citation to a sentence unless that source's actual content supports that specific sentence. If you are not confident a specific detail is stated in the context, omit that detail rather than stating it with an unsupported citation.

Context:
{context}

Question: {question}

Answer (cite every claim):"""
    )


PROMPT = _make_prompt(
    corpus_newest=CORPUS_NEWEST_DATE,
    corpus_oldest=CORPUS_OLDEST_DATE,
    corpus_oldest_commit=CORPUS_OLDEST_COMMIT,
    commits_partial=CORPUS_COMMITS_PARTIAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_docs(docs) -> str:
    """Format retrieved documents into a context string with source tags."""
    parts = []
    for doc in docs:
        meta = doc.metadata
        chunk_id = meta.get("id", "")
        source_type = meta.get("source_type", "")
        author = meta.get("author", "unknown")
        date = (meta.get("date", "") or "")[:10]  # YYYY-MM-DD
        header = f"[Source: {chunk_id} | type={source_type} | author={author} | date={date}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_id_url_map() -> dict[str, str]:
    """Build {id -> url} map from all chunks for citation resolution."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    return {
        c["metadata"]["id"]: c["metadata"]["url"]
        for c in chunks
        if "id" in c["metadata"] and "url" in c["metadata"]
    }


def resolve_citations(text: str, id_url_map: dict[str, str]) -> str:
    """
    Replace [PR#123] with [PR#123](https://github.com/...) markdown links.
    Tags that can't be resolved are left unchanged (shouldn't happen if the
    model followed its instructions, but handled gracefully).
    """
    def _replace(match: re.Match) -> str:
        tag = match.group(1)
        url = id_url_map.get(tag, "")
        if url:
            return f"[{tag}]({url})"
        return match.group(0)  # Leave unresolvable tags alone

    return CITATION_PATTERN.sub(_replace, text)


def load_retriever():
    """
    Load FAISS index and BM25 index, return an EnsembleRetriever (Hybrid Search).
    Called by other modules (evaluate.py, hallucination_guard.py, benchmark_helper.py).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # Safe: we built this index ourselves
    )
    faiss_retriever = vectorstore.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={
            "k": RETRIEVER_FETCH_K,
            "fetch_k": RETRIEVER_FETCH_K * 2,
            "lambda_mult": RETRIEVER_LAMBDA_MULT,
        },
    )
    
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = RETRIEVER_FETCH_K
    
    # 0.4 FAISS, 0.6 BM25 based on codebase skew preference
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.4, 0.6]
    )
    return ensemble_retriever


def source_docs_to_metadata(docs):
    return [
        {
            "id": d.metadata.get("id", ""),
            "source_type": d.metadata.get("source_type", ""),
            "url": d.metadata.get("url", ""),
            "author": d.metadata.get("author", ""),
            "date": (d.metadata.get("date", "") or "")[:10],
            "text_preview": d.page_content[:300],
            "labels": d.metadata.get("labels", []),
            "similarity_score": None,
        }
        for d in docs
    ]


# ---------------------------------------------------------------------------
# Positional Query Routing
# ---------------------------------------------------------------------------

POSITIONAL_QUERY_PATTERNS = [
    (re.compile(r'\b(first|earliest|oldest)\b.*\bcommit', re.IGNORECASE), "first"),
    (re.compile(r'\b(last|latest|newest|most recent)\b.*\bcommit', re.IGNORECASE), "last"),
]

def _try_positional_commit_answer(question: str) -> dict | None:
    """Handle 'first/last commit' queries via direct metadata lookup instead of embedding search."""
    for pattern, which in POSITIONAL_QUERY_PATTERNS:
        if pattern.search(question):
            with open(CHUNKS_PATH, encoding="utf-8") as f:
                chunks = json.load(f)
            commit_chunks = [c for c in chunks if c["metadata"].get("source_type") == "commit"]
            if not commit_chunks:
                return None
            commit_chunks.sort(key=lambda c: c["metadata"].get("date", ""))
            target = commit_chunks[0] if which == "first" else commit_chunks[-1]
            meta = target["metadata"]
            clean_text = target['text'][:200].replace('\n', ' ').strip()
            answer = (
                f"The {which} commit in the indexed corpus is {meta.get('id')}, "
                f"by {meta.get('author', 'unknown')} on {meta.get('date', 'unknown date')[:10]}: "
                f"{clean_text}"
            )
            return {
                "question": question, "raw_answer": answer, "answer": answer,
                "sources": [{"id": meta.get("id"), "source_type": "commit",
                             "url": meta.get("url", ""), "author": meta.get("author", ""),
                             "date": meta.get("date", "")[:10], "text_preview": target["text"][:300],
                             "labels": [], "similarity_score": None}],
                "citations": {meta.get("id"): meta.get("url", "")},
                "retrieved_source_texts": {meta.get("id"): target["text"]},
                "skip_guard": True,
            }
    return None

# ---------------------------------------------------------------------------
# Main RAG class
# ---------------------------------------------------------------------------

class PatchContextRAG:
    """
    LCEL chain: MMR retrieval → Gemini 2.5 Flash generation → citation resolution.

    Architecture (per spec §7):
        RunnableParallel(context=retriever | format_docs, question=passthrough)
        | PROMPT
        | ChatGoogleGenerativeAI(gemini-2.5-flash, temperature=0)
        | StrOutputParser()
    """

    def __init__(self):
        self.retriever = load_retriever()
        self.id_url_map = _build_id_url_map()
        self.llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=GROQ_API_KEY_EVAL)

    def query(self, question: str) -> dict:
        """
        Run the full RAG pipeline.

        Returns:
            {
                "question": str,
                "raw_answer": str,        # answer with [PR#n] style tags
                "answer": str,            # answer with resolved clickable links
                "sources": list[dict],    # retrieved chunk metadata
                "citations": dict,        # {tag: url} for all cited tags
            }
        """
        # --- Query Routing: Positional / Aggregation check ---
        positional_result = _try_positional_commit_answer(question)
        if positional_result:
            return positional_result

        # Retrieve source docs separately to capture metadata
        source_docs = self.retriever.invoke(question)

        from config import RETRIEVAL_BLOCKLIST
        source_docs = [d for d in source_docs
                       if d.metadata.get("id") not in RETRIEVAL_BLOCKLIST]
        if not source_docs:
            return {
                "question": question,
                "raw_answer": NO_ANSWER_TEXT,
                "answer": NO_ANSWER_TEXT,
                "sources": [],
                "citations": {},
                "retrieved_source_texts": {},
                "forbidden_citation_format": False,
                "skip_guard": True,
            }
            
        # Retrieval confidence gate: if the average relevance distance across all
        # retrieved docs is above the threshold, the corpus has no relevant content
        # for this query. Abort before calling the LLM to prevent fabrication.
        # Uses average rather than top-1 to be robust against one accidentally
        # close chunk dragging a bad retrieval set past the gate.
        #
        # Threshold calibration for all-MiniLM-L6-v2 L2 distance:
        #   < 0.7  — strong match
        #   0.7-1.0 — reasonable match
        #   > 1.2  — likely off-topic
        #   > 1.3  — almost certainly unrelated
        RELEVANCE_GATE_THRESHOLD = 1.15  # average across all k docs

        # Get actual similarity scores for the retrieved docs
        try:
            import numpy as np
            embedding_fn = None
            for r in getattr(self.retriever, "retrievers", [self.retriever]):
                if hasattr(r, "vectorstore"):
                    embedding_fn = r.vectorstore.embedding_function
                    break

            if embedding_fn is not None:
                query_embedding = np.array(embedding_fn.embed_query(question))
                
                scored_docs_map = {}
                for doc in source_docs:
                    doc_embedding = np.array(embedding_fn.embed_query(doc.page_content))
                    scored_docs_map[doc.page_content] = float(np.linalg.norm(query_embedding - doc_embedding))
                
                avg_score = sum(scored_docs_map.values()) / max(len(scored_docs_map), 1)

                if avg_score > RELEVANCE_GATE_THRESHOLD:
                    return {
                        "question": question,
                        "raw_answer": NO_ANSWER_TEXT,
                        "answer": NO_ANSWER_TEXT,
                        "sources": [],
                        "citations": {},
                        "retrieved_source_texts": {},
                        "forbidden_citation_format": False,
                        "confidence_gate_triggered": True,
                        "avg_relevance_score": round(avg_score, 3),
                        "skip_guard": True,
                    }
            else:
                scored_docs_map = {}
        except Exception as e:
            print(f"Warning: Failed to compute relevance scores: {e}")
            scored_docs_map = {}
            
        # Temporal Re-Ranking Logic
        q_lower = question.lower()
        temporal_keywords_recent = {"last", "latest", "newest", "most recent", "current", "end"}
        temporal_keywords_old = {"first", "oldest", "initial", "start"}

        is_recent = any(kw in q_lower for kw in temporal_keywords_recent)
        is_old = any(kw in q_lower for kw in temporal_keywords_old)

        if is_recent or is_old:
            def _safe_date_sort_key(doc):
                d = (doc.metadata.get("date", "") or "")[:10]
                if not d:
                    # missing dates go to the bottom of the list
                    return "0000-00-00" if is_recent else "9999-99-99"
                return d
            
            # Additive Temporal Filter: Inject absolute top 20 from global corpus
            try:
                # bm25_retriever is the second in the ensemble
                bm25_retriever = self.retriever.retrievers[1]
                all_docs = getattr(bm25_retriever, "docs", [])
                if all_docs:
                    # Optional heuristic filter to ensure injected chunks match requested type
                    filtered_docs = all_docs
                    if "commit" in q_lower:
                        filtered_docs = [d for d in all_docs if d.metadata.get("source_type") == "commit"]
                    elif "issue" in q_lower:
                        filtered_docs = [d for d in all_docs if "issue" in d.metadata.get("source_type", "")]
                    elif "pr " in q_lower or "pull request" in q_lower:
                        filtered_docs = [d for d in all_docs if "pr" in d.metadata.get("source_type", "")]

                    sorted_all = sorted(filtered_docs, key=_safe_date_sort_key, reverse=is_recent)
                    temporal_candidates = sorted_all[:20]
                    # Merge and deduplicate by id
                    existing_ids = {doc.metadata.get("id", doc.page_content[:32]) for doc in source_docs}
                    for tdoc in temporal_candidates:
                        tid = tdoc.metadata.get("id", tdoc.page_content[:32])
                        if tid not in existing_ids:
                            source_docs.append(tdoc)
                            existing_ids.add(tid)
            except Exception as e:
                print(f"Warning: Failed to inject absolute temporal candidates: {e}")
            
            # Sort the combined candidate pool by date
            source_docs.sort(key=_safe_date_sort_key, reverse=is_recent)

        # Truncate back to RETRIEVER_K
        source_docs = source_docs[:RETRIEVER_K]
            
        # Compute a real relevance score for each retrieved doc, using the same
        # embedding model the retriever already has loaded. This reflects the
        # ACTUAL documents shown to the user, not a separately-fetched list that
        # can miss MMR/BM25-selected docs (which was the previous bug's cause).
        # (Old scoring logic removed in favor of relevance gate score map)

        # Generate answer using explicitly provided docs
        context_str = _format_docs(source_docs)
        generation_chain = PROMPT | self.llm | StrOutputParser()
        try:
            raw_answer = generation_chain.invoke({
                "context": context_str,
                "question": question
            })
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str or "quota" in error_str.lower():
                raise RuntimeError("QUOTA_EXCEEDED: API daily quota reached.") from e
            raise RuntimeError(f"Generation failed: {error_str[:200]}") from e

        # Strip the <thinking> block so it's not rendered in the UI or sent to the hallucination guard
        raw_answer = re.sub(r'<thinking>.*?</thinking>\n*', '', raw_answer, flags=re.DOTALL).strip()

        # Detect non-standard citation formats the guard cannot check.
        # These indicate the model used a forbidden citation style — flag them
        # so the UI can warn the user rather than silently showing "0 verified".
        _source_tag_pattern = re.compile(
            r'\[Source:[^\]]+\]'           # [Source: PR#35 | ...] style
            r'|\[Source:\s*\w+[^\]]*\]'    # [Source: anything]
        )
        _doc_link_pattern = re.compile(
            r'\[FastAPI documentation[^\]]*\]'  # [FastAPI documentation: X]
            r'|^\s*[*-]\s*\[FastAPI',           # bullet-style doc links
            re.MULTILINE
        )
        _sources_section_pattern = re.compile(
            r'^\s*Sources?\s*:',
            re.MULTILINE | re.IGNORECASE
        )

        has_forbidden_citation_format = bool(
            _source_tag_pattern.search(raw_answer) or
            _doc_link_pattern.search(raw_answer) or
            _sources_section_pattern.search(raw_answer)
        )

        # Build citation map from retrieved docs only (not the entire corpus)
        # This ensures citations can only point to actually-retrieved sources
        retrieved_id_url = {
            doc.metadata["id"]: doc.metadata["url"]
            for doc in source_docs
            if "id" in doc.metadata and "url" in doc.metadata
        }

        # Resolve [tag] → [tag](url)
        answer = resolve_citations(raw_answer, retrieved_id_url)

        # Extract all cited tags from raw answer
        cited_tags = CITATION_PATTERN.findall(raw_answer)
        citations = {tag: retrieved_id_url.get(tag, "") for tag in set(cited_tags)}

        # Format source metadata for UI
        sources = []
        for doc in source_docs:
            meta = doc.metadata.copy()
            # Grab the batched L2 distance score
            score = scored_docs_map.get(doc.page_content, None)
            
            sources.append({
                "id": meta.get("id", ""),
                "source_type": meta.get("source_type", ""),
                "url": meta.get("url", ""),
                "author": meta.get("author", ""),
                "date": (meta.get("date", "") or "")[:10],
                "text_preview": doc.page_content[:300],
                "labels": meta.get("labels", []),
                "similarity_score": score,
            })

        # Build {id: full_text} map of the 6 retrieved docs — used by the guard
        # to verify claims against exactly the text the LLM saw, not the full corpus.
        # Capped at MAX_TOTAL_CONTEXT chars so NLI model doesn't silently truncate.
        retrieved_source_texts: dict[str, str] = {}
        for doc in source_docs:
            doc_id = doc.metadata.get("id")
            if doc_id:
                existing = retrieved_source_texts.get(doc_id, "")
                combined = (existing + "\n\n" + doc.page_content).strip()
                retrieved_source_texts[doc_id] = combined[:MAX_TOTAL_CONTEXT]

        return {
            "question": question,
            "raw_answer": raw_answer,
            "answer": answer,
            "sources": sources,
            "citations": citations,
            "retrieved_source_texts": retrieved_source_texts,
            "forbidden_citation_format": has_forbidden_citation_format,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_chain.py \"Your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\n🔍 Question: {question}\n")

    rag = PatchContextRAG()
    result = rag.query(question)

    print("=" * 70)
    print("📝 ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "=" * 70)
    print(f"📚 SOURCES ({len(result['sources'])} retrieved)")
    print("=" * 70)
    for src in result["sources"]:
        print(f"  [{src['id']}] {src['source_type']} | {src['url']}")
        print(f"    {src['text_preview'][:120]}...")
        print()


if __name__ == "__main__":
    main()
