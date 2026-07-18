"""
Benchmark Helper — speeds up building benchmark_questions.json ground truth.

For each question, surfaces the top-k retrieved chunks with their id, url,
and text preview so you can quickly identify the correct sources without
manually crawling GitHub.

Usage:
    python benchmark_helper.py "Why does FastAPI use Depends() for dependency injection?"
    python benchmark_helper.py --all   # process all questions in template
"""

import json
import sys

from config import BENCHMARK_PATH, BENCHMARK_TEMPLATE_PATH
from rag_chain import load_retriever


def find_candidate_sources(question: str, top_k: int = 10) -> list[dict]:
    """
    Retrieve top-k candidate chunks for a question.
    Returns list of {id, url, source_type, author, date, text_preview}.
    """
    retriever = load_retriever()
    # Temporarily raise k to get more candidates for manual review
    retriever.search_kwargs["k"] = top_k
    retriever.search_kwargs["fetch_k"] = top_k * 4

    docs = retriever.invoke(question)
    candidates = []
    for doc in docs:
        meta = doc.metadata
        candidates.append({
            "id": meta.get("id", ""),
            "url": meta.get("url", ""),
            "source_type": meta.get("source_type", ""),
            "author": meta.get("author", ""),
            "date": (meta.get("date", "") or "")[:10],
            "text_preview": doc.page_content[:400],
        })
    return candidates


def print_candidates(question: str, candidates: list[dict]) -> None:
    """Pretty-print candidates for manual review."""
    print(f"\n{'=' * 70}")
    print(f"❓ Question: {question}")
    print(f"{'=' * 70}")
    print(f"📚 Top {len(candidates)} candidate sources:\n")
    for i, c in enumerate(candidates, 1):
        print(f"  [{i}] {c['id']}  ({c['source_type']})")
        print(f"       URL: {c['url']}")
        print(f"       Author: {c['author']} | Date: {c['date']}")
        print(f"       Preview: {c['text_preview'][:200]}")
        print()


def process_all_template() -> None:
    """
    Process all questions in the benchmark template, surface candidates,
    and write a pre-filled benchmark_questions.json for manual completion.
    """
    with open(BENCHMARK_TEMPLATE_PATH, encoding="utf-8") as f:
        template = json.load(f)

    print(f"📋 Processing {len(template)} template questions...\n")
    filled = []

    for entry in template:
        question = entry["question"]
        candidates = find_candidate_sources(question, top_k=5)
        print_candidates(question, candidates)

        filled.append({
            "question": question,
            "ground_truth": entry.get("ground_truth", ""),  # fill in manually
            "expected_source_ids": [c["id"] for c in candidates[:3]],  # top-3 as hint
            "candidate_sources": candidates,
        })

    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(filled, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Pre-filled benchmark saved → {BENCHMARK_PATH}")
    print("   Review each entry and fill in 'ground_truth' before running evaluate.py")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python benchmark_helper.py \"Your question\"")
        print("  python benchmark_helper.py --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        process_all_template()
    else:
        question = " ".join(sys.argv[1:])
        candidates = find_candidate_sources(question, top_k=10)
        print_candidates(question, candidates)
        print("\n💡 Use these source ids as 'expected_source_ids' in your benchmark JSON.")


if __name__ == "__main__":
    main()
