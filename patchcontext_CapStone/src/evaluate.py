"""
Phase 6 — RAGAs Evaluation

Runs the benchmark questions through the RAG pipeline and scores with:
  - Faithfulness          (generation quality)
  - ResponseRelevancy     (answer alignment)
  - LLMContextPrecisionWithReference  (retrieval precision)
  - LLMContextRecallWithReference     (retrieval recall)

Reports results with and without MMR as a comparison table.

Usage:
    python evaluate.py
    python evaluate.py --no-mmr   # Compare: disable MMR (plain similarity)
"""

import json
import os
import sys

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from config import (
    BENCHMARK_PATH,
    BENCHMARK_TEMPLATE_PATH,
    CHUNKS_PATH,
    HF_EMBEDDING_MODEL,
    EVAL_RESULTS_PATH,
    FAISS_INDEX_DIR,
    GROQ_API_KEY,
    GROQ_API_KEY_EVAL,
    LLM_MODEL,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    RETRIEVER_LAMBDA_MULT,
)
from rag_chain import PatchContextRAG

MAX_CALLS_PER_RUN = 80


def load_benchmark() -> list[dict]:
    """Load benchmark questions. Falls back to template if filled version missing."""
    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Filter to only questions with ground truth filled in
        filled = [q for q in data if q.get("ground_truth", "").strip()]
        if filled:
            print(f"📋 Loaded {len(filled)} benchmark questions with ground truth")
            return filled
        print("⚠️  benchmark_questions.json exists but no ground_truth filled in.")
        print("   Run benchmark_helper.py --all and fill in ground_truth fields first.")
        sys.exit(1)
    else:
        print(f"⚠️  {BENCHMARK_PATH} not found.")
        print(f"   Run: python benchmark_helper.py --all")
        print(f"   Then fill in the 'ground_truth' field for each question.")
        sys.exit(1)


def build_rag_no_mmr():
    """Build a PatchContextRAG instance with plain similarity (no MMR) for comparison."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    # Plain similarity search instead of MMR
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    class NoMMRRAG(PatchContextRAG):
        def __init__(self):
            # Skip parent __init__, build chain manually with plain retriever
            from langchain.schema import StrOutputParser
            from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
            from langchain_groq import ChatGroq
            from rag_chain import PROMPT, _format_docs, _build_id_url_map

            self.retriever = retriever
            self.id_url_map = _build_id_url_map()
            self.llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)
            self.chain = (
                RunnableParallel(
                    context=self.retriever | _format_docs,
                    question=RunnablePassthrough(),
                )
                | PROMPT
                | self.llm
                | StrOutputParser()
            )

    return NoMMRRAG()


def collect_samples(rag: PatchContextRAG, questions: list[dict]) -> dict:
    """Run all benchmark questions through the RAG pipeline and collect samples."""
    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    for i, q in enumerate(questions, 1):
        question = q["question"]
        ground_truth = q["ground_truth"]
        print(f"  [{i}/{len(questions)}] {question[:70]}...")

        try:
            result = rag.query(question)
            dataset_dict["question"].append(question)
            dataset_dict["answer"].append(result["answer"])
            dataset_dict["contexts"].append(list(result.get("retrieved_source_texts", {}).values()))
            dataset_dict["ground_truth"].append(ground_truth)
            
            import time
            time.sleep(5)  # Stay under Gemini 15 RPM free tier limit
        except Exception as e:
            print(f"    ⚠️  Error on question {i}: {e}")
    if not dataset_dict["question"]:
        print("❌ No samples collected — all questions failed during generation. Aborting eval.")
        sys.exit(1)

    return dataset_dict


import asyncio as _asyncio
import time as _time_module
from typing import ClassVar

class ThrottledLLM(ChatGroq):
    """
    Wrapper around Gemini to ensure we don't exceed rate limits (15 RPM).
    Also manually pops 'temperature' and 'n' kwargs to avoid conflicts.
    in langchain-google-genai >= 2.0 where temperature is deprecated but Ragas forces it.
    """

    _last_call: ClassVar[float] = 0.0
    _min_gap: ClassVar[float] = 20.0  # seconds between calls
    _call_count: ClassVar[int] = 0

    def _check_budget(self):
        ThrottledLLM._call_count += 1
        if ThrottledLLM._call_count > MAX_CALLS_PER_RUN:
            raise RuntimeError("EVAL_BUDGET_EXCEEDED: stopping to preserve remaining daily quota for the live app")

    def _throttle(self):
        elapsed = _time_module.time() - ThrottledLLM._last_call
        if elapsed < ThrottledLLM._min_gap:
            _time_module.sleep(ThrottledLLM._min_gap - elapsed)
        ThrottledLLM._last_call = _time_module.time()

    async def _athrottle(self):
        # No asyncio.Lock here — a module-level Lock is bound to the first event
        # loop and raises RuntimeError when RAGAs creates a new loop for the
        # second evaluate() pass.  max_workers=1 ensures sequential execution
        # so a lock provides no additional safety.
        elapsed = _time_module.time() - ThrottledLLM._last_call
        if elapsed < ThrottledLLM._min_gap:
            await _asyncio.sleep(ThrottledLLM._min_gap - elapsed)
        ThrottledLLM._last_call = _time_module.time()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._check_budget()
        requested_n = kwargs.pop("n", getattr(self, "n", 1)) or 1
        self.n = 1
        kwargs.pop("temperature", None)
        self._throttle()
        res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if requested_n > 1 and len(res.generations) == 1:
            res.generations = res.generations * requested_n
        return res

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._check_budget()
        requested_n = kwargs.pop("n", getattr(self, "n", 1)) or 1
        self.n = 1
        kwargs.pop("temperature", None)
        await self._athrottle()
        res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if requested_n > 1 and len(res.generations) == 1:
            res.generations = res.generations * requested_n
        return res


def run_ragas(samples_dict: dict, label: str) -> dict:
    """Run RAGAs evaluation and return scores dict."""
    evaluator_llm = ThrottledLLM(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_API_KEY_EVAL)
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    dataset = Dataset.from_dict(samples_dict)
    print(f"\n📊 Running RAGAs evaluation ({label})...")
    raise_exc = "--debug" in sys.argv
    # Ragas runs async by default. Since we use ThrottledLLM to enforce a 5s gap
    # between calls, a large batch of concurrent queries will queue up and hit Ragas'
    # default 180s timeout before they can execute. We increase the timeout and limit
    # max_workers to prevent tasks from timing out while waiting in the throttle queue.
    run_config = RunConfig(timeout=900, max_workers=1)

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=raise_exc,  # log failures, don't crash unless --debug
        run_config=run_config,
    )
    df = results.to_pandas()
    
    per_question_file = EVAL_RESULTS_PATH.parent / f"eval_results_per_question_{label.lower().replace(' ', '_')}.json"
    df.to_json(per_question_file, orient="records", indent=2)
    print(f"✅ Saved per-question results to {per_question_file.name}")

    scores = {
        "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df.columns else 0.0,
        "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df.columns else 0.0,
        "context_precision": float(df["context_precision"].mean()) if "context_precision" in df.columns else 0.0,
        "context_recall": float(df["context_recall"].mean()) if "context_recall" in df.columns else 0.0,
    }
    return scores


def print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table of all evaluation runs."""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    col_w = 22

    print("\n" + "=" * 80)
    print("📊 RAGAS EVALUATION RESULTS")
    print("=" * 80)

    # Header
    header = f"{'Metric':<25}"
    for label in results:
        header += f"{label:>{col_w}}"
    print(header)
    print("-" * (25 + col_w * len(results)))

    # Metric rows
    for metric in metrics:
        row = f"{metric:<25}"
        for label, scores in results.items():
            val = scores.get(metric, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)

    print("=" * 80)
    print("\n💡 Interpretation:")
    print("  Faithfulness + Answer Relevancy → generation quality")
    print("  Context Precision + Recall      → retrieval quality")
    print("  Higher is better for all metrics (0–1 scale)")


def main():
    compare_mode = "--compare" in sys.argv or "--no-mmr" in sys.argv
    
    questions = load_benchmark()
    if "--fast" in sys.argv:
        print("⚠️ Running in fast subset mode. Removing --fast will run all questions.")
        questions = questions[:3]

    all_results = {}

    # Run with MMR (primary)
    print("\n🔗 Running pipeline WITH MMR retrieval...")
    rag_mmr = PatchContextRAG()
    samples_mmr = collect_samples(rag_mmr, questions)
    scores_mmr = run_ragas(samples_mmr, "WITH MMR")
    all_results["With MMR"] = scores_mmr

    # Optionally compare without MMR
    if compare_mode:
        print("\n🔗 Running pipeline WITHOUT MMR (plain similarity)...")
        rag_plain = build_rag_no_mmr()
        samples_plain = collect_samples(rag_plain, questions)
        scores_plain = run_ragas(samples_plain, "WITHOUT MMR")
        all_results["Without MMR"] = scores_plain

    # Print comparison table
    print_comparison_table(all_results)

    # Save results
    output = {
        "benchmark_size": len(questions),
        "retriever_config": {
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_LAMBDA_MULT,
        },
        "results": all_results,
    }
    EVAL_RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n✅ Results saved → {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
