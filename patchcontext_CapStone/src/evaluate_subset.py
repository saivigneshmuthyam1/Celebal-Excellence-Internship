import json
import os
import sys

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
    HF_EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    GROQ_API_KEY_EVAL,
    LLM_MODEL,
    RETRIEVER_K,
)
from evaluate import ThrottledLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag_chain import PatchContextRAG

os.environ["GROQ_API_KEY"] = GROQ_API_KEY_EVAL

def main():
    print("Loading benchmark...")
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        data = json.load(f)
    
    # Only evaluate the 5 corpus-answerable questions
    subset = [q for q in data if q.get("corpus_answerable")]
    print(f"Found {len(subset)} corpus-answerable questions.")

    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    # Inject Temporal Accuracy test case
    subset.append({
        "question": "What was the very last commit made to the FastAPI repository?",
        "ground_truth": "The very last commit should match CORPUS_NEWEST_DATE in the context.",
        "corpus_answerable": True
    })
    
    rag = PatchContextRAG()
    
    print("\nCollecting RAG pipeline results for the subset...")
    for i, q in enumerate(subset, 1):
        question = q["question"]
        ground_truth = q["ground_truth"]
        print(f"  [{i}/{len(subset)}] {question[:70]}...")
        result = rag.query(question)
        dataset_dict["question"].append(question)
        dataset_dict["answer"].append(result["answer"])
        dataset_dict["contexts"].append(list(result.get("retrieved_source_texts", {}).values()))
        dataset_dict["ground_truth"].append(ground_truth)
        import time
        time.sleep(20)
    print("\n=== FINAL SUBSET GENERATIONS ===")
    
    with open("data/eval_results_subset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2)
    
    print("Results written to data/eval_results_subset.json. You can view them to verify the prompt constraints!")

if __name__ == "__main__":
    main()
