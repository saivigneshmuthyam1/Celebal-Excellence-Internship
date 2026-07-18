"""
Phase 5 — NLI Hallucination Guard

Verifies each cited claim in a generated answer against the exact source
chunk behind that citation — NOT against the whole context blob.

This per-claim, per-citation design catches a single fabricated PR number
sitting next to four correct ones, which a generic "does the answer roughly
match the context" check would miss.

Model: cross-encoder/nli-deberta-v3-small
  Output score order: [contradiction, entailment, neutral]

Usage (CLI):
    python hallucination_guard.py "Why does FastAPI use Depends() for dependency injection?"
"""

import json
import re
import sys

import numpy as np
from sentence_transformers import CrossEncoder

from config import (
    CHUNKS_PATH,
    MAX_TOTAL_CONTEXT,
    NLI_MODEL,
    NLI_THRESHOLD,
)
from rag_chain import CITATION_PATTERN, PatchContextRAG

NLI_MAX_CHARS_PER_CHUNK = 800  # cross-encoder/nli-deberta-v3-small has
                               # a ~512 token limit; MAX_TOTAL_CONTEXT
                               # from config.py is sized for RAGAs's
                               # LLM judge, NOT this NLI model - keep
                               # them separate.
MAX_CHARS_PER_CHUNK = 800

# Verification status labels
VERIFIED = "VERIFIED"
FLAGGED_CONTRADICTION = "FLAGGED_CONTRADICTION"
FLAGGED_UNSUPPORTED = "FLAGGED_UNSUPPORTED"
UNVERIFIED = "UNVERIFIED"

# NLI model output indices
IDX_CONTRADICTION = 0
IDX_ENTAILMENT = 1
IDX_NEUTRAL = 2


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using simple regex.
    Handles common abbreviations to avoid false splits.
    """
    # Remove resolved markdown links [tag](url) → tag for clean splitting
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'[\1]', text)
    # Split on sentence-ending punctuation followed by whitespace + capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\["])', clean)
    return [s.strip() for s in sentences if s.strip()]


def _strip_citations(sentence: str) -> str:
    """Remove citation tags from sentence for clean NLI hypothesis."""
    return CITATION_PATTERN.sub("", sentence).strip()


class HallucinationGuard:
    """
    Per-claim, per-citation NLI verification of generated answers.

    For each sentence:
    - No citation → UNVERIFIED
    - Has citation(s) → look up exact chunk text, run NLI, determine status
    """

    def __init__(self, model_name: str = NLI_MODEL, threshold: float = NLI_THRESHOLD):
        print(f"🔬 Loading NLI model: {model_name} (first run downloads ~500MB)...")
        self.model = CrossEncoder(model_name)
        self.threshold = threshold
        self.chunks_by_id = self._load_chunks_index()
        print(f"   Loaded {len(self.chunks_by_id)} chunk texts for verification")

    def _load_chunks_index(self) -> dict[str, str]:
        """Build {id: concatenated_text} map from chunks.json for O(1) lookup."""
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunks = json.load(f)

        grouped: dict[str, list[str]] = {}
        for c in chunks:
            cid = c["metadata"].get("id")
            if cid:
                grouped.setdefault(cid, []).append(c["text"][:MAX_CHARS_PER_CHUNK])

        result = {cid: "\n\n".join(texts)[:NLI_MAX_CHARS_PER_CHUNK] for cid, texts in grouped.items()}
        print(f"   Loaded {len(result)} unique issue/PR threads in chunk index")
        return result

    def check_answer(self, answer: str, source_texts: dict[str, str] | None = None) -> list[dict]:
        """
        Verify each sentence in the answer against its cited source(s).

        Args:
            answer: The resolved answer string (with [tag](url) links).
            source_texts: Optional {id: text} map of the actual retrieved docs the LLM
                          saw. When provided, used preferentially over self.chunks_by_id
                          so the guard evaluates against the exact context that generated
                          the claim.
        """
        lookup = source_texts if source_texts else self.chunks_by_id
        sentences = _split_sentences(answer)

        # 1. First pass: extract citations, build pairs for a single mega-batch
        all_pairs = []
        sentence_metadata = []

        for i, sentence in enumerate(sentences):
            cited_tags = CITATION_PATTERN.findall(sentence)
            
            meta = {
                "sentence": sentence,
                "cited_tags": cited_tags,
                "tag_pairs": {}, # map tag -> pair_index
                "hypothesis": ""
            }

            if cited_tags:
                hypothesis = _strip_citations(sentence)
                if len(hypothesis) >= 10:
                    meta["hypothesis"] = hypothesis
                    for tag in cited_tags:
                        chunk_text = lookup.get(tag) or self.chunks_by_id.get(tag)
                        if chunk_text:
                            chunk_text = chunk_text[:NLI_MAX_CHARS_PER_CHUNK]
                            meta["tag_pairs"][tag] = len(all_pairs)
                            all_pairs.append((chunk_text, hypothesis))
            
            sentence_metadata.append(meta)

        # 2. Single batch prediction for the entire answer (massive speedup)
        all_scores = self.model.predict(all_pairs, apply_softmax=True) if all_pairs else []

        # 3. Second pass: resolve results
        results = []
        for meta in sentence_metadata:
            sentence = meta["sentence"]
            cited_tags = meta["cited_tags"]

            if not cited_tags:
                results.append({
                    "sentence": sentence,
                    "status": UNVERIFIED,
                    "reason": "No citation attached to this claim.",
                    "citations": [],
                    "scores": {},
                })
                continue

            if not meta["hypothesis"]:
                results.append({
                    "sentence": sentence,
                    "status": UNVERIFIED,
                    "reason": "Sentence too short to verify after stripping citations.",
                    "citations": cited_tags,
                    "scores": {},
                })
                continue

            if not meta["tag_pairs"]:
                results.append({
                    "sentence": sentence,
                    "status": FLAGGED_UNSUPPORTED,
                    "reason": f"Cited source(s) {cited_tags} not found in chunk index.",
                    "citations": cited_tags,
                    "scores": {},
                })
                continue

            # Evaluate each tag independently
            sent_scores = {}
            best_entailment = -1.0
            any_contradiction = False
            best_neutral = -1.0
            
            for tag, idx in meta["tag_pairs"].items():
                scores = all_scores[idx]
                ent = float(scores[IDX_ENTAILMENT])
                con = float(scores[IDX_CONTRADICTION])
                neu = float(scores[IDX_NEUTRAL])
                
                sent_scores[tag] = {
                    "entailment": round(ent, 4),
                    "contradiction": round(con, 4),
                    "neutral": round(neu, 4),
                }
                
                if ent > best_entailment:
                    best_entailment = ent
                if neu > best_neutral:
                    best_neutral = neu
                
                # Use np.argmax to safely handle both numpy arrays and plain Python lists
                max_idx = int(np.argmax(scores))
                if max_idx == IDX_CONTRADICTION:
                    any_contradiction = True
                    
            if any_contradiction:
                status = FLAGGED_CONTRADICTION
                reason = "One or more cited sources contradict this claim."
            elif best_entailment >= self.threshold:
                status = VERIFIED
                reason = f"Supported by at least one cited source (max entailment={best_entailment:.3f})."
            else:
                status = FLAGGED_UNSUPPORTED
                reason = f"The cited sources are neutral/unsupported regarding this claim (max entailment={best_entailment:.3f})."

            results.append({
                "sentence": sentence,
                "status": status,
                "reason": reason,
                "citations": cited_tags,
                "scores": sent_scores,
            })

        return results

    def check_from_rag_result(self, rag_result: dict) -> list[dict]:
        """Convenience wrapper for PatchContextRAG.query() output."""
        return self.check_answer(
            answer=rag_result["answer"],
            source_texts=rag_result.get("retrieved_source_texts"),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

STATUS_EMOJI = {
    VERIFIED: "✅",
    FLAGGED_CONTRADICTION: "❌",
    FLAGGED_UNSUPPORTED: "⚠️ ",
    UNVERIFIED: "❓",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python hallucination_guard.py \"Your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\n🔍 Question: {question}\n")

    # Run RAG chain
    print("🔗 Running RAG chain...")
    rag = PatchContextRAG()
    result = rag.query(question)

    print("\n📝 ANSWER:")
    print(result["answer"])

    # Run hallucination guard
    print("\n🔬 Running NLI hallucination guard...")
    guard = HallucinationGuard()
    verifications = guard.check_from_rag_result(result)

    print("\n" + "=" * 70)
    print("🔍 PER-CLAIM VERIFICATION")
    print("=" * 70)
    for v in verifications:
        emoji = STATUS_EMOJI.get(v["status"], "?")
        print(f"\n{emoji} [{v['status']}]")
        print(f"   Claim: {v['sentence'][:150]}")
        if v["citations"]:
            print(f"   Citations: {v['citations']}")
        print(f"   Reason: {v['reason']}")
        if v["scores"]:
            for tag, s in v["scores"].items():
                print(f"   {tag}: entailment={s['entailment']:.3f} | "
                      f"contradiction={s['contradiction']:.3f} | neutral={s['neutral']:.3f}")

    # Summary
    counts = {VERIFIED: 0, FLAGGED_CONTRADICTION: 0, FLAGGED_UNSUPPORTED: 0, UNVERIFIED: 0}
    for v in verifications:
        counts[v["status"]] = counts.get(v["status"], 0) + 1

    print(f"\n📊 Summary: {len(verifications)} claims checked")
    for status, count in counts.items():
        if count:
            print(f"   {STATUS_EMOJI[status]} {status}: {count}")


if __name__ == "__main__":
    main()
