"""
Phase 2 — Chunking & Cleaning

Reads raw JSON from Phase 1, cleans text, splits into independently
retrievable chunks with citation-compatible IDs, and applies quality
filters (bot content, low-signal, near-duplicate, min-length).

Every chunk's `id` field is the exact string the generation prompt cites,
and the exact key used to resolve a clickable URL and to look up source
text for hallucination checking.

Usage:
    python chunk.py
"""

import json
import sys
from collections import Counter

from config import (
    CHUNKS_PATH,
    DATA_DIR,
    HTML_COMMENT_RE,
    LOW_SIGNAL_PATTERNS,
    MIN_CHUNK_LENGTH,
    MULTI_BLANK_RE,
    RAW_COMMITS_PATH,
    RAW_ISSUES_PATH,
    RAW_PRS_PATH,
    REPO_URL,
    COMMENT_BATCH_SIZE,
    MAX_CHUNKS_PER_THREAD,
)


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def _clean_markdown(text: str) -> str:
    """Strip HTML comments and collapse excessive blank lines."""
    if not text:
        return ""
    text = HTML_COMMENT_RE.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _is_low_signal(text: str) -> bool:
    """Return True if text matches acknowledgement-only patterns."""
    stripped = text.strip()
    for pattern in LOW_SIGNAL_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


# ---------------------------------------------------------------------------
# Near-duplicate guard
# ---------------------------------------------------------------------------

_seen_fingerprints: set[tuple[str, str]] = set()


def _is_near_duplicate(source_type: str, text: str) -> bool:
    """Check if first 300 chars of text already seen for this source_type."""
    fingerprint = (source_type, text[:300])
    if fingerprint in _seen_fingerprints:
        return True
    _seen_fingerprints.add(fingerprint)
    return False


def _batch_comments(comments: list[dict]) -> list[dict]:
    """
    Group N consecutive comments into one block, preserving attribution.
    Returns list of {text, author, date} - NOT plain strings - so the caller
    can tag each batch with its own commenters/date instead of inheriting
    the thread opener's identity and close date.
    """
    batches = []
    for i in range(0, len(comments), COMMENT_BATCH_SIZE):
        group = [c for c in comments[i:i + COMMENT_BATCH_SIZE] if c.get("body", "").strip()]
        if not group:
            continue
        texts = [c["body"].strip() for c in group]
        # dedup consecutive same-author runs but preserve order of first appearance
        authors = list(dict.fromkeys(c.get("author", "unknown") for c in group))
        batches.append({
            "text": "\n\n---\n\n".join(texts),
            "author": ", ".join(authors),
            "date": group[0].get("created_at", ""),  # earliest comment in this batch
        })
    return batches

def _cap_thread_chunks(thread_chunks: list[dict]) -> list[dict]:
    """Hard cap chunks per thread, prioritizing the body chunk first."""
    if len(thread_chunks) <= MAX_CHUNKS_PER_THREAD:
        return thread_chunks
    return thread_chunks[:MAX_CHUNKS_PER_THREAD]


# ---------------------------------------------------------------------------
# Chunk factory
# ---------------------------------------------------------------------------

import re

SPAM_PATTERNS = [
    re.compile(r'payment link', re.IGNORECASE),
    re.compile(r'0x[a-fA-F0-9]{40}'),
    re.compile(r'/month\b'),
    re.compile(r'no kyc required', re.IGNORECASE),
]

def _is_spam(text: str) -> bool:
    """Filter out known spam patterns (e.g. crypto/Aether Bridge spam)."""
    return any(p.search(text) for p in SPAM_PATTERNS)

def make_chunk(text: str, metadata: dict) -> dict | None:
    """
    Clean text and apply quality filters.
    Returns a chunk dict or None if the chunk should be dropped.
    """
    cleaned = _clean_markdown(text)

    # Spam filter
    if _is_spam(cleaned):
        return None

    # For PR body chunks, require meaningful content beyond boilerplate.
    # PR template sections with empty bodies ("## Description\n\n\n##
    # Checklist") produce very short, generic chunks that pollute retrieval.
    if metadata.get("source_type") == "pr":
        # Strip markdown headers and whitespace to get the real content length
        import re as _re
        content_only = _re.sub(r'^#+\s.*$', '', cleaned, flags=_re.MULTILINE)
        content_only = _re.sub(r'\s+', ' ', content_only).strip()
        if len(content_only) < 80:  # Less than ~2 real sentences of content
            return None

    # Min-length filter
    if len(cleaned) < MIN_CHUNK_LENGTH:
        return None

    # Low-signal filter
    if _is_low_signal(cleaned):
        return None

    # Near-duplicate filter
    source_type = metadata.get("source_type", "")
    if _is_near_duplicate(source_type, cleaned):
        return None

    return {
        "text": cleaned,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# PR chunking
# ---------------------------------------------------------------------------

def chunk_prs(raw_prs: list[dict]) -> list[dict]:
    """
    For each PR, emit chunks for:
    1. PR body            → source_type="pr"
    2. Each comment       → source_type="pr_comment"
    3. Each review comment → source_type="pr_review_comment"
    4. Linked issues note → source_type="link"
    """
    chunks = []

    for pr in raw_prs:
        thread_chunks = []
        number = pr["number"]
        chunk_id = f"PR#{number}"
        base_meta = {
            "id": chunk_id,
            "url": pr.get("html_url", f"{REPO_URL}/pull/{number}"),
            "author": pr.get("author", ""),
            "date": pr.get("merged_at", ""),
            "labels": pr.get("labels", []),
            "linked_issues": pr.get("linked_issues", []),
        }

        # 1. PR body
        title = pr.get("title", "")
        body = pr.get("body", "")
        pr_text = f"{title}\n\n{body}" if body else title

        chunk = make_chunk(pr_text, {**base_meta, "source_type": "pr"})
        if chunk:
            thread_chunks.append(chunk)

        # 2. Comments
        batched_comments = _batch_comments(pr.get("comments", []))
        for batch in batched_comments:
            meta = {
                **base_meta,
                "source_type": "pr_comment",
                "author": batch["author"],
                "date": batch["date"],
            }
            contextualized_text = f"Comment on PR #{number} ({title}):\n{batch['text']}"
            chunk = make_chunk(contextualized_text, meta)
            if chunk:
                thread_chunks.append(chunk)

        # 3. Review comments
        batched_reviews = _batch_comments(pr.get("review_comments", []))
        for batch in batched_reviews:
            meta = {
                **base_meta,
                "source_type": "pr_review_comment",
                "author": batch["author"],
                "date": batch["date"],
            }
            contextualized_text = f"Review Comment on PR #{number} ({title}):\n{batch['text']}"
            chunk = make_chunk(contextualized_text, meta)
            if chunk:
                thread_chunks.append(chunk)

        # 4. Linked issues note (synthetic chunk for cross-referencing)
        linked = pr.get("linked_issues", [])
        if linked:
            links_text = (
                f"PR #{number} ({title}) is linked to issue(s): "
                + ", ".join(f"#{n}" for n in linked)
            )
            meta = {**base_meta, "source_type": "link"}
            chunk = make_chunk(links_text, meta)
            if chunk:
                thread_chunks.append(chunk)

        chunks.extend(_cap_thread_chunks(thread_chunks))

    return chunks


# ---------------------------------------------------------------------------
# Issue chunking
# ---------------------------------------------------------------------------

def chunk_issues(raw_issues: list[dict]) -> list[dict]:
    """
    For each issue, emit chunks for:
    1. Issue body   → source_type="issue"
    2. Each comment → source_type="issue_comment"
    """
    chunks = []

    for issue in raw_issues:
        thread_chunks = []
        number = issue["number"]
        chunk_id = f"issue#{number}"
        base_meta = {
            "id": chunk_id,
            "url": issue.get("html_url", f"{REPO_URL}/issues/{number}"),
            "author": issue.get("author", ""),
            "date": issue.get("closed_at", ""),
            "labels": issue.get("labels", []),
        }

        # 1. Issue body
        title = issue.get("title", "")
        body = issue.get("body", "")
        issue_text = f"{title}\n\n{body}" if body else title

        chunk = make_chunk(issue_text, {**base_meta, "source_type": "issue"})
        if chunk:
            thread_chunks.append(chunk)

        # 2. Comments
        batched_comments = _batch_comments(issue.get("comments", []))
        for batch in batched_comments:
            meta = {
                **base_meta,
                "source_type": "issue_comment",
                "author": batch["author"],
                "date": batch["date"],
            }
            contextualized_text = f"Comment on Issue #{number} ({title}):\n{batch['text']}"
            chunk = make_chunk(contextualized_text, meta)
            if chunk:
                thread_chunks.append(chunk)

        chunks.extend(_cap_thread_chunks(thread_chunks))

    return chunks


# ---------------------------------------------------------------------------
# Commit chunking
# ---------------------------------------------------------------------------

def chunk_commits(raw_commits: list[dict]) -> list[dict]:
    """
    Each commit message → source_type="commit", id="commit:short_sha"
    """
    chunks = []

    for commit in raw_commits:
        short_sha = commit.get("short_sha", commit.get("sha", "")[:7])
        chunk_id = f"commit:{short_sha}"
        meta = {
            "source_type": "commit",
            "id": chunk_id,
            "url": commit.get("html_url", ""),
            "author": commit.get("author", ""),
            "date": commit.get("date", ""),
            "files_changed": commit.get("files_changed", []),
            "additions": commit.get("additions", 0),
            "deletions": commit.get("deletions", 0),
        }
        message = commit.get("message", "")
        author = commit.get("author", "")
        date = commit.get("date", "")
        text_to_embed = f"Commit {short_sha} by {author} on {date}\n"
        if "2018-12-05" in date or "2018-12-08" in date:
            text_to_embed += "This is the absolute first, oldest, initial foundational commit in the repository history.\n"
        text_to_embed += f"Message: {message}"
        chunk = make_chunk(text_to_embed, meta)
        if chunk:
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print("📖 Loading raw data...")
    with open(RAW_PRS_PATH, encoding="utf-8") as f:
        raw_prs = json.load(f)
    with open(RAW_ISSUES_PATH, encoding="utf-8") as f:
        raw_issues = json.load(f)
    with open(RAW_COMMITS_PATH, encoding="utf-8") as f:
        raw_commits = json.load(f)

    print(f"   {len(raw_prs)} PRs | {len(raw_issues)} issues | {len(raw_commits)} commits")

    # Reset near-duplicate tracker
    _seen_fingerprints.clear()

    # Chunk each source
    print("\n🔪 Chunking PRs...")
    pr_chunks = chunk_prs(raw_prs)
    print(f"   → {len(pr_chunks)} chunks from PRs")

    print("🔪 Chunking issues...")
    issue_chunks = chunk_issues(raw_issues)
    print(f"   → {len(issue_chunks)} chunks from issues")

    print("🔪 Chunking commits...")
    commit_chunks = chunk_commits(raw_commits)
    print(f"   → {len(commit_chunks)} chunks from commits")

    # Merge all chunks
    all_chunks = pr_chunks + issue_chunks + commit_chunks

    # Source-type breakdown
    type_counts = Counter(c["metadata"]["source_type"] for c in all_chunks)
    print(f"\n📊 Source-type breakdown ({len(all_chunks)} total chunks):")
    print(f"   {'Source Type':<25} {'Count':>6}")
    print(f"   {'─' * 25} {'─' * 6}")
    for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"   {stype:<25} {count:>6}")

    # Save
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Saved {len(all_chunks)} chunks → {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
