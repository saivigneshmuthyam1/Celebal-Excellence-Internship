"""
Phase 1 — GitHub Data Extraction

Fetches merged PRs, closed issues, and standalone commits from the FastAPI
repository via the GitHub REST API. Outputs raw JSON files ready for chunking.

Handles rate limiting (sleep on 403/429), pagination (Link headers),
and bot filtering automatically.

Usage:
    python extract.py
"""

import json
import re
import sys
import time
from datetime import datetime

import requests

from config import (
    BOT_AUTHORS,
    DATA_DIR,
    EXTRACT_FORCE_REFRESH,
    GITHUB_API_BASE,
    GITHUB_PER_PAGE,
    GITHUB_RATE_LIMIT_BUFFER,
    GITHUB_TOKEN,
    MAX_COMMITS,
    MAX_ISSUES,
    MAX_PRS,
    RAW_COMMITS_PATH,
    RAW_ISSUES_PATH,
    RAW_PRS_PATH,
    REPO_FULL,
    REPO_URL,
)

# ---------------------------------------------------------------------------
# HTTP session with auth
# ---------------------------------------------------------------------------

session = requests.Session()
if GITHUB_TOKEN:
    session.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
session.headers.update({"Accept": "application/vnd.github.v3+json"})


def _check_rate_limit(response: requests.Response) -> None:
    """Sleep if we're running low on GitHub API quota."""
    remaining = int(response.headers.get("X-RateLimit-Remaining", 999))
    if remaining < GITHUB_RATE_LIMIT_BUFFER:
        reset_ts = int(response.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset_ts - int(time.time()), 1) + 5  # 5s buffer
        print(f"  ⏳ Rate limit low ({remaining} remaining). Sleeping {wait}s...")
        time.sleep(wait)


def _handle_error(response: requests.Response) -> bool:
    """Handle 403/429. Returns True if caller should retry."""
    if response.status_code in (403, 429):
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            wait = int(retry_after) + 5
        else:
            reset_ts = int(response.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_ts - int(time.time()), 1) + 5
        print(f"  ⚠️  {response.status_code} — sleeping {wait}s before retry...")
        time.sleep(wait)
        return True  # retry
    return False


def _get(url: str, params: dict = None) -> requests.Response:
    """GET with retry on rate-limit and connection errors."""
    for attempt in range(5):
        try:
            resp = session.get(url, params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = min(2 ** attempt * 5, 60)  # 5, 10, 20, 40, 60s
            print(f"  ⚠️  Connection error (attempt {attempt+1}/5): {e}")
            print(f"      Retrying in {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            _check_rate_limit(resp)
            return resp
        if _handle_error(resp):
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed after 5 attempts: {url}")


def _paginate(url: str, params: dict = None, max_items: int = None) -> list:
    """Fetch all pages, respecting max_items limit."""
    params = dict(params or {})
    params.setdefault("per_page", GITHUB_PER_PAGE)
    items = []

    while url and (max_items is None or len(items) < max_items):
        resp = _get(url, params=params)
        batch = resp.json()
        if not isinstance(batch, list):
            break
        items.extend(batch)
        # Follow Link: <url>; rel="next"
        link_header = resp.headers.get("Link", "")
        next_match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        url = next_match.group(1) if next_match else None
        params = None  # params are baked into the Link URL

    if max_items is not None:
        items = items[:max_items]
    return items


# ---------------------------------------------------------------------------
# PR extraction
# ---------------------------------------------------------------------------

def _extract_linked_issues(body: str) -> list[int]:
    """Extract issue numbers referenced as #NNN in PR body."""
    if not body:
        return []
    # Match #123, fixes #123, closes #123, resolves #123, etc.
    matches = re.findall(
        r'(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)',
        body,
        re.IGNORECASE,
    )
    # Also match standalone #NNN references
    standalone = re.findall(r'(?<!\w)#(\d+)', body)
    all_nums = set(int(n) for n in matches + standalone)
    return sorted(all_nums)


def _is_bot(author_login: str) -> bool:
    """Return True if the author is a known bot.

    Handles both login-style matches (e.g. 'dependabot[bot]' in BOT_AUTHORS)
    and name-style strings (e.g. commit.commit.author.name = 'github-actions[bot]')
    which bypass login matching because they're not GitHub account logins.
    """
    if not author_login:
        return False
    # Exact match against the configured bot login set
    if author_login in BOT_AUTHORS:
        return True
    # Catch any author whose name/login contains the '[bot]' suffix pattern
    # (covers github-actions[bot], dependabot[bot], etc. coming in as names)
    if author_login.lower().endswith("[bot]"):
        return True
    return False


def build_pr_records() -> list[dict]:
    """Fetch merged PRs with comments, review comments, and linked issues."""
    limit_str = str(MAX_PRS) if MAX_PRS is not None else "all"
    print(f"\n📦 Fetching {limit_str} merged PRs from {REPO_FULL}...")

    url = f"{GITHUB_API_BASE}/repos/{REPO_FULL}/pulls"

    # Deduplicated bidirectional fetch: newest-first catches recent activity;
    # oldest-first catches founding decisions (Dec 2018 onward).
    seen_pr_numbers: set[int] = set()
    raw_prs_deduped: list[dict] = []

    for sort, direction in [("updated", "desc"), ("created", "asc")]:
        batch = _paginate(url, {"state": "closed", "sort": sort, "direction": direction},
                          max_items=MAX_PRS)
        for pr in batch:
            num = pr.get("number")
            if num and num not in seen_pr_numbers:
                seen_pr_numbers.add(num)
                raw_prs_deduped.append(pr)

    raw_prs = raw_prs_deduped
    print(f"  Fetched {len(raw_prs)} unique PRs (after dedup)")

    records = []
    for i, pr in enumerate(raw_prs):
        if MAX_PRS is not None and len(records) >= MAX_PRS:
            break
        if not pr.get("merged_at"):
            continue

        author = (pr.get("user") or {}).get("login", "")
        if _is_bot(author):
            continue

        number = pr["number"]
        if (i + 1) % 50 == 0 or i == 0:
            limit_str = str(MAX_PRS) if MAX_PRS is not None else "all"
            print(f"  Processing PR #{number} ({len(records)+1}/{limit_str})...")

        # Fetch comments
        comments_url = f"{GITHUB_API_BASE}/repos/{REPO_FULL}/issues/{number}/comments"
        raw_comments = _paginate(comments_url)
        comments = []
        for c in raw_comments:
            c_author = (c.get("user") or {}).get("login", "")
            if _is_bot(c_author):
                continue
            comments.append({
                "author": c_author,
                "body": c.get("body", ""),
                "created_at": c.get("created_at", ""),
            })

        # Fetch review comments
        reviews_url = f"{GITHUB_API_BASE}/repos/{REPO_FULL}/pulls/{number}/comments"
        raw_reviews = _paginate(reviews_url)
        review_comments = []
        for r in raw_reviews:
            r_author = (r.get("user") or {}).get("login", "")
            if _is_bot(r_author):
                continue
            review_comments.append({
                "author": r_author,
                "body": r.get("body", ""),
                "path": r.get("path", ""),
                "created_at": r.get("created_at", ""),
            })

        body = pr.get("body") or ""
        linked_issues = _extract_linked_issues(body)
        labels = [lbl["name"] for lbl in (pr.get("labels") or [])]

        records.append({
            "type": "pr",
            "number": number,
            "title": pr.get("title", ""),
            "body": body,
            "author": author,
            "merged_at": pr["merged_at"],
            "html_url": pr.get("html_url", f"{REPO_URL}/pull/{number}"),
            "merge_commit_sha": pr.get("merge_commit_sha", ""),
            "labels": labels,
            "linked_issues": linked_issues,
            "comments": comments,
            "review_comments": review_comments,
        })

    print(f"  ✅ Collected {len(records)} merged PRs")
    return records


# ---------------------------------------------------------------------------
# Issue extraction
# ---------------------------------------------------------------------------

def build_issue_records(pr_records: list[dict] = None) -> list[dict]:
    """Fetch closed issues — linked issues from PRs + standalone issues."""
    print(f"\n📦 Fetching issues from {REPO_FULL}...")

    # Collect linked issue numbers from PRs
    linked_numbers = set()
    if pr_records:
        for pr in pr_records:
            linked_numbers.update(pr.get("linked_issues", []))
    print(f"  Found {len(linked_numbers)} linked issue numbers from PRs")

    records = []
    seen_numbers: set[int] = set()

    # Fetch linked issues first (by number)
    for num in sorted(linked_numbers):
        url = f"{GITHUB_API_BASE}/repos/{REPO_FULL}/issues/{num}"
        try:
            resp = _get(url)
            issue = resp.json()
        except Exception as e:
            print(f"  ⚠️  Could not fetch issue #{num}: {e}")
            continue

        # Skip if it's actually a PR (GitHub API returns PRs as issues too)
        if issue.get("pull_request"):
            continue

        author = (issue.get("user") or {}).get("login", "")
        if _is_bot(author):
            continue

        # Fetch comments
        comments_url = f"{GITHUB_API_BASE}/repos/{REPO_FULL}/issues/{num}/comments"
        raw_comments = _paginate(comments_url)
        comments = []
        for c in raw_comments:
            c_author = (c.get("user") or {}).get("login", "")
            if _is_bot(c_author):
                continue
            comments.append({
                "author": c_author,
                "body": c.get("body", ""),
                "created_at": c.get("created_at", ""),
            })

        labels = [lbl["name"] for lbl in (issue.get("labels") or [])]

        records.append({
            "type": "issue",
            "number": num,
            "title": issue.get("title", ""),
            "body": issue.get("body") or "",
            "author": author,
            "closed_at": issue.get("closed_at", ""),
            "html_url": issue.get("html_url", f"{REPO_URL}/issues/{num}"),
            "labels": labels,
            "comments": comments,
        })
        seen_numbers.add(num)

    # Fetch standalone closed issues (deduplicated bidirectional fetch)
    limit_str = str(MAX_ISSUES) if MAX_ISSUES is not None else "all"
    print(f"  Fetching {limit_str} standalone closed issues...")
    url = f"{GITHUB_API_BASE}/repos/{REPO_FULL}/issues"

    raw_issues_map: dict[int, dict] = {}  # number -> issue; automatic dedup
    for sort, direction in [("updated", "desc"), ("created", "asc")]:
        batch = _paginate(url, {"state": "closed", "sort": sort, "direction": direction},
                          max_items=MAX_ISSUES)
        for issue in batch:
            num = issue.get("number")
            if num and num not in raw_issues_map:
                raw_issues_map[num] = issue

    raw_issues = list(raw_issues_map.values())

    # Fetch targeted keywords via Search API
    TARGETED_KEYWORDS = ["422", "Depends", "Pydantic", "Starlette"]
    for kw in TARGETED_KEYWORDS:
        print(f"  Fetching targeted issues for keyword: {kw}")
        search_url = f"{GITHUB_API_BASE}/search/issues"
        q = f"{kw} in:body repo:{REPO_FULL} is:issue is:closed"
        try:
            resp = _get(search_url, params={"q": q, "per_page": 20})
            items = resp.json().get("items", [])
            for item in items:
                num = item.get("number")
                if num and num not in raw_issues_map:
                    raw_issues_map[num] = item
        except Exception as e:
            print(f"  ⚠️  Failed to fetch targeted issues for '{kw}': {e}")

    # Re-extract list after potential search additions
    raw_issues = list(raw_issues_map.values())

    for issue in raw_issues:
        if MAX_ISSUES is not None and len(records) >= MAX_ISSUES + len(linked_numbers):
            break
        # Skip PRs
        if issue.get("pull_request"):
            continue
        num = issue["number"]
        if num in seen_numbers:
            continue

        author = (issue.get("user") or {}).get("login", "")
        if _is_bot(author):
            continue

        # Fetch comments
        comments_url = f"{GITHUB_API_BASE}/repos/{REPO_FULL}/issues/{num}/comments"
        raw_comments = _paginate(comments_url)
        comments = []
        for c in raw_comments:
            c_author = (c.get("user") or {}).get("login", "")
            if _is_bot(c_author):
                continue
            comments.append({
                "author": c_author,
                "body": c.get("body", ""),
                "created_at": c.get("created_at", ""),
            })

        labels = [lbl["name"] for lbl in (issue.get("labels") or [])]

        records.append({
            "type": "issue",
            "number": num,
            "title": issue.get("title", ""),
            "body": issue.get("body") or "",
            "author": author,
            "closed_at": issue.get("closed_at", ""),
            "html_url": issue.get("html_url", f"{REPO_URL}/issues/{num}"),
            "labels": labels,
            "comments": comments,
        })
        seen_numbers.add(num)

    print(f"  ✅ Collected {len(records)} issues")
    return records


# ---------------------------------------------------------------------------
# Commit extraction
# ---------------------------------------------------------------------------

def build_commit_records(pr_records: list[dict] = None) -> list[dict]:
    """Fetch standalone commits (excluding PR merge commits)."""
    limit_str = str(MAX_COMMITS) if MAX_COMMITS is not None else "all"
    print(f"\n📦 Fetching {limit_str} standalone commits from {REPO_FULL}...")

    # Build set of merge commit SHAs to exclude (already represented via PRs)
    merge_shas = set()
    if pr_records:
        for pr in pr_records:
            sha = pr.get("merge_commit_sha", "")
            if sha:
                merge_shas.add(sha)
    print(f"  Excluding {len(merge_shas)} PR merge-commit SHAs")

    url = f"{GITHUB_API_BASE}/repos/{REPO_FULL}/commits"
    params = {"per_page": GITHUB_PER_PAGE}
    # Fetch newest-first (GitHub default). For full history, paginate to the end.
    extra = len(merge_shas) + 200  # over-fetch to account for filtered merge commits
    max_fetch = (MAX_COMMITS + extra) if MAX_COMMITS is not None else None
    raw_commits = _paginate(url, params, max_items=max_fetch)

    records = []
    for commit in raw_commits:
        if MAX_COMMITS is not None and len(records) >= MAX_COMMITS:
            break

        sha = commit.get("sha", "")
        if sha in merge_shas:
            continue

        author = ""
        if commit.get("author"):
            author = commit["author"].get("login", "")
        elif commit.get("commit", {}).get("author"):
            author = commit["commit"]["author"].get("name", "")

        if _is_bot(author):
            continue

        commit_detail = commit.get("commit", {})
        message = commit_detail.get("message", "")
        date = commit_detail.get("author", {}).get("date", "")

        # Fetch file list for this commit
        files_changed = []
        additions = 0
        deletions = 0
        try:
            detail_resp = _get(f"{GITHUB_API_BASE}/repos/{REPO_FULL}/commits/{sha}")
            detail = detail_resp.json()
            for f in detail.get("files", []):
                files_changed.append(f.get("filename", ""))
                additions += f.get("additions", 0)
                deletions += f.get("deletions", 0)
        except Exception:
            pass  # File list is nice-to-have, not critical

        short_sha = sha[:7]
        records.append({
            "type": "commit",
            "sha": sha,
            "short_sha": short_sha,
            "message": message,
            "author": author,
            "date": date,
            "html_url": f"{REPO_URL}/commit/{sha}",
            "files_changed": files_changed,
            "additions": additions,
            "deletions": deletions,
        })

    print(f"  ✅ Collected {len(records)} standalone commits")
    return records


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    if not GITHUB_TOKEN:
        print("⚠️  GITHUB_TOKEN not set. Unauthenticated requests are limited to 60/hr.")
        print("   Set GITHUB_TOKEN in .env or as an environment variable.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Phase 1a: PRs (resume if already saved, unless force-refresh)
    if RAW_PRS_PATH.exists() and not EXTRACT_FORCE_REFRESH:
        print(f"\n♻️  Found existing {RAW_PRS_PATH.name}, loading... (set EXTRACT_FORCE_REFRESH=True to re-fetch)")
        with open(RAW_PRS_PATH, encoding="utf-8") as f:
            pr_records = json.load(f)
        print(f"  Loaded {len(pr_records)} PRs from cache")
    else:
        if EXTRACT_FORCE_REFRESH and RAW_PRS_PATH.exists():
            print(f"\n🔄 Force-refresh enabled — re-fetching PRs (ignoring cache)...")
        pr_records = build_pr_records()
        RAW_PRS_PATH.write_text(json.dumps(pr_records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved → {RAW_PRS_PATH}")

    # Phase 1b: Issues (resume if already saved, unless force-refresh)
    if RAW_ISSUES_PATH.exists() and not EXTRACT_FORCE_REFRESH:
        print(f"\n♻️  Found existing {RAW_ISSUES_PATH.name}, loading... (set EXTRACT_FORCE_REFRESH=True to re-fetch)")
        with open(RAW_ISSUES_PATH, encoding="utf-8") as f:
            issue_records = json.load(f)
        print(f"  Loaded {len(issue_records)} issues from cache")
    else:
        if EXTRACT_FORCE_REFRESH and RAW_ISSUES_PATH.exists():
            print(f"\n🔄 Force-refresh enabled — re-fetching issues (ignoring cache)...")
        issue_records = build_issue_records(pr_records)
        RAW_ISSUES_PATH.write_text(json.dumps(issue_records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved → {RAW_ISSUES_PATH}")

    # Phase 1c: Commits (resume if already saved, unless force-refresh)
    if RAW_COMMITS_PATH.exists() and not EXTRACT_FORCE_REFRESH:
        print(f"\n♻️  Found existing {RAW_COMMITS_PATH.name}, loading... (set EXTRACT_FORCE_REFRESH=True to re-fetch)")
        with open(RAW_COMMITS_PATH, encoding="utf-8") as f:
            commit_records = json.load(f)
        print(f"  Loaded {len(commit_records)} commits from cache")
    else:
        if EXTRACT_FORCE_REFRESH and RAW_COMMITS_PATH.exists():
            print(f"\n🔄 Force-refresh enabled — re-fetching commits (ignoring cache)...")
        commit_records = build_commit_records(pr_records)
        RAW_COMMITS_PATH.write_text(json.dumps(commit_records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved → {RAW_COMMITS_PATH}")

    elapsed = time.time() - start
    print(f"\n🎉 Extraction complete in {elapsed/60:.1f} minutes")
    print(f"   {len(pr_records)} PRs | {len(issue_records)} issues | {len(commit_records)} commits")


if __name__ == "__main__":
    main()
