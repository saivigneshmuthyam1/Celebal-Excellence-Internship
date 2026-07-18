"""
bootstrap_early_history.py
Fetches founding-era commits (Dec 2018 through 2021) from GitHub API,
merges into existing raw_commits.json, then rebuilds chunks + FAISS index.
Runs in ~10-20 min instead of hours.
"""
import json, sys, time, re, subprocess
sys.path.insert(0, "src")

import requests
from config import (
    GITHUB_API_BASE, GITHUB_TOKEN, GITHUB_PER_PAGE,
    RAW_COMMITS_PATH, REPO_FULL, REPO_URL, BOT_AUTHORS
)

session = requests.Session()
if GITHUB_TOKEN:
    session.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
session.headers.update({"Accept": "application/vnd.github.v3+json"})

def _check_rate_limit(resp):
    remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
    if remaining < 50:
        reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset_ts - int(time.time()), 1) + 5
        print(f"  Sleeping {wait}s (rate limit: {remaining} left)...")
        time.sleep(wait)

def _get(url, params=None):
    for attempt in range(5):
        try:
            resp = session.get(url, params=params, timeout=30)
        except Exception as e:
            wait = min(2**attempt * 5, 60)
            print(f"  Connection error: {e}. Retry in {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            _check_rate_limit(resp)
            return resp
        if resp.status_code in (403, 429):
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_ts - int(time.time()), 1) + 5
            print(f"  {resp.status_code} - sleeping {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed: {url}")

def _is_bot(login):
    if not login: return False
    if login in BOT_AUTHORS: return True
    if login.lower().endswith("[bot]"): return True
    return False

def fetch_window(since, until):
    url = f"{GITHUB_API_BASE}/repos/{REPO_FULL}/commits"
    params = {"since": since, "until": until, "per_page": GITHUB_PER_PAGE}
    all_commits, page = [], 1
    while url:
        print(f"    page {page}...", end=" ", flush=True)
        resp = _get(url, params=params)
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            print()
            break
        all_commits.extend(batch)
        print(f"{len(batch)} commits")
        link_header = resp.headers.get("Link", "")
        m = re.search(r"<([^>]+)>;\s*rel=\"next\"", link_header)
        url = m.group(1) if m else None
        params = None
        page += 1
    return all_commits

def build_record(commit):
    sha = commit.get("sha", "")
    author = ""
    if commit.get("author"):
        author = commit["author"].get("login", "")
    if not author:
        author = commit.get("commit", {}).get("author", {}).get("name", "")
    if _is_bot(author):
        return None
    commit_detail = commit.get("commit", {})
    message = commit_detail.get("message", "")
    date = commit_detail.get("author", {}).get("date", "")
    files_changed, additions, deletions = [], 0, 0
    try:
        detail = _get(f"{GITHUB_API_BASE}/repos/{REPO_FULL}/commits/{sha}").json()
        for f in detail.get("files", []):
            files_changed.append(f.get("filename", ""))
            additions += f.get("additions", 0)
            deletions += f.get("deletions", 0)
    except Exception:
        pass
    return {
        "type": "commit", "sha": sha, "short_sha": sha[:7],
        "message": message, "author": author, "date": date,
        "html_url": f"{REPO_URL}/commit/{sha}",
        "files_changed": files_changed, "additions": additions, "deletions": deletions,
    }

def main():
    print("\n=== Targeted Early-History Bootstrap ===")
    existing = []
    if RAW_COMMITS_PATH.exists():
        with open(RAW_COMMITS_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing commits")
    existing_shas = {c["sha"] for c in existing}

    windows = [
        ("2018-12-01T00:00:00Z", "2019-06-01T00:00:00Z"),
        ("2019-06-01T00:00:00Z", "2019-12-31T23:59:59Z"),
        ("2020-01-01T00:00:00Z", "2020-12-31T23:59:59Z"),
        ("2021-01-01T00:00:00Z", "2021-12-31T23:59:59Z"),
        ("2022-01-01T00:00:00Z", "2022-12-31T23:59:59Z"),
        ("2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z"),
        ("2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z"),
        ("2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z"),
    ]

    new_records = []
    for since, until in windows:
        label = f"{since[:10]} -> {until[:10]}"
        print(f"\n[Window] {label}")
        raw = fetch_window(since, until)
        added = 0
        for commit in raw:
            sha = commit.get("sha", "")
            if sha in existing_shas:
                continue
            r = build_record(commit)
            if r:
                new_records.append(r)
                existing_shas.add(sha)
                added += 1
        print(f"  Added {added} new non-bot commits")

    print(f"\nTotal new commits: {len(new_records)}")
    all_commits = existing + new_records
    all_commits.sort(key=lambda c: c.get("date", ""), reverse=True)
    RAW_COMMITS_PATH.write_text(
        json.dumps(all_commits, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(all_commits)} total commits")

    oldest = min(all_commits, key=lambda c: c.get("date", "9999"))
    print(f"\nOldest commit now:")
    print(f"  SHA:     {oldest['sha'][:12]}")
    print(f"  Date:    {oldest['date'][:10]}")
    print(f"  Author:  {oldest['author']}")
    print(f"  Message: {oldest['message'][:80]}")

    print("\nRebuilding chunks...")
    r = subprocess.run([sys.executable, "src/chunk.py"], capture_output=True, text=True)
    if r.returncode == 0:
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        print("\n".join(lines[-5:]))
    else:
        print("ERROR:", r.stderr[-300:])
        return

    print("\nRebuilding FAISS index...")
    r = subprocess.run([sys.executable, "src/build_index.py"], capture_output=True, text=True)
    if r.returncode == 0:
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        print("\n".join(lines[-5:]))
    else:
        print("ERROR:", r.stderr[-300:])
        return

    print("\nDone! Restart Streamlit to use the updated index.")

if __name__ == "__main__":
    main()
