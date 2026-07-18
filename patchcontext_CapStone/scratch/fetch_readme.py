import json
import requests
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
CHUNKS_PATH = DATA_DIR / "chunks.json"

def main():
    print("Fetching README from FastAPI...")
    resp = requests.get("https://raw.githubusercontent.com/fastapi/fastapi/master/README.md")
    resp.raise_for_status()
    readme_text = resp.text

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_text(readme_text)

    print(f"Split README into {len(splits)} chunks.")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Remove any existing README chunks
    chunks = [c for c in chunks if c["metadata"].get("id") != "README"]

    for i, split in enumerate(splits):
        chunks.append({
            "text": f"Repository README (FastAPI overview): {split}",
            "metadata": {
                "id": "README",
                "url": "https://github.com/fastapi/fastapi/blob/master/README.md",
                "source_type": "documentation",
                "author": "tiangolo",
                "date": "2026-07-18",
                "labels": ["documentation"]
            }
        })

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print("Appended README chunks to chunks.json.")

if __name__ == "__main__":
    main()
