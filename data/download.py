"""Download FinQA dataset from the official GitHub repository."""

import os
import requests
import json

REPO_BASE = "https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset"
FILES = ["train.json", "dev.json", "test.json"]

def download_finqa(output_dir: str = "data/raw"):
    os.makedirs(output_dir, exist_ok=True)

    for fname in FILES:
        url = f"{REPO_BASE}/{fname}"
        output_path = os.path.join(output_dir, fname)

        if os.path.exists(output_path):
            print(f"[skip] {fname} already exists at {output_path}")
            continue

        print(f"[download] {fname} from {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        data = resp.json()
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  -> saved {len(data)} examples to {output_path}")

    print("[done] FinQA dataset downloaded.")

if __name__ == "__main__":
    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    download_finqa(os.path.join(project_root, "data", "raw"))
