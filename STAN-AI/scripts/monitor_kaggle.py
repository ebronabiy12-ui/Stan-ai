#!/usr/bin/env python3
"""
Monitor and manage the STAN AI Kaggle GPU training run.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

TOKEN_PATH = Path.home() / ".kaggle" / "kaggle.json"
KERNEL_SLUG = "stan-ai-gpu-training"

def get_credentials():
    if not TOKEN_PATH.exists():
        print(f"Error: {TOKEN_PATH} not found.")
        sys.exit(1)
    data = json.loads(TOKEN_PATH.read_text())
    return data["username"], data["key"]

def api_request(path: str) -> dict:
    username, token = get_credentials()
    url = f"https://www.kaggle.com/api/v1{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def check_status() -> dict:
    username, _ = get_credentials()
    return api_request(f"/kernels/status?userName={username}&kernelSlug={KERNEL_SLUG}")

def download_output(target_dir: Path):
    username, token = get_credentials()
    url = f"https://www.kaggle.com/api/v1/kernels/output?userName={username}&kernelSlug={KERNEL_SLUG}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    print(f"[Kaggle] Downloading output bundle from {url}...")
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            zf.extractall(target_dir)
            print(f"[Kaggle] Extracted files: {zf.namelist()}")
    except zipfile.BadZipFile:
        # Might be a single raw file or json response
        print(f"[Kaggle] Raw output: {data[:200]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Poll status continuously")
    parser.add_argument("--download", action="store_true", help="Download output artifacts")
    parser.add_argument("--output-dir", default="STAN-AI/checkpoints", help="Output directory")
    args = parser.parse_args()

    username, _ = get_credentials()
    print(f"Kaggle User: {username}")
    print(f"Kernel URL:  https://www.kaggle.com/code/{username}/{KERNEL_SLUG}")

    if args.download:
        download_output(Path(args.output_dir))
        return

    if args.watch:
        print("[Kaggle] Monitoring training progress (Ctrl+C to exit)...")
        while True:
            st = check_status()
            current = st.get("status", "unknown")
            print(f"[{time.strftime('%H:%M:%S')}] Status: {current}")
            if current in ("complete", "completeWithErrors", "error", "failed"):
                if st.get("hasFailureMessage"):
                    print("Failure message:", st.get("failureMessage"))
                if current == "complete":
                    print("[Kaggle] Training complete! Run --download to fetch checkpoints.")
                break
            time.sleep(30)
    else:
        st = check_status()
        print("Status:", st.get("status"))
        if st.get("hasFailureMessage"):
            print("Failure message:", st.get("failureMessage"))

if __name__ == "__main__":
    main()
