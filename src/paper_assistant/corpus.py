"""Download the PDFs listed in data/papers.csv (the manifest written by scripts/download_papers.py).

This is what makes the project reproducible from a clean clone: the Kaggle CSV is only needed
to *choose* papers; once data/papers.csv is committed, anyone can re-fetch the same 50 PDFs.
"""

from __future__ import annotations

import csv
import time
import urllib.request
from pathlib import Path

from paper_assistant import config

USER_AGENT = "rag-research-assistant/0.1 (student project)"
DELAY_SECONDS = 3  # arXiv asks automated clients to wait between requests


def download_pdf(url: str, dest: Path, retries: int = 3) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            dest.write_bytes(data)
            return True
        except Exception as error:  # noqa: BLE001 - network, HTTP and content errors alike
            print(f"   attempt {attempt} failed: {error}")
            time.sleep(DELAY_SECONDS * attempt)
    return False


def fetch_corpus(manifest: Path | None = None, pdf_dir: Path | None = None) -> list[str]:
    """Download every PDF in the manifest that is not already on disk. Returns failed ids."""
    manifest = Path(manifest or config.MANIFEST_PATH)
    pdf_dir = Path(pdf_dir or config.PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    failed = []
    for i, row in enumerate(rows, start=1):
        dest = pdf_dir / f"{row['paper_id']}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        print(f"[{i}/{len(rows)}] downloading {dest.name}")
        if not download_pdf(row["pdf_url"], dest):
            failed.append(row["paper_id"])
        time.sleep(DELAY_SECONDS)
    return failed
