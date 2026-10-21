"""Pick ~50 explainable-AI papers from the arXiv CSV and download their PDFs.

Usage (from the repo root):
    python scripts/download_papers.py --dry-run   # just show the chosen papers
    python scripts/download_papers.py             # download them (about 3 minutes)

Safe to re-run: PDFs that already exist are skipped.
"""
import argparse
import csv
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "archive (1)" / "arxiv_ai.csv"
PDF_DIR = ROOT / "data" / "pdfs"
MANIFEST = ROOT / "data" / "papers.csv"

KEYWORDS = ("explainab", "interpretab")  # matches explainable/explainability, interpretable/interpretability
N_PAPERS = 50
MIN_YEAR = 2016
SEED = 42            # fixed seed -> the same 50 papers every run (reproducible)
DELAY_SECONDS = 3    # arXiv asks automated clients to wait between requests
USER_AGENT = "rag-research-assistant/0.1 (student project; mailto:aatalhatalukder@gmail.com)"


def clean(text: str) -> str:
    """Collapse the newlines and runs of spaces that arXiv leaves in titles."""
    return " ".join(text.split())


def arxiv_id(entry_id: str) -> str:
    """'http://arxiv.org/abs/1602.04938v5' -> '1602.04938v5'."""
    return entry_id.rsplit("/abs/", 1)[-1]


def select_papers() -> list[dict]:
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = [
        r for r in rows
        if int(r["published"][:4]) >= MIN_YEAR
        and any(k in r["title"].lower() for k in KEYWORDS)
    ]
    print(f"{len(rows)} papers in CSV, {len(candidates)} match {KEYWORDS} in the title (>= {MIN_YEAR})")
    rng = random.Random(SEED)
    chosen = rng.sample(candidates, min(N_PAPERS, len(candidates)))
    chosen.sort(key=lambda r: r["published"])
    papers = []
    for r in chosen:
        aid = arxiv_id(r["entry_id"])
        papers.append({
            "paper_id": aid.replace("/", "_"),
            "title": clean(r["title"]),
            "published": r["published"][:10],
            "pdf_url": r["pdf_url"].replace("http://", "https://") + ".pdf",
        })
    return papers


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            dest.write_bytes(data)
            return True
        except Exception as e:  # network errors, 4xx/5xx, bad content
            print(f"   attempt {attempt} failed: {e}")
            time.sleep(DELAY_SECONDS * attempt)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    papers = select_papers()
    for p in papers:
        print(f"{p['published']}  {p['paper_id']:<16} {p['title'][:80]}")
    if args.dry_run:
        return

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, p in enumerate(papers, 1):
        dest = PDF_DIR / f"{p['paper_id']}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[{i}/{len(papers)}] have {dest.name}")
            continue
        print(f"[{i}/{len(papers)}] downloading {dest.name}")
        if not download(p["pdf_url"], dest):
            failed.append(p["paper_id"])
        time.sleep(DELAY_SECONDS)

    ok = [p for p in papers if (PDF_DIR / f"{p['paper_id']}.pdf").exists()]
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["paper_id", "title", "published", "pdf_url"])
        w.writeheader()
        w.writerows(ok)
    print(f"\nDone: {len(ok)} PDFs in {PDF_DIR}, manifest at {MANIFEST}")
    if failed:
        print("Failed (re-run the script to retry):", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
