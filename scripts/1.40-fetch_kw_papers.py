"""
fetch_kw_papers.py — fetch PDFs for keyword-discovered papers

Reads kw_metadata.jsonl and attempts PDF acquisition using the same
source chain as fetch_papers.py:
  1. Unpaywall (use oa_url from S2 first if available)
  2. PubMed Central OA API
  3. Sci-Hub

Run build_kw_list.py first and review kw_metadata.jsonl before running this.

Usage:
    python src/fetch_kw_papers.py
"""

import json
import re
import time
import requests
from pathlib import Path

RAW_DIR      = Path("/mnt/data/raw")
KW_META_FILE = Path("/mnt/data/raw/kw_metadata.jsonl")
HEADERS      = {"User-Agent": "Cortexa/0.1 (research tool; mailto:vaibhavt459@gmail.com)"}
SCIHUB_BASE  = "https://sci-hub.ru"
UNPAYWALL_EMAIL = "vaibhavt459@gmail.com"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── PDF helpers (same as fetch_papers.py) ─────────────────────────────────

def download_pdf(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=30, headers=HEADERS, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 5_000 and b"%PDF" in r.content[:8]:
            dest.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"    Download error: {e}")
    return False


def unpaywall_fetch(doi: str) -> str | None:
    if not doi:
        return None
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": UNPAYWALL_EMAIL}, timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        best = data.get("best_oa_location")
        if best and best.get("url_for_pdf"):
            return best["url_for_pdf"]
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
        return None
    except Exception as e:
        print(f"    Unpaywall error: {e}")
        return None


def pmc_fetch(doi: str) -> str | None:
    """Look up PMCID from DOI, then use OA API."""
    if not doi:
        return None
    try:
        # get PMCID from DOI
        r = requests.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
            params={"ids": doi, "format": "json"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        records = r.json().get("records", [])
        if not records:
            return None
        pmcid = records[0].get("pmcid")
        if not pmcid:
            return None

        # OA API for PDF link
        time.sleep(0.4)
        r = requests.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
            params={"id": pmcid}, timeout=10,
        )
        m = re.search(r'href="(ftp://[^"]+\.pdf)"', r.text)
        if m:
            return m.group(1).replace(
                "ftp://ftp.ncbi.nlm.nih.gov",
                "https://ftp.ncbi.nlm.nih.gov"
            )
        m = re.search(r'href="(https://[^"]+\.pdf)"', r.text)
        if m:
            return m.group(1)
        return None
    except Exception as e:
        print(f"    PMC error: {e}")
        return None


def scihub_fetch(doi: str) -> str | None:
    if not doi:
        return None
    try:
        r = requests.get(f"{SCIHUB_BASE}/{doi}", timeout=15, headers=HEADERS)
        if r.status_code != 200:
            return None
        html = r.text
        m = re.search(r'citation_pdf_url[^>]+content="([^"]+)"', html)
        if m:
            path = m.group(1).strip()
            return path if path.startswith("http") else SCIHUB_BASE + path
        m = re.search(r'<iframe[^>]+src="([^"]+\.pdf[^"]*)"', html)
        if m:
            path = m.group(1).strip()
            if path.startswith("//"):
                return "https:" + path
            if path.startswith("/"):
                return SCIHUB_BASE + path
            return path
        m = re.search(r'"url"\s*:\s*"([^"]+\.pdf)"', html)
        if m:
            path = m.group(1).strip()
            return SCIHUB_BASE + path if path.startswith("/") else path
        return None
    except Exception as e:
        print(f"    Sci-Hub error: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────

def fetch_all():
    if not KW_META_FILE.exists():
        print(f"No kw_metadata.jsonl found. Run build_kw_list.py first.")
        return

    records = [json.loads(l) for l in KW_META_FILE.read_text().splitlines() if l.strip()]
    pending = [r for r in records if not r.get("has_full_text")]
    already = len(records) - len(pending)

    print(f"Total in kw list:  {len(records)}")
    print(f"Already fetched:   {already}")
    print(f"To fetch:          {len(pending)}")
    print(f"Sources:           OA URL -> Unpaywall -> PMC -> Sci-Hub\n")

    stats = {"pdf": 0, "abstract_only": 0}

    for i, record in enumerate(pending):
        title = record.get("title", "")
        doi   = record.get("doi", "")
        oa_url = record.get("oa_url")
        subfield = record.get("subfield", "")

        import hashlib
        slug = hashlib.md5(title.lower().encode()).hexdigest()[:12]

        print(f"\n[{i+1:03d}/{len(pending)}] {record.get('authors', ['?'])[0]} "
              f"{record.get('year', '')} ({subfield})")
        print(f"  {title[:65]}")

        pdf_path = None

        # 0. Use OA URL from S2 directly if available (fastest)
        if oa_url and not pdf_path:
            print(f"  [0] S2 OA URL...")
            dest = RAW_DIR / f"{slug}_s2.pdf"
            if not dest.exists() and download_pdf(oa_url, dest):
                pdf_path = str(dest)
                print(f"      Saved: {dest.name}")
            elif dest.exists():
                pdf_path = str(dest)

        # 1. Unpaywall
        if not pdf_path:
            print(f"  [1] Unpaywall...")
            url = unpaywall_fetch(doi)
            if url:
                dest = RAW_DIR / f"{slug}_uw.pdf"
                if download_pdf(url, dest):
                    pdf_path = str(dest)
                    print(f"      Saved: {dest.name}")
                else:
                    print(f"      Download failed")
            else:
                print(f"      No OA PDF")

        # 2. PMC
        if not pdf_path:
            print(f"  [2] PMC...")
            url = pmc_fetch(doi)
            if url:
                dest = RAW_DIR / f"{slug}_pmc.pdf"
                if download_pdf(url, dest):
                    pdf_path = str(dest)
                    print(f"      Saved: {dest.name}")
                else:
                    print(f"      Download failed")
            else:
                print(f"      Not OA on PMC")

        # 3. Sci-Hub
        if not pdf_path:
            print(f"  [3] Sci-Hub...")
            url = scihub_fetch(doi)
            if url:
                dest = RAW_DIR / f"{slug}_sh.pdf"
                if download_pdf(url, dest):
                    pdf_path = str(dest)
                    print(f"      Saved: {dest.name}")
                else:
                    print(f"      Download failed")
            else:
                print(f"      Not found")

        # update record in memory
        record["pdf_path"]      = pdf_path
        record["has_full_text"] = pdf_path is not None

        if pdf_path:
            stats["pdf"] += 1
        else:
            stats["abstract_only"] += 1
            print(f"  Abstract only")

        time.sleep(1.2)

    # rewrite kw_metadata.jsonl with updated pdf_path fields
    with open(KW_META_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'─'*55}")
    print(f"Done.")
    print(f"  PDFs fetched:  {stats['pdf']}")
    print(f"  Abstract only: {stats['abstract_only']}")
    print(f"\nNext step: python src/ingest.py")
    print(f"(ingest.py will process both metadata.jsonl and kw_metadata.jsonl)")


if __name__ == "__main__":
    fetch_all()
