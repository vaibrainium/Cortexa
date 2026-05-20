"""
retry_missing.py — retry PDF acquisition for papers that came back without full text

Tries additional strategies on top of the original four sources:
  - Semantic Scholar open-access PDF field (different index than Unpaywall)
  - Europe PMC (broader OA coverage than US PMC)
  - DOI.org redirect + PDF sniffing
  - arXiv (some neuro papers have preprint versions)

Usage:
    python src/retry_missing.py
"""

import json
import re
import time
import hashlib
import requests
from pathlib import Path

RAW_DIR   = Path("/mnt/data/raw")
META_FILE = Path("/mnt/data/raw/metadata.jsonl")
HEADERS   = {"User-Agent": "Cortexa/0.1 (research tool; mailto:vaibhavt459@gmail.com)"}
SCIHUB_BASE = "https://sci-hub.ru"


# ── Load papers needing retry ──────────────────────────────────────────────

def load_missing() -> list[dict]:
    if not META_FILE.exists():
        return []
    missing = []
    for line in META_FILE.read_text().splitlines():
        try:
            r = json.loads(line)
            if not r.get("has_full_text"):
                missing.append(r)
        except Exception:
            pass
    return missing


def update_record(record: dict, pdf_path: str):
    """Update metadata.jsonl in place for a given pmid/title."""
    lines = META_FILE.read_text().splitlines()
    updated = []
    for line in lines:
        try:
            r = json.loads(line)
            match_key = (
                (r.get("pmid") and r.get("pmid") == record.get("pmid")) or
                (r.get("title", "").lower()[:50] == record.get("title", "").lower()[:50])
            )
            if match_key:
                r["pdf_path"]      = pdf_path
                r["has_full_text"] = True
                r["source"]        = r.get("source", "") + "+retry"
                updated.append(json.dumps(r))
            else:
                updated.append(line)
        except Exception:
            updated.append(line)
    META_FILE.write_text("\n".join(updated) + "\n")


def title_slug(title: str) -> str:
    return hashlib.md5(title.lower().encode()).hexdigest()[:12]


def download_pdf(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=30, headers=HEADERS, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 5_000 and b"%PDF" in r.content[:8]:
            dest.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"    Download error: {e}")
    return False


# ── Retry sources ──────────────────────────────────────────────────────────

def try_semantic_scholar(doi: str, title: str) -> str | None:
    """Try Semantic Scholar's own OA PDF index."""
    if not doi and not title:
        return None
    try:
        query = doi if doi else title
        field = "externalIds" if doi else "query"
        if doi:
            r = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                params={"fields": "openAccessPdf,title"},
                timeout=10,
            )
        else:
            r = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": title, "limit": 1, "fields": "openAccessPdf,title"},
                timeout=10,
            )
            data = r.json().get("data", [])
            if not data:
                return None
            r = type("R", (), {"json": lambda self: data[0], "status_code": 200})()

        if r.status_code != 200:
            return None
        data = r.json()
        oa = data.get("openAccessPdf")
        if oa and oa.get("url"):
            return oa["url"]
        return None
    except Exception as e:
        print(f"    S2 error: {e}")
        return None


def try_europe_pmc(doi: str, pmid: str) -> str | None:
    """Try Europe PMC — broader OA coverage than US PMC."""
    try:
        query = f"DOI:{doi}" if doi else f"EXT_ID:{pmid}"
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "resultType": "core"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("resultList", {}).get("result", [])
        if not results:
            return None
        result = results[0]
        pmcid = result.get("pmcid")
        if pmcid and result.get("isOpenAccess") == "Y":
            # construct Europe PMC PDF URL
            return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
        return None
    except Exception as e:
        print(f"    Europe PMC error: {e}")
        return None


def try_arxiv(title: str, author: str) -> str | None:
    """Try arXiv for preprint version."""
    try:
        import urllib.parse
        query = urllib.parse.quote(f"ti:{title} au:{author}")
        r = requests.get(
            f"http://export.arxiv.org/api/query?search_query={query}&max_results=1",
            timeout=10,
        )
        if r.status_code != 200:
            return None
        # parse arxiv id from atom response
        m = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", r.text)
        if m:
            arxiv_id = m.group(1).strip()
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return None
    except Exception as e:
        print(f"    arXiv error: {e}")
        return None


def try_scihub(doi: str) -> str | None:
    """Sci-Hub with citation_pdf_url parsing."""
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

def retry_all():
    missing = load_missing()
    if not missing:
        print("No missing papers found in metadata.jsonl")
        return

    print(f"Papers to retry: {len(missing)}")
    print(f"Sources: Sci-Hub -> Semantic Scholar -> Europe PMC -> arXiv\n")

    recovered = 0

    for record in missing:
        title  = record.get("title", record.get("title", ""))
        author = record.get("first_author", "")
        doi    = record.get("doi")
        pmid   = record.get("pmid")
        pmcid  = record.get("pmcid")
        year   = record.get("year", "")

        print(f"\n[{author} {year}] {title[:60]}")
        print(f"  DOI: {doi or 'none'}  |  PMCID: {pmcid or 'none'}")

        slug     = title_slug(title) if title else hashlib.md5(str(doi).encode()).hexdigest()[:12]
        pdf_path = None

        # 1. Sci-Hub (retry with updated parser)
        print(f"  [1] Sci-Hub...")
        url = try_scihub(doi)
        if url:
            dest = RAW_DIR / f"{slug}_sh2.pdf"
            print(f"      URL found, downloading...")
            if download_pdf(url, dest):
                pdf_path = str(dest)
                print(f"      Saved: {dest.name}")

        # 2. Semantic Scholar
        if not pdf_path:
            print(f"  [2] Semantic Scholar...")
            url = try_semantic_scholar(doi, title)
            if url:
                dest = RAW_DIR / f"{slug}_s2.pdf"
                print(f"      URL found, downloading...")
                if download_pdf(url, dest):
                    pdf_path = str(dest)
                    print(f"      Saved: {dest.name}")
            else:
                print(f"      No OA PDF found")

        # 3. Europe PMC
        if not pdf_path:
            print(f"  [3] Europe PMC...")
            url = try_europe_pmc(doi, pmid)
            if url:
                dest = RAW_DIR / f"{slug}_epmc.pdf"
                print(f"      URL found, downloading...")
                if download_pdf(url, dest):
                    pdf_path = str(dest)
                    print(f"      Saved: {dest.name}")
            else:
                print(f"      No OA PDF found")

        # 4. arXiv preprint
        if not pdf_path:
            print(f"  [4] arXiv...")
            url = try_arxiv(title, author)
            if url:
                dest = RAW_DIR / f"{slug}_arxiv.pdf"
                print(f"      URL found, downloading...")
                if download_pdf(url, dest):
                    pdf_path = str(dest)
                    print(f"      Saved: {dest.name}")
            else:
                print(f"      No preprint found")

        if pdf_path:
            update_record(record, pdf_path)
            recovered += 1
            print(f"  Recovered!")
        else:
            print(f"  Still no PDF — will need manual download")

        time.sleep(1.5)

    print(f"\n{'─'*55}")
    print(f"Recovered: {recovered}/{len(missing)} papers")
    if recovered < len(missing):
        remaining = len(missing) - recovered
        print(f"Still missing: {remaining} papers")
        print(f"These will need manual download through institutional access")
        print(f"Drop PDFs into /mnt/data/raw/ and re-run ingest.py")

if __name__ == '__main__':
    retry_all()
