"""
fetch_papers.py — Cortexa anchor paper fetcher

Source priority for each paper:
  1. PubMed E-utilities  — metadata + abstract + DOI + PMCID
  2. Unpaywall           — open-access PDF lookup by DOI
  3. PubMed Central      — free full text via OA API if PMCID exists
  4. Sci-Hub             — last resort fallback (sci-hub.ru)

Usage (inside container):
    python src/fetch_papers.py

Outputs:
    /mnt/data/raw/           PDFs where available
    /mnt/data/raw/metadata.jsonl one record per paper
"""

import json
import time
import hashlib
import requests
import re
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────

RAW_DIR   = Path("/mnt/data/raw")
META_FILE = Path("/mnt/data/raw/metadata.jsonl")
RAW_DIR.mkdir(parents=True, exist_ok=True)

UNPAYWALL_EMAIL = "vaibhavt459@gmail.com"
PUBMED_BASE     = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNPAYWALL_BASE  = "https://api.unpaywall.org/v2"
SCIHUB_BASE     = "https://sci-hub.ru"

HEADERS = {"User-Agent": "Cortexa/0.1 (research tool; mailto:vaibhavt459@gmail.com)"}

# ── Anchor papers ──────────────────────────────────────────────────────────

ANCHOR_PAPERS = [
    # 1. Value-based decision making
    ("framework neurobiology value-based decision making",                       "Rangel",         2008, "value_based_dm",     "REVIEW"),
    ("neurobiology economic choice good-based model",                            "Padoa-Schioppa", 2011, "value_based_dm",     "REVIEW"),
    ("orbitofrontal cortex encodes willingness to pay",                          "Plassmann",      2007, "value_based_dm",     "LANDMARK"),
    ("neurons orbitofrontal cortex encode economic value",                       "Padoa-Schioppa", 2006, "value_based_dm",     "LANDMARK"),
    ("neuro-computational model economic decisions",                             "Rustichini",     2015, "value_based_dm",     "BRIDGE"),

    # 2. Perceptual decision making
    ("neural basis decision making annual review neuroscience",                  "Gold",           2007, "perceptual_dm",      "REVIEW"),
    ("neural basis perceptual decision lateral intraparietal",                   "Shadlen",        2001, "perceptual_dm",      "LANDMARK"),
    ("analysis visual motion neuronal psychophysical performance",               "Britten",        1992, "perceptual_dm",      "LANDMARK"),
    ("do humans make good decisions",                                            "Summerfield",    2015, "perceptual_dm",      "BRIDGE"),
    ("cost accumulating evidence perceptual decision making",                    "Drugowitsch",    2012, "perceptual_dm",      "BRIDGE"),

    # 3. Reinforcement learning in the brain
    ("model-based model-free pavlovian reward learning",                         "Dayan",          2014, "rl_brain",           "REVIEW"),
    ("reinforcement learning in the brain",                                      "Niv",            2009, "rl_brain",           "REVIEW"),
    ("neural substrate prediction and reward",                                   "Schultz",        1997, "rl_brain",           "LANDMARK"),
    ("cortical substrates exploratory decisions humans",                         "Daw",            2006, "rl_brain",           "LANDMARK"),
    ("ubiquity model-based reinforcement learning",                              "Doll",           2012, "rl_brain",           "BRIDGE"),

    # 4. Drift-diffusion / computational models
    ("diffusion decision model theory data",                                     "Ratcliff",       2008, "drift_diffusion",    "REVIEW"),
    ("physics optimal decision making",                                          "Bogacz",         2006, "drift_diffusion",    "REVIEW"),
    ("visual fixations computation value simple choice",                         "Krajbich",       2010, "drift_diffusion",    "BRIDGE"),
    ("time course perceptual choice leaky competing accumulator",                "Usher",          2001, "drift_diffusion",    "LANDMARK"),
    ("HDDM hierarchical bayesian estimation diffusion decision",                 "Wiecki",         2013, "drift_diffusion",    "BRIDGE"),

    # 5. Neuroeconomics
    ("neuroeconomics how neuroscience can inform economics",                     "Camerer",        2005, "neuroeconomics",     "REVIEW"),
    ("prospect theory analysis decision under risk",                             "Kahneman",       1979, "neuroeconomics",     "LANDMARK"),
    ("value normalization in decision making",                                   "Rangel",         2012, "neuroeconomics",     "BRIDGE"),
    ("social neuroeconomics neural circuitry social preferences",                "Fehr",           2007, "neuroeconomics",     "BRIDGE"),

    # 6. Cognitive control
    ("frontal cortex hierarchical control behavior",                             "Badre",          2018, "cognitive_control",  "REVIEW"),
    ("expected value of control anterior cingulate",                             "Shenhav",        2013, "cognitive_control",  "REVIEW"),
    ("integrative theory prefrontal cortex function",                            "Miller",         2001, "cognitive_control",  "LANDMARK"),
    ("motivation cognitive control behavior neural mechanism",                   "Botvinick",      2015, "cognitive_control",  "BRIDGE"),
    ("uncertainty-based competition prefrontal striatal systems",                "Daw",            2005, "cognitive_control",  "BRIDGE"),

    # 7. Confidence and metacognition
    ("neural basis metacognitive ability",                                       "Fleming",        2012, "confidence_metacog", "REVIEW"),
    ("metacognition human decision-making",                                      "Yeung",          2012, "confidence_metacog", "REVIEW"),
    ("neural correlates computation decision confidence",                        "Kepecs",         2008, "confidence_metacog", "LANDMARK"),
    ("confidence and certainty distinct probabilistic quantities",               "Pouget",         2016, "confidence_metacog", "BRIDGE"),
    ("foundations human reasoning prefrontal cortex",                           "Donoso",         2014, "confidence_metacog", "BRIDGE"),

    # 8. SC and subcortical decision making
    ("visual-motor function primate superior colliculus",                        "Wurtz",          1980, "sc_subcortical",     "REVIEW"),
    ("neural substrates sensory-guided locomotor decisions superior colliculus", "Felsen",         2008, "sc_subcortical",     "REVIEW"),
    ("modulation neuronal activity superior colliculus target probability",      "Basso",          1998, "sc_subcortical",     "LANDMARK"),
    ("causal role primate superior colliculus computation evidence perceptual",  "Jun",            2021, "sc_subcortical",     "LANDMARK"),
    ("role superior colliculus decision criteria",                               "Crapse",         2018, "sc_subcortical",     "LANDMARK"),
    ("superior colliculus visual spatial attention",                             "Krauzlis",       2013, "sc_subcortical",     "BRIDGE"),
    ("integrative role superior colliculus selecting targets movements",         "Felsen",         2012, "sc_subcortical",     "BRIDGE"),
    ("primate superior colliculus causally engaged abstract higher-order",       "Duan",           2024, "sc_subcortical",     "BRIDGE"),
    ("superior colliculus inactivation biases target choice mice",               "McAlonan",       2020, "sc_subcortical",     "BRIDGE"),
    ("superior colliculus encodes alternations spatial target selections",       "Kim",            2008, "sc_subcortical",     "BRIDGE"),
]

# ── Helpers ────────────────────────────────────────────────────────────────

def title_slug(title: str) -> str:
    return hashlib.md5(title.lower().encode()).hexdigest()[:12]


def load_seen() -> set:
    seen = set()
    if META_FILE.exists():
        for line in META_FILE.read_text().splitlines():
            try:
                r = json.loads(line)
                seen.add(f"{r['first_author'].lower()}_{r['year']}")
            except Exception:
                pass
    return seen


def save_meta(record: dict):
    with open(META_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def download_pdf(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=30, headers=HEADERS, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 5_000:
            if b"%PDF" in r.content[:8]:
                dest.write_bytes(r.content)
                return True
            else:
                print(f"    Not a valid PDF")
    except Exception as e:
        print(f"    Download error: {e}")
    return False


# ── Source 1: PubMed ───────────────────────────────────────────────────────

def pubmed_fetch(title: str, author: str, year: int) -> dict | None:
    query = f"{title} {author}[author]"
    try:
        ids = []
        for params in [
            {"db": "pubmed", "term": query, "retmax": 3, "retmode": "json",
             "datetype": "pdat", "mindate": str(year-1), "maxdate": str(year+1)},
            {"db": "pubmed", "term": query, "retmax": 3, "retmode": "json"},
        ]:
            r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=10)
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                break

        if not ids:
            return None

        time.sleep(0.4)

        r = requests.get(f"{PUBMED_BASE}/esummary.fcgi",
                         params={"db": "pubmed", "id": ids[0], "retmode": "json"},
                         timeout=10)
        s = r.json().get("result", {}).get(ids[0], {})

        doi   = next((x["value"] for x in s.get("articleids", []) if x["idtype"] == "doi"),  None)
        pmcid = next((x["value"] for x in s.get("articleids", []) if x["idtype"] == "pmc"),  None)

        time.sleep(0.4)

        r = requests.get(f"{PUBMED_BASE}/efetch.fcgi",
                         params={"db": "pubmed", "id": ids[0],
                                 "rettype": "abstract", "retmode": "text"},
                         timeout=10)
        abstract = r.text.strip()

        return {
            "pmid":     ids[0],
            "pmcid":    pmcid,
            "doi":      doi,
            "title":    s.get("title", ""),
            "authors":  [a["name"] for a in s.get("authors", [])],
            "year":     s.get("pubdate", str(year))[:4],
            "journal":  s.get("source", ""),
            "abstract": abstract,
        }

    except Exception as e:
        print(f"    PubMed error: {e}")
        return None


# ── Source 2: Unpaywall ────────────────────────────────────────────────────

def unpaywall_fetch(doi: str) -> str | None:
    if not doi:
        return None
    try:
        r = requests.get(f"{UNPAYWALL_BASE}/{doi}",
                         params={"email": UNPAYWALL_EMAIL}, timeout=10)
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


# ── Source 3: PubMed Central OA API ───────────────────────────────────────

def pmc_fetch(pmcid: str) -> str | None:
    if not pmcid:
        return None
    pmcid_clean = pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"
    try:
        r = requests.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
            params={"id": pmcid_clean}, timeout=10,
        )
        if r.status_code != 200:
            return None
        # look for PDF link: <link format="pdf" href="ftp://..."/>
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


# ── Source 4: Sci-Hub ──────────────────────────────────────────────────────

def scihub_fetch(doi: str) -> str | None:
    if not doi:
        return None
    try:
        r = requests.get(f"{SCIHUB_BASE}/{doi}", timeout=15, headers=HEADERS)
        if r.status_code != 200:
            return None
        html = r.text

        # pattern 1: citation_pdf_url meta tag (most reliable)
        m = re.search(r'citation_pdf_url[^>]+content="([^"]+)"', html)
        if m:
            path = m.group(1).strip()
            if path.startswith("http"):
                return path
            return SCIHUB_BASE + path

        # pattern 2: iframe src
        m = re.search(r'<iframe[^>]+src="([^"]+\.pdf[^"]*)"', html)
        if m:
            path = m.group(1).strip()
            if path.startswith("//"):
                return "https:" + path
            if path.startswith("/"):
                return SCIHUB_BASE + path
            return path

        # pattern 3: storage path in json blob
        m = re.search(r'"url"\s*:\s*"([^"]+\.pdf)"', html)
        if m:
            path = m.group(1).strip()
            if path.startswith("/"):
                return SCIHUB_BASE + path
            return path

        return None
    except Exception as e:
        print(f"    Sci-Hub error: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────

def try_download(label: str, url: str, dest: Path) -> str | None:
    if not url:
        print(f"    {label}: no PDF found")
        return None
    print(f"    {label}: found PDF, downloading...")
    if download_pdf(url, dest):
        print(f"    {label}: saved {dest.name}")
        return str(dest)
    print(f"    {label}: download failed")
    return None


def fetch_all():
    seen = load_seen()
    print(f"Already in corpus: {len(seen)} papers")
    print(f"Papers to fetch:   {len(ANCHOR_PAPERS)}")
    print(f"Sources:           PubMed -> Unpaywall -> PMC -> Sci-Hub\n")

    stats = {"fetched": 0, "pdf": 0, "abstract_only": 0,
             "no_pubmed": 0, "skipped": 0}

    for i, (title, author, year, subfield, paper_type) in enumerate(ANCHOR_PAPERS):
        key = f"{author.lower()}_{year}"
        print(f"\n[{i+1:02d}/{len(ANCHOR_PAPERS)}] [{paper_type}] {author} {year}")
        print(f"  {title[:65]}")

        if key in seen:
            print(f"  Already fetched - skipping")
            stats["skipped"] += 1
            continue

        # Step 1: PubMed
        print(f"  [1] PubMed...")
        pm = pubmed_fetch(title, author, year)

        if pm is None:
            print(f"  Not found on PubMed - saving stub")
            save_meta({
                "title": title, "first_author": author, "year": str(year),
                "subfield": subfield, "paper_type": paper_type,
                "source": "not_found", "has_full_text": False,
                "doi": None, "pmid": None, "pmcid": None,
                "abstract": "", "pdf_path": None,
            })
            seen.add(key)
            stats["no_pubmed"] += 1
            time.sleep(1)
            continue

        print(f"  Found: {pm['title'][:65]}")
        doi   = pm.get("doi")
        pmcid = pm.get("pmcid")
        print(f"  DOI: {doi or 'none'}  |  PMCID: {pmcid or 'none'}")

        slug     = title_slug(pm["title"])
        pdf_path = None

        # Step 2: Unpaywall
        print(f"  [2] Unpaywall...")
        pdf_path = try_download("Unpaywall", unpaywall_fetch(doi), RAW_DIR / f"{slug}.pdf")

        # Step 3: PMC
        if not pdf_path:
            if pmcid:
                print(f"  [3] PubMed Central...")
                pdf_path = try_download("PMC", pmc_fetch(pmcid), RAW_DIR / f"{slug}_pmc.pdf")
            else:
                print(f"  [3] PMC: no PMCID available")

        # Step 4: Sci-Hub
        if not pdf_path:
            if doi:
                print(f"  [4] Sci-Hub...")
                pdf_path = try_download("Sci-Hub", scihub_fetch(doi), RAW_DIR / f"{slug}_sh.pdf")
            else:
                print(f"  [4] Sci-Hub: no DOI available")

        if not pdf_path:
            print(f"  All sources exhausted - abstract only")

        record = {
            "title":         pm["title"],
            "first_author":  author,
            "authors":       pm["authors"],
            "year":          pm["year"],
            "doi":           doi,
            "pmid":          pm["pmid"],
            "pmcid":         pmcid,
            "abstract":      pm["abstract"],
            "journal":       pm["journal"],
            "subfield":      subfield,
            "paper_type":    paper_type,
            "source":        "pubmed+unpaywall+pmc+scihub",
            "pdf_path":      pdf_path,
            "has_full_text": pdf_path is not None,
        }
        save_meta(record)
        seen.add(key)

        stats["fetched"] += 1
        if pdf_path:
            stats["pdf"] += 1
        else:
            stats["abstract_only"] += 1

        time.sleep(1.0)

    print(f"\n{'─'*55}")
    print(f"Done.")
    print(f"  Fetched:       {stats['fetched']}")
    print(f"  Full PDF:      {stats['pdf']}")
    print(f"  Abstract only: {stats['abstract_only']}")
    print(f"  Not on PubMed: {stats['no_pubmed']}")
    print(f"  Skipped:       {stats['skipped']}")
    print(f"\n  Metadata -> {META_FILE}")
    print(f"  PDFs     -> {RAW_DIR}")
    print(f"\nNext step: python src/ingest.py")


if __name__ == "__main__":
    fetch_all()
