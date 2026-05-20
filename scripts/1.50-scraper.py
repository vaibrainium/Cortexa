"""
scraper.py — Cortexa corpus builder

Mode 1 (run today): fetch anchor papers by title from Semantic Scholar
Mode 2 (later):     keyword search per subfield — uncomment KEYWORD_QUERIES section

Usage (inside container):
    python src/scraper.py

Outputs:
    /mnt/data/raw/          PDFs where open access
    /mnt/data/metadata.jsonl  one record per paper (title, authors, year, doi,
                              subfield, paper_type, abstract, pdf_path)
"""

import os
import json
import time
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────

RAW_DIR   = Path("/mnt/data/raw")
META_FILE = Path("/mnt/data/raw/metadata.jsonl")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Semantic Scholar config ────────────────────────────────────────────────

S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
S2_HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
S2_SEARCH  = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS  = "title,authors,year,externalIds,abstract,openAccessPdf,publicationTypes"

# ── Anchor papers corpus map ───────────────────────────────────────────────
# Format: (first_author, short_title, subfield, paper_type)
# paper_type: REVIEW | LANDMARK | BRIDGE

ANCHOR_PAPERS = [
    # 1. Value-based decision making
    ("Rangel",        "framework neurobiology value-based decision making",         "value_based_dm",        "REVIEW"),
    ("Padoa-Schioppa","neurobiology economic choice good-based model",              "value_based_dm",        "REVIEW"),
    ("Plassmann",     "orbitofrontal cortex encodes willingness to pay",            "value_based_dm",        "LANDMARK"),
    ("Padoa-Schioppa","neurons orbitofrontal cortex encode economic value",         "value_based_dm",        "LANDMARK"),
    ("Rustichini",    "neuro-computational model economic decisions",               "value_based_dm",        "BRIDGE"),

    # 2. Perceptual decision making
    ("Gold",          "neural basis decision making",                               "perceptual_dm",         "REVIEW"),
    ("Shadlen",       "neural basis perceptual decision parietal cortex",           "perceptual_dm",         "LANDMARK"),
    ("Britten",       "analysis visual motion neuronal psychophysical performance", "perceptual_dm",         "LANDMARK"),
    ("Summerfield",   "do humans make good decisions",                              "perceptual_dm",         "BRIDGE"),
    ("Drugowitsch",   "cost accumulating evidence perceptual decision making",      "perceptual_dm",         "BRIDGE"),

    # 3. Reinforcement learning in the brain
    ("Dayan",         "model-based model-free pavlovian reward learning",           "rl_brain",              "REVIEW"),
    ("Niv",           "reinforcement learning in the brain",                        "rl_brain",              "REVIEW"),
    ("Schultz",       "neural substrate prediction and reward",                     "rl_brain",              "LANDMARK"),
    ("Daw",           "cortical substrates exploratory decisions humans",           "rl_brain",              "LANDMARK"),
    ("Doll",          "ubiquity model-based reinforcement learning",                "rl_brain",              "BRIDGE"),

    # 4. Drift-diffusion / computational models
    ("Ratcliff",      "diffusion decision model theory data",                       "drift_diffusion",       "REVIEW"),
    ("Bogacz",        "physics optimal decision making",                            "drift_diffusion",       "REVIEW"),
    ("Krajbich",      "visual fixations computation value simple choice",           "drift_diffusion",       "BRIDGE"),
    ("Usher",         "time course perceptual choice leaky competing accumulator",  "drift_diffusion",       "LANDMARK"),
    ("Wiecki",        "HDDM hierarchical bayesian estimation diffusion decision",   "drift_diffusion",       "BRIDGE"),

    # 5. Neuroeconomics
    ("Camerer",       "neuroeconomics how neuroscience can inform economics",       "neuroeconomics",        "REVIEW"),
    ("Kahneman",      "prospect theory analysis decision under risk",               "neuroeconomics",        "LANDMARK"),
    ("Rangel",        "value normalization in decision making",                     "neuroeconomics",        "BRIDGE"),
    ("Fehr",          "social neuroeconomics neural circuitry social preferences",  "neuroeconomics",        "BRIDGE"),

    # 6. Cognitive control
    ("Badre",         "frontal cortex hierarchical control behavior",               "cognitive_control",     "REVIEW"),
    ("Shenhav",       "expected value of control integrative theory anterior cingulate", "cognitive_control","REVIEW"),
    ("Miller",        "integrative theory prefrontal cortex function",              "cognitive_control",     "LANDMARK"),
    ("Botvinick",     "motivation cognitive control behavior neural mechanism",     "cognitive_control",     "BRIDGE"),
    ("Daw",           "uncertainty-based competition prefrontal striatal systems",  "cognitive_control",     "BRIDGE"),

    # 7. Confidence and metacognition
    ("Fleming",       "neural basis metacognitive ability",                         "confidence_metacog",    "REVIEW"),
    ("Yeung",         "metacognition human decision-making",                        "confidence_metacog",    "REVIEW"),
    ("Kepecs",        "neural correlates computation decision confidence",          "confidence_metacog",    "LANDMARK"),
    ("Pouget",        "confidence and certainty distinct probabilistic quantities", "confidence_metacog",    "BRIDGE"),
    ("Donoso",        "foundations human reasoning prefrontal cortex",              "confidence_metacog",    "BRIDGE"),

    # 8. SC and subcortical decision making
    ("Wurtz",         "visual-motor function primate superior colliculus",          "sc_subcortical",        "REVIEW"),
    ("Felsen",        "neural substrates sensory-guided locomotor decisions superior colliculus", "sc_subcortical", "REVIEW"),
    ("Basso",         "modulation neuronal activity superior colliculus target probability",      "sc_subcortical", "LANDMARK"),
    ("Jun",           "causal role primate superior colliculus computation evidence perceptual",  "sc_subcortical", "LANDMARK"),
    ("Crapse",        "role superior colliculus decision criteria",                 "sc_subcortical",        "LANDMARK"),
    ("Krauzlis",      "superior colliculus visual spatial attention",               "sc_subcortical",        "BRIDGE"),
    ("Felsen",        "integrative role superior colliculus selecting targets movements",         "sc_subcortical", "BRIDGE"),
    ("Duan",          "primate superior colliculus causally engaged abstract higher-order cognition", "sc_subcortical", "BRIDGE"),
    ("McAlonan",      "superior colliculus inactivation biases target choice",      "sc_subcortical",        "BRIDGE"),
    ("Kim",           "superior colliculus encodes alternations spatial target selections",       "sc_subcortical", "BRIDGE"),
]

# ── Keyword search queries (Mode 2 — uncomment to run) ────────────────────
# Run after anchor papers are fetched. Broadens corpus per subfield.
#
KEYWORD_QUERIES = {
    "value_based_dm":    ["orbitofrontal cortex value decision", "reward valuation brain fMRI"],
    "perceptual_dm":     ["evidence accumulation neural decision", "random dot motion LIP"],
    "rl_brain":          ["dopamine reward prediction error striatum", "model-based striatum prefrontal"],
    "drift_diffusion":   ["drift diffusion model reaction time", "sequential sampling neural decision"],
    "neuroeconomics":    ["utility neural economic choice", "value normalization cortex"],
    "cognitive_control": ["prefrontal cognitive control decision", "anterior cingulate conflict"],
    "confidence_metacog":["decision confidence neural", "metacognition prefrontal"],
    "sc_subcortical":    ["superior colliculus decision making primate", "superior colliculus evidence accumulation"],
}
MAX_PER_QUERY = 25


# ── Helpers ────────────────────────────────────────────────────────────────

def load_seen_titles() -> set:
    seen = set()
    if META_FILE.exists():
        for line in META_FILE.read_text().splitlines():
            try:
                seen.add(json.loads(line)["title"].lower()[:60])
            except Exception:
                pass
    return seen


def save_meta(record: dict):
    with open(META_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def title_slug(title: str) -> str:
    return hashlib.md5(title.lower().encode()).hexdigest()[:12]


def download_pdf(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Cortexa/0.1"})
        if r.status_code == 200 and b"%PDF" in r.content[:8]:
            dest.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"    PDF download error: {e}")
    return False


# ── Semantic Scholar fetch ─────────────────────────────────────────────────

def fetch_paper(first_author: str, title_query: str, subfield: str, paper_type: str, seen: set) -> dict | None:
    query = f"{first_author} {title_query}"
    try:
        r = requests.get(
            S2_SEARCH,
            headers=S2_HEADERS,
            params={"query": query, "limit": 3, "fields": S2_FIELDS},
            timeout=15,
        )
        if r.status_code != 200:
            print(f"  S2 error {r.status_code} for: {query[:60]}")
            return None

        results = r.json().get("data", [])
        if not results:
            print(f"  No results: {query[:60]}")
            return None

        paper = results[0]
        title = paper.get("title", "")

        # skip if already in corpus
        if title.lower()[:60] in seen:
            print(f"  Already have: {title[:60]}")
            return None

        seen.add(title.lower()[:60])

        doi = paper.get("externalIds", {}).get("DOI", "")
        arxiv_id = paper.get("externalIds", {}).get("ArXiv", "")

        # try to download PDF
        pdf_path = None
        oa = paper.get("openAccessPdf")
        if oa and oa.get("url"):
            fname = title_slug(title) + ".pdf"
            fpath = RAW_DIR / fname
            if not fpath.exists():
                print(f"    Downloading PDF...")
                if download_pdf(oa["url"], fpath):
                    pdf_path = str(fpath)
                    print(f"    Saved: {fname}")
                else:
                    print(f"    PDF unavailable — abstract only")
            else:
                pdf_path = str(fpath)

        record = {
            "title":      title,
            "authors":    [a["name"] for a in paper.get("authors", [])],
            "year":       paper.get("year"),
            "doi":        doi,
            "arxiv_id":   arxiv_id,
            "abstract":   paper.get("abstract", ""),
            "subfield":   subfield,
            "paper_type": paper_type,
            "source":     "semantic_scholar",
            "pdf_path":   pdf_path,
            "has_full_text": pdf_path is not None,
        }
        save_meta(record)
        return record

    except Exception as e:
        print(f"  Error fetching {query[:60]}: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    seen = load_seen_titles()
    print(f"Starting anchor paper fetch.")
    print(f"Already in corpus: {len(seen)} papers")
    print(f"Papers to fetch: {len(ANCHOR_PAPERS)}\n")

    results = {"fetched": 0, "pdf": 0, "abstract_only": 0, "failed": 0}

    for first_author, title_query, subfield, paper_type in tqdm(ANCHOR_PAPERS):
        print(f"\n[{paper_type}] {first_author} — {title_query[:50]}")
        record = fetch_paper(first_author, title_query, subfield, paper_type, seen)

        if record is None:
            results["failed"] += 1
        else:
            results["fetched"] += 1
            if record["has_full_text"]:
                results["pdf"] += 1
            else:
                results["abstract_only"] += 1

        time.sleep(1.2)  # S2 rate limit: ~1 req/sec unauthenticated

    print(f"\n{'─'*50}")
    print(f"Done.")
    print(f"  Fetched:        {results['fetched']}")
    print(f"  Full PDF:       {results['pdf']}")
    print(f"  Abstract only:  {results['abstract_only']}")
    print(f"  Failed/skipped: {results['failed']}")
    print(f"  Metadata:       {META_FILE}")
    print(f"  PDFs:           {RAW_DIR}")
    print(f"\nNext: run python src/ingest.py")


if __name__ == "__main__":
    run()
