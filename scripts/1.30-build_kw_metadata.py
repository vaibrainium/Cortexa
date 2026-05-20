"""
build_kw_list.py — build keyword-based paper list per subfield

Searches Semantic Scholar for each query, ranks results by citation count,
deduplicates against existing corpus, and saves top N papers to kw_metadata.jsonl
for review before downloading.

Usage:
    python src/build_kw_list.py

Output:
    /mnt/data/kw_metadata.jsonl   — candidate papers, sorted by citations
    /mnt/data/kw_summary.txt      — human-readable summary per subfield

Review kw_metadata.jsonl before running fetch_kw_papers.py.
Remove any irrelevant entries manually if needed.
"""

import json
import time
import requests
from pathlib import Path
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────

META_FILE    = Path("/mnt/data/raw/metadata.jsonl")        # existing anchor corpus
KW_META_FILE = Path("/mnt/data/raw/kw_metadata.jsonl")     # output
KW_SUMMARY   = Path("/mnt/data/raw/kw_summary.txt")

S2_SEARCH  = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS  = "title,authors,year,externalIds,abstract,citationCount,openAccessPdf,venue"

MAX_PER_QUERY   = 50   # results to fetch per query
TOP_N_PER_SUBFIELD = 100  # max papers to keep per subfield after ranking
MIN_CITATIONS   = 10   # filter out very low-impact papers
MIN_YEAR        = 1990 # filter out very old papers with limited relevance

REQUEST_DELAY = 2.5    # seconds between requests (unauthenticated limit)

# ── Keyword queries per subfield ───────────────────────────────────────────
# Multiple queries per subfield — results are merged and deduplicated

KEYWORD_QUERIES = {
    "value_based_dm": [
        "orbitofrontal cortex value decision making",
        "reward valuation neural basis",
        "economic choice neural computation",
        "willingness to pay brain",
        "utility maximization prefrontal",
    ],
    "perceptual_dm": [
        "perceptual decision making evidence accumulation",
        "random dot motion neural decision",
        "lateral intraparietal area decision",
        "drift diffusion model neural",
        "signal detection theory brain",
    ],
    "rl_brain": [
        "dopamine reward prediction error striatum",
        "model-based model-free reinforcement learning brain",
        "temporal difference learning neural",
        "basal ganglia reinforcement learning",
        "exploration exploitation neural decision",
    ],
    "drift_diffusion": [
        "drift diffusion model decision making",
        "evidence accumulation threshold neural",
        "sequential sampling decision model",
        "reaction time decision model",
        "accumulator model perceptual decision",
    ],
    "neuroeconomics": [
        "neuroeconomics decision brain",
        "prospect theory neural basis",
        "loss aversion brain imaging",
        "intertemporal choice neural",
        "economic utility brain",
    ],
    "cognitive_control": [
        "prefrontal cortex cognitive control",
        "anterior cingulate conflict monitoring",
        "inhibitory control neural basis",
        "working memory decision making",
        "executive function prefrontal",
    ],
    "confidence_metacog": [
        "decision confidence neural correlates",
        "metacognition brain prefrontal",
        "subjective confidence perceptual",
        "uncertainty estimation neural",
        "post-decisional processing confidence",
    ],
    "sc_subcortical": [
        "superior colliculus decision making",
        "superior colliculus evidence accumulation",
        "superior colliculus perceptual choice primate",
        "midbrain decision making subcortical",
        "superior colliculus attention selection",
    ],
}

# ── Helpers ────────────────────────────────────────────────────────────────

def load_existing_dois() -> set:
    """Load DOIs already in the anchor corpus to avoid duplicates."""
    seen = set()
    if META_FILE.exists():
        for line in META_FILE.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("doi"):
                    seen.add(r["doi"].lower())
            except Exception:
                pass
    return seen


def load_existing_kw_dois() -> set:
    """Load DOIs already in kw_metadata to avoid re-adding."""
    seen = set()
    if KW_META_FILE.exists():
        for line in KW_META_FILE.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("doi"):
                    seen.add(r["doi"].lower())
            except Exception:
                pass
    return seen


def s2_search(query: str, offset: int = 0) -> list[dict]:
    """Search Semantic Scholar, return list of paper dicts."""
    try:
        r = requests.get(
            S2_SEARCH,
            params={
                "query":  query,
                "limit":  MAX_PER_QUERY,
                "offset": offset,
                "fields": S2_FIELDS,
            },
            timeout=15,
        )
        if r.status_code == 429:
            print(f"    Rate limited, waiting 30s...")
            time.sleep(30)
            return s2_search(query, offset)  # retry once
        if r.status_code != 200:
            print(f"    S2 error {r.status_code}")
            return []
        return r.json().get("data", [])
    except Exception as e:
        print(f"    S2 error: {e}")
        return []


def parse_paper(paper: dict, subfield: str, query: str) -> dict | None:
    """Extract relevant fields from S2 paper dict."""
    title = paper.get("title", "")
    if not title:
        return None

    year = paper.get("year")
    if year and year < MIN_YEAR:
        return None

    citations = paper.get("citationCount", 0) or 0
    if citations < MIN_CITATIONS:
        return None

    doi = paper.get("externalIds", {}).get("DOI", "")
    arxiv_id = paper.get("externalIds", {}).get("ArXiv", "")

    oa = paper.get("openAccessPdf")
    oa_url = oa.get("url") if oa else None

    return {
        "title":       title,
        "authors":     [a["name"] for a in paper.get("authors", [])[:5]],
        "year":        year,
        "doi":         doi,
        "arxiv_id":    arxiv_id,
        "venue":       paper.get("venue", ""),
        "citations":   citations,
        "abstract":    paper.get("abstract", ""),
        "oa_url":      oa_url,
        "subfield":    subfield,
        "query":       query,
        "source":      "keyword_search",
        "has_full_text": False,  # will be set by fetch script
        "pdf_path":    None,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def build_list():
    existing_dois = load_existing_dois()
    kw_dois       = load_existing_kw_dois()
    all_seen_dois = existing_dois | kw_dois

    print(f"Existing anchor corpus: {len(existing_dois)} papers")
    print(f"Already in kw list:     {len(kw_dois)} papers")
    print(f"Min citations filter:   {MIN_CITATIONS}")
    print(f"Min year filter:        {MIN_YEAR}")
    print(f"Max per subfield:       {TOP_N_PER_SUBFIELD}")
    print()

    subfield_counts = defaultdict(int)
    total_added = 0

    # open in append mode so we can resume if interrupted
    with open(KW_META_FILE, "a") as f_out:
        for subfield, queries in KEYWORD_QUERIES.items():
            print(f"\n{'─'*50}")
            print(f"Subfield: {subfield}")

            subfield_papers = {}  # doi -> paper dict, for dedup within subfield

            for query in queries:
                print(f"  Query: {query}")
                results = s2_search(query)
                print(f"  Got {len(results)} results")

                added = 0
                for paper in results:
                    parsed = parse_paper(paper, subfield, query)
                    if parsed is None:
                        continue

                    doi = parsed["doi"].lower() if parsed["doi"] else ""
                    arxiv = parsed["arxiv_id"].lower() if parsed["arxiv_id"] else ""

                    # skip if already in anchor corpus or kw list
                    if doi and doi in all_seen_dois:
                        continue
                    if arxiv and arxiv in all_seen_dois:
                        continue

                    # deduplicate within this subfield by doi or title
                    dedup_key = doi or parsed["title"].lower()[:60]
                    if dedup_key in subfield_papers:
                        continue

                    subfield_papers[dedup_key] = parsed
                    added += 1

                print(f"  New unique papers: {added}")
                time.sleep(REQUEST_DELAY)

            # sort by citation count, take top N
            ranked = sorted(
                subfield_papers.values(),
                key=lambda x: x["citations"],
                reverse=True,
            )[:TOP_N_PER_SUBFIELD]

            # write to file
            for paper in ranked:
                f_out.write(json.dumps(paper) + "\n")
                doi = paper["doi"].lower() if paper["doi"] else ""
                if doi:
                    all_seen_dois.add(doi)
                subfield_counts[subfield] += 1
                total_added += 1

            print(f"  Kept (top by citations): {len(ranked)}")
            if ranked:
                print(f"  Citation range: {ranked[-1]['citations']} - {ranked[0]['citations']}")

    # write summary
    with open(KW_SUMMARY, "w") as f:
        f.write("Cortexa keyword paper list summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total papers: {total_added}\n\n")
        for sf, count in subfield_counts.items():
            f.write(f"  {sf}: {count}\n")
        f.write(f"\nFilters applied:\n")
        f.write(f"  Min citations: {MIN_CITATIONS}\n")
        f.write(f"  Min year: {MIN_YEAR}\n")
        f.write(f"  Max per subfield: {TOP_N_PER_SUBFIELD}\n")
        f.write(f"\nReview kw_metadata.jsonl before running fetch_kw_papers.py\n")

    print(f"\n{'='*50}")
    print(f"Done. Total papers: {total_added}")
    for sf, count in subfield_counts.items():
        print(f"  {sf}: {count}")
    print(f"\nMetadata -> {KW_META_FILE}")
    print(f"Summary  -> {KW_SUMMARY}")
    print(f"\nReview the list, then run: python src/fetch_kw_papers.py")


if __name__ == "__main__":
    build_list()
