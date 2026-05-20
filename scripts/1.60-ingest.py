"""
ingest.py — PDF parsing and chunking pipeline for Cortexa.

Reads PDFs from data/raw/ (and old_folders subfolders), matches each PDF
to metadata from four sources (landmark jsonl, keyword jsonl, perceptual pkl,
sc pkl), parses to markdown via pymupdf4llm, chunks with overlap, and writes
data/processed/chunks.jsonl.

Usage:
    python src/ingest.py
    python src/ingest.py --dry-run        # print stats, don't write
    python src/ingest.py --rejects        # also write rejects.jsonl

Chunk schema (one line per chunk in chunks.jsonl):
    chunk_id      str   "{pdf_stem}_{chunk_index:04d}"
    text          str   chunk text
    title         str   paper title
    first_author  str   first author last name (or "Unknown")
    year          str   publication year (or "")
    subfield      str   e.g. "value_based_dm"
    paper_type    str   "REVIEW" | "LANDMARK" | "BRIDGE" | "KEYWORD" | "UNKNOWN"
    doi           str   DOI or ""
    pmid          str   PubMed ID or ""
    source_file   str   pdf filename (basename)
"""

import argparse
import json
import logging
import pickle
import re
import sys
from pathlib import Path

import pymupdf4llm
from tiktoken import get_encoding

# ---------------------------------------------------------------------------
# Paths — adjust RAW_DIR if your layout differs
# ---------------------------------------------------------------------------
from config import dir_config
ROOT_DIR = Path(dir_config.data.root)

RAW_DIR = ROOT_DIR / "raw"
PROCESSED_DIR = ROOT_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_OUT = PROCESSED_DIR / "chunks.jsonl"
REJECTS_OUT = PROCESSED_DIR / "rejects.jsonl"

# ---------------------------------------------------------------------------
# Chunking config
# ---------------------------------------------------------------------------
CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64
TOKENIZER = get_encoding("cl100k_base")  # same tokenizer used by BGE-M3 internally

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"JSON decode error in {path.name}: {e}")
    return records


def load_pickle(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df.to_dict(orient="records")


def _safe_str(val) -> str:
    """Convert a value to string, treating NaN/NaT/None as empty string."""
    if val is None:
        return ""
    try:
        import math
        if math.isnan(float(str(val).replace("NaT", "nan"))):
            return ""
    except (ValueError, TypeError):
        pass
    s = str(val)
    return "" if s in ("nan", "NaT", "None", "<NA>") else s


def _extract_first_author(record: dict) -> str:
    """Pull first author last name from either schema."""
    # landmark schema has explicit first_author field
    if record.get("first_author"):
        return str(record["first_author"])
    authors = record.get("authors", [])
    if not authors:
        return "Unknown"
    first = authors[0]
    # Semantic Scholar schema: {"authorId": ..., "name": "First Last"}
    if isinstance(first, dict):
        name = first.get("name", "")
    else:
        name = str(first)
    # Return last token as last name (handles "Rangel A" and "Antonio Rangel")
    parts = name.strip().split()
    return parts[-1] if parts else "Unknown"


def normalize_record(record: dict, default_paper_type: str = "UNKNOWN") -> dict:
    """Produce a consistent metadata dict regardless of source schema."""
    return {
        "title": _safe_str(record.get("title", "")),
        "first_author": _extract_first_author(record),
        "year": _safe_str(record.get("year", "")),
        "subfield": _safe_str(record.get("subfield", "unknown")),
        "paper_type": _safe_str(record.get("paper_type", default_paper_type)) or default_paper_type,
        "doi": _safe_str(record.get("doi") or record.get("externalIds.DOI", "")),
        "pmid": _safe_str(record.get("pmid") or record.get("externalIds.PubMed", "")),
        "source_file": "",  # filled in per-PDF below
    }


def build_lookup(records: list[dict], default_paper_type: str = "UNKNOWN") -> dict[str, dict]:
    """
    Build a dict keyed by PDF basename -> normalized metadata.
    Records without a pdf_path are skipped.
    """
    lookup = {}
    for rec in records:
        pdf_path = _safe_str(rec.get("pdf_path", ""))
        if not pdf_path:
            continue
        basename = Path(pdf_path).name
        if not basename.endswith(".pdf"):
            continue
        # If a PDF appears in multiple metadata sources, last one wins.
        # In practice each source covers a different set of files.
        lookup[basename] = normalize_record(rec, default_paper_type)
    return lookup


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """
    Split text into overlapping chunks by token count.
    Uses tiktoken for accurate token counting.
    Returns list of chunk strings.
    """
    tokens = TOKENIZER.encode(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    step = chunk_tokens - overlap_tokens

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_tokens_slice = tokens[start:end]
        chunk_text_decoded = TOKENIZER.decode(chunk_tokens_slice)
        # Strip leading/trailing whitespace but keep internal structure
        chunk_text_decoded = chunk_text_decoded.strip()
        if chunk_text_decoded:
            chunks.append(chunk_text_decoded)
        if end == len(tokens):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path) -> str | None:
    """
    Parse a PDF to markdown using pymupdf4llm.
    Returns markdown string or None on failure.
    """
    try:
        md = pymupdf4llm.to_markdown(str(pdf_path))
        return md
    except Exception as e:
        log.error(f"Failed to parse {pdf_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_master_lookup(raw_dir: Path) -> dict[str, dict]:
    """Load all four metadata sources and merge into one filename -> metadata dict."""
    lookup = {}

    # 1. Landmark papers (jsonl)
    landmark_meta_path = raw_dir / "metadata.jsonl"
    if landmark_meta_path.exists():
        records = load_jsonl(landmark_meta_path)
        lookup.update(build_lookup(records, default_paper_type="LANDMARK"))
        log.info(f"Loaded {len(records)} landmark records")
    else:
        log.warning(f"Not found: {landmark_meta_path}")

    # 2. Keyword papers (jsonl)
    kw_meta_path = raw_dir / "kw_metadata.jsonl"
    if kw_meta_path.exists():
        records = load_jsonl(kw_meta_path)
        lookup.update(build_lookup(records, default_paper_type="KEYWORD"))
        log.info(f"Loaded {len(records)} keyword records")
    else:
        log.warning(f"Not found: {kw_meta_path}")

    # 3. Perceptual DM papers (pickle)
    perceptual_pkl = raw_dir / "old_folders" / "perceptual_decision_making" / "metadata.pkl"
    if perceptual_pkl.exists():
        records = load_pickle(perceptual_pkl)
        lookup.update(build_lookup(records, default_paper_type="UNKNOWN"))
        log.info(f"Loaded {len(records)} perceptual records")
    else:
        log.warning(f"Not found: {perceptual_pkl}")

    # 4. SC papers (pickle)
    sc_pkl = raw_dir / "old_folders" / "superior_colliculus" / "metadata.pkl"
    if sc_pkl.exists():
        records = load_pickle(sc_pkl)
        lookup.update(build_lookup(records, default_paper_type="UNKNOWN"))
        log.info(f"Loaded {len(records)} SC records")
    else:
        log.warning(f"Not found: {sc_pkl}")

    log.info(f"Master lookup: {len(lookup)} unique PDFs with metadata")
    return lookup


def collect_pdfs(raw_dir: Path) -> list[Path]:
    """
    Collect all PDFs across:
      data/raw/*.pdf
      data/raw/old_folders/perceptual_decision_making/pdf/*.pdf
      data/raw/old_folders/superior_colliculus/pdf/*.pdf
    """
    pdfs = []
    pdfs.extend(raw_dir.glob("*.pdf"))
    pdfs.extend((raw_dir / "old_folders" / "perceptual_decision_making" / "pdf").glob("*.pdf"))
    pdfs.extend((raw_dir / "old_folders" / "superior_colliculus" / "pdf").glob("*.pdf"))
    return sorted(set(pdfs))


def run(dry_run: bool = False, write_rejects: bool = False):
    lookup = build_master_lookup(RAW_DIR)
    pdfs = collect_pdfs(RAW_DIR)
    log.info(f"Found {len(pdfs)} PDFs total")

    chunks_written = 0
    rejects = []

    chunk_writer = None
    if not dry_run:
        chunk_writer = open(CHUNKS_OUT, "w")

    reject_writer = None
    if write_rejects and not dry_run:
        reject_writer = open(REJECTS_OUT, "w")

    try:
        for pdf_path in pdfs:
            basename = pdf_path.name

            # --- Metadata lookup ---
            meta = lookup.get(basename)
            if meta is None:
                log.warning(f"No metadata for {basename} -- skipping")
                rejects.append({"pdf": basename, "reason": "no_metadata"})
                if reject_writer:
                    reject_writer.write(json.dumps({"pdf": basename, "reason": "no_metadata"}) + "\n")
                continue

            # --- Parse PDF ---
            log.info(f"Parsing {basename} ({meta['title'][:60]}...)")
            markdown = parse_pdf(pdf_path)
            if not markdown or len(markdown.strip()) < 100:
                log.warning(f"Empty or very short parse for {basename} -- skipping")
                rejects.append({"pdf": basename, "reason": "parse_failed_or_empty"})
                if reject_writer:
                    reject_writer.write(json.dumps({"pdf": basename, "reason": "parse_failed"}) + "\n")
                continue

            # --- Chunk ---
            chunks = chunk_text(markdown)
            if not chunks:
                log.warning(f"No chunks produced for {basename}")
                rejects.append({"pdf": basename, "reason": "no_chunks"})
                continue

            log.info(f"  -> {len(chunks)} chunks")

            if dry_run:
                chunks_written += len(chunks)
                continue

            # --- Write chunks ---
            meta["source_file"] = basename
            pdf_stem = pdf_path.stem

            for i, text in enumerate(chunks):
                record = {
                    "chunk_id": f"{pdf_stem}_{i:04d}",
                    "text": text,
                    **meta,
                }
                chunk_writer.write(json.dumps(record) + "\n")
                chunks_written += 1

    finally:
        if chunk_writer:
            chunk_writer.close()
        if reject_writer:
            reject_writer.close()

    # --- Summary ---
    log.info("=" * 50)
    log.info(f"PDFs processed : {len(pdfs)}")
    log.info(f"Rejects        : {len(rejects)}")
    log.info(f"Chunks written : {chunks_written}")
    if not dry_run:
        log.info(f"Output         : {CHUNKS_OUT}")
    if rejects:
        log.warning("Rejected files:")
        for r in rejects:
            log.warning(f"  {r['pdf']} -- {r['reason']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cortexa ingest pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing output")
    parser.add_argument("--rejects", action="store_true", help="Write rejects.jsonl alongside chunks.jsonl")
    args = parser.parse_args()

    run(dry_run=args.dry_run, write_rejects=args.rejects)
