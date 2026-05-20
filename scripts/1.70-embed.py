"""
embed.py — BGE-M3 encoding and Qdrant upsert pipeline for Cortexa.

Reads data/processed/chunks.jsonl, encodes each chunk with BGE-M3 (dense
vectors only for MVP), and upserts into a Qdrant collection with full
metadata payload.

Usage:
    python src/embed.py                    # full run
    python src/embed.py --dry-run          # encode first batch only, no upsert
    python src/embed.py --batch-size 64    # override batch size

Collection schema:
    Vector : BGE-M3 dense, 1024 dims, cosine similarity
    Payload fields stored per point:
        chunk_id      str
        text          str
        title         str
        first_author  str
        year          str
        subfield      str
        paper_type    str
        doi           str
        pmid          str
        source_file   str

Post-MVP note:
    Qdrant natively supports sparse vectors alongside dense in the same
    collection. When adding BM25 in S9, add a second named vector field
    ("sparse") to the same collection -- no migration needed.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Paths — adjust RAW_DIR if your layout differs
# ---------------------------------------------------------------------------
from config import dir_config
ROOT_DIR = Path(dir_config.data.root)
PROCESSED_DIR = ROOT_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "cortexa"
VECTOR_DIM = 1024          # BGE-M3 dense output dimension
BATCH_SIZE = 32            # safe default for a single GPU; increase if VRAM allows
BGE_MODEL_NAME = "BAAI/bge-m3"

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
# Helpers
# ---------------------------------------------------------------------------

def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning(f"Skipping malformed line: {e}")
    return chunks


def setup_collection(client: QdrantClient, collection_name: str, vector_dim: int):
    """
    Create the Qdrant collection if it doesn't exist.
    If it already exists, log its current point count and continue --
    this makes embed.py safely re-runnable (e.g. after adding new papers).
    """
    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        info = client.get_collection(collection_name)
        count = info.points_count
        log.info(f"Collection '{collection_name}' already exists ({count} points). Upserting into it.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
        ),
    )
    log.info(f"Created collection '{collection_name}' (dim={vector_dim}, cosine)")


def build_points(batch_chunks: list[dict], embeddings: list[list[float]]) -> list[PointStruct]:
    """Package chunk metadata + vector into Qdrant PointStructs."""
    points = []
    for chunk, vector in zip(batch_chunks, embeddings):
        # Use a stable integer ID derived from chunk_id string for Qdrant.
        # Qdrant requires integer or UUID point IDs.
        point_id = abs(hash(chunk["chunk_id"])) % (2**63)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "title": chunk.get("title", ""),
                    "first_author": chunk.get("first_author", ""),
                    "year": chunk.get("year", ""),
                    "subfield": chunk.get("subfield", ""),
                    "paper_type": chunk.get("paper_type", ""),
                    "doi": chunk.get("doi", ""),
                    "pmid": chunk.get("pmid", ""),
                    "source_file": chunk.get("source_file", ""),
                },
            )
        )
    return points


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(batch_size: int = BATCH_SIZE, dry_run: bool = False):
    # --- Load chunks ---
    log.info(f"Loading chunks from {CHUNKS_PATH}")
    chunks = load_chunks(CHUNKS_PATH)
    log.info(f"Loaded {len(chunks)} chunks")

    if not chunks:
        log.error("No chunks found. Run ingest.py first.")
        sys.exit(1)

    # --- Load BGE-M3 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading BGE-M3 on {device} ...")
    model = SentenceTransformer(BGE_MODEL_NAME, device=device)
    if device == "cuda":
        model = model.half()   # fp16 to match original VRAM behaviour
    log.info("BGE-M3 loaded")

    # --- Qdrant client + collection ---
    if not dry_run:
        client = QdrantClient(url=QDRANT_URL)
        setup_collection(client, COLLECTION_NAME, VECTOR_DIM)

    # --- Batch encode + upsert ---
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    points_upserted = 0
    t0 = time.time()

    for batch_idx in tqdm(range(total_batches), desc="Embedding batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        texts = [c["text"] for c in batch]

        # MVP: dense only via sentence-transformers
        # S9: swap back to FlagEmbedding BGEM3FlagModel for sparse support
        dense_vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        if dry_run:
            log.info(f"Dry run: encoded batch {batch_idx + 1}/{total_batches}, "
                     f"vector dim={len(dense_vecs[0])}")
            if batch_idx == 0:
                log.info("Dry run complete -- first batch encoded successfully. Exiting.")
                return

        points = build_points(batch, dense_vecs)
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        points_upserted += len(points)

    elapsed = time.time() - t0
    log.info("=" * 50)
    log.info(f"Chunks embedded  : {len(chunks)}")
    log.info(f"Points upserted  : {points_upserted}")
    log.info(f"Time elapsed     : {elapsed:.1f}s  ({elapsed/len(chunks)*1000:.1f}ms/chunk)")
    log.info(f"Collection       : {COLLECTION_NAME} @ {QDRANT_URL}")


# ---------------------------------------------------------------------------
# Sanity query -- run after embedding to verify retrieval works
# ---------------------------------------------------------------------------

def sanity_check():
    """
    Run a few test queries against the collection and print top-5 results.
    Call this manually after embed.py completes:
        python src/embed.py --sanity
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(BGE_MODEL_NAME, device=device)
    if device == "cuda":
        model = model.half()
    client = QdrantClient(url=QDRANT_URL)

    test_queries = [
        "How does the Superior Colliculus encode Bayesian priors?",
        "drift diffusion model evidence accumulation threshold",
        "dopamine reward prediction error striatum",
        "Basso Wurtz buildup neurons probability",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print("=" * 60)

        vec = model.encode([query], normalize_embeddings=True)[0].tolist()

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=5,
            with_payload=True,
        ).points

        for i, hit in enumerate(results):
            p = hit.payload
            print(f"\n  [{i+1}] score={hit.score:.4f}")
            print(f"       {p['first_author']} ({p['year']}) -- {p['title'][:70]}")
            print(f"       subfield={p['subfield']}  type={p['paper_type']}")
            print(f"       chunk: {p['text'][:120].strip()}...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cortexa embed pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Encode first batch only, skip Qdrant upsert")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Embedding batch size (default: {BATCH_SIZE})")
    parser.add_argument("--sanity", action="store_true",
                        help="Run sanity queries against existing collection and exit")
    args = parser.parse_args()

    if args.sanity:
        sanity_check()
    else:
        run(batch_size=args.batch_size, dry_run=args.dry_run)
