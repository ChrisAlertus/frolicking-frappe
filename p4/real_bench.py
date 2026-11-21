# Faiss benchmark over 768-dim embeddings: exact scan vs HNSW vs IVFFlat vs
# IVF-PQ, reporting recall@10 against exact ground truth and ms per query.
# IVF-PQ compresses each vector to 96-byte codes (768 dims must be divisible
# by the 96 subquantizers); the re-rank pass rescores the top 100 with the
# full vectors (IndexRefineFlat, k_factor=10 when k=10).
#
# Default synthesizes 235,476 stand-in vectors; pass --n for a smaller quick
# run, or --vectors word_embeddings.npy from embed_words.py for real ones.

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np

from common import DIM, random_unit_vectors, recall_at_k, timed_ms


def load_or_synth(path: str | None, n: int, seed: int) -> np.ndarray:
    if path:
        xb = np.load(path).astype(np.float32)
        if xb.ndim != 2 or xb.shape[1] != DIM:
            raise SystemExit(f"expected (N, {DIM}) embeddings in {path}, got {xb.shape}")
        return xb
    return random_unit_vectors(n, DIM, seed)


def search_top(index, xq, k, params=None):
    if params is None:
        _, I = index.search(xq, k)
    else:
        _, I = index.search(xq, k, params=params)
    return I


def main():
    p = argparse.ArgumentParser(description="Faiss exact/HNSW/IVF/PQ bench.")
    p.add_argument("--vectors", default=None, help="npy from embed_words.py")
    p.add_argument("--n", type=int, default=235_476, help="synthetic rows if --vectors is omitted")
    p.add_argument("--queries", type=int, default=500, help="held-out queries")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n = args.n
    nq = args.queries
    xb = load_or_synth(args.vectors, n, args.seed)
    n, dim = xb.shape
    nq = min(nq, max(1, n // 20))
    # Hold out the last nq rows as queries.
    xq, db = xb[-nq:], xb[:-nq]
    n_db = db.shape[0]
    nlist = min(1024, max(4, int(n_db ** 0.5)))  # 1024 at full scale
    print(f"database {n_db} x {dim}  queries {nq}  nlist={nlist}")
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    # ground truth: exact inner product
    flat = faiss.IndexFlatIP(dim)
    flat.add(db)
    _, truth = flat.search(xq, 10)

    def report(label, index, params=None, extra=""):
        I = search_top(index, xq, 10, params)
        rec = recall_at_k(I, truth, 10)
        ms = timed_ms(lambda: search_top(index, xq, 10, params), repeats=3)
        print(f"  {label}: recall@10 {rec:.3f}, {ms:.2f} ms per query{extra}")
        return rec, ms

    print("\n# exact, graph, clusters")
    report("exact scan  IndexFlatIP", flat)

    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.add(db)
    for ef in (16, 128, 256):
        hnsw.hnsw.efSearch = ef
        report(f"HNSW M=32 efSearch={ef}", hnsw)
    hnsw.hnsw.efSearch = 128

    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(db)
    ivf.add(db)
    for nprobe in (1, 16):
        ivf.nprobe = nprobe
        report(f"IVFFlat nprobe={nprobe}", ivf)

    # IVF-PQ section
    print("\n# IVF-PQ section")
    m = 96  # 96 bytes/vector when nbits=8; 768 must be divisible by m
    pq = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    pq.train(db)
    pq.add(db)
    pq.nprobe = 16
    raw_gb = n_db * dim * 4 / 1e9
    codes_gb = n_db * m / 1e9
    ratio = raw_gb / codes_gb if codes_gb else float("inf")
    print(f"[IVF-PQ {m} bytes per vector]  codes {codes_gb:.3f} GB vs {raw_gb:.2f} GB raw ({ratio:.0f}x smaller)")
    report("nprobe 16", pq)

    # re-rank top 100 with the full vectors (k_factor=10, k=10)
    try:
        rescored = faiss.IndexRefineFlat(pq, faiss.swig_ptr(db))
    except TypeError:
        # fallback: some Faiss builds only accept IndexRefineFlat(base)
        rescored = faiss.IndexRefineFlat(pq)
    rescored.k_factor = 10
    report("nprobe 16 + re-rank top 100 with the full vectors", rescored)


if __name__ == "__main__":
    main()
