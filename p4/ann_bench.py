# The million-vector test is synthetic, so this script does not call embed().

from __future__ import annotations

import argparse

import faiss
import numpy as np

from common import DIM, random_unit_vectors, recall_at_k, timed_ms


def main():
    p = argparse.ArgumentParser(
        description="ANN bench on synthetic vectors.")
    p.add_argument("--n",
                   type=int,
                   default=1_000_000,
                   help="synthetic vectors (default 1,000,000 ≈ 3.07 GB RAM;"
                        " --n 20000 for a quick demo)")
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    n = args.n
    nq = args.queries
    print(f"synthetic unit vectors: {n} x {DIM}  queries {nq}")
    xb = random_unit_vectors(n + nq, DIM, args.seed)
    xq, db = xb[:nq], xb[nq:]
    nlist = min(1024, max(4, int(len(db)**0.5)))
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    flat = faiss.IndexFlatIP(DIM)
    t0_add = timed_ms(lambda: flat.add(db), repeats=1)
    _, truth = flat.search(xq, 10)
    ms_exact = timed_ms(lambda: flat.search(xq, 10), repeats=3)
    print(
        f"exact scan  add {t0_add:.0f} ms  search {ms_exact:.2f} ms per query  recall@10 1.000"
    )


    hnsw = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    ms_build = timed_ms(lambda: hnsw.add(db), repeats=1)
    hnsw.hnsw.efSearch = 128
    I = hnsw.search(xq, 10)[1]
    ms = timed_ms(lambda: hnsw.search(xq, 10), repeats=3)
    print(
        f"HNSW M=32 efSearch=128  build {ms_build/1000:.1f} s  search {ms:.2f} ms  recall@10 {recall_at_k(I, truth):.3f}"
    )

    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(DIM), DIM, nlist,
                             faiss.METRIC_INNER_PRODUCT)
    ivf.train(db)
    ivf.add(db)
    ivf.nprobe = 16
    I = ivf.search(xq, 10)[1]
    ms = timed_ms(lambda: ivf.search(xq, 10), repeats=3)
    print(
        f"IVFFlat nlist={nlist} nprobe=16  search {ms:.2f} ms  recall@10 {recall_at_k(I, truth):.3f}"
    )


if __name__ == "__main__":
    main()
