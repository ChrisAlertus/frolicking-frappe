# HNSW freshness and filtering on synthetic 768-dim vectors: adding new
# items after the build, tombstone deletes (ntotal never shrinks), and a
# rare ~0.8% "in-stock sofas" filter — top-k-then-filter vs filtering
# inside the graph walk (IDSelectorBitmap) vs an exact scan of the matches.
#
# Defaults are the full sizes (200,000 products + 1,000 adds + 1,000
# tombstones); pass --n/--add/--delete for a quick run.

from __future__ import annotations

import argparse
import time

import faiss
import numpy as np

from common import DIM, random_unit_vectors, timed_ms


def pack_selector(ok: np.ndarray):
    """Bitmap selector over the boolean mask, with a batch-ids fallback."""
    n = ok.shape[0]
    bits = np.packbits(ok.astype(np.uint8), bitorder="little")
    try:
        return faiss.IDSelectorBitmap(n, faiss.swig_ptr(bits))
    except Exception:
        # fallback for Faiss builds without IDSelectorBitmap
        ids = np.where(ok)[0].astype("int64")
        return faiss.IDSelectorBatch(ids.size, faiss.swig_ptr(ids))


def main():
    p = argparse.ArgumentParser(description="Fresh items, tombstones, filters.")
    p.add_argument("--n", type=int, default=200_000, help="base products")
    p.add_argument("--add", type=int, default=1_000, help="new items")
    p.add_argument("--delete", type=int, default=1_000, help="tombstones")
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    n = args.n
    n_add = args.add
    n_del = args.delete
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    xb = random_unit_vectors(n + n_add, DIM, args.seed)
    base, extra = xb[:n], xb[n:]

    hnsw = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    hnsw.add(base)
    build_s = time.perf_counter() - t0
    print(f"built HNSW over {n:,} products in {build_s:.0f} s")

    # new items: add, then search with the new vector itself
    t0 = time.perf_counter()
    hnsw.add(extra)
    add_s = time.perf_counter() - t0
    first = 0
    for i in range(n_add):
        q = extra[i : i + 1]
        _, I = hnsw.search(q, 1)
        if int(I[0, 0]) == n + i:
            first += 1
    print(
        f"[new items] added {n_add:,} new products in {add_s*1000:.0f} ms "
        f"({add_s/n_add*1000:.2f} ms each); searching for each one right after: "
        f"{first}/{n_add} come back first"
    )

    # deletes as tombstones: ntotal stays the same
    ntotal_before = hnsw.ntotal
    tombstone = np.zeros(hnsw.ntotal, dtype=bool)
    # delete a slice of the original products, not the brand-new ones
    tombstone[n - n_del : n] = True
    alive = ~tombstone
    q = random_unit_vectors(1, DIM, args.seed + 99)
    _, I = hnsw.search(q, min(10_000, hnsw.ntotal))
    returned_deleted = int(tombstone[I[0]].sum())
    print(
        f"[deletes] marked {n_del:,} products deleted; ... returns {returned_deleted} deleted ids "
        f"in {I.shape[1]:,} results;"
    )
    print(
        f"        element count before {ntotal_before:,}, after {hnsw.ntotal:,} "
        f"(a delete is a tombstone)"
    )
    if returned_deleted:
        print("        (note: raw HNSW may still return tombstoned ids; filter them in application code)")

    # rare filter: ~0.8% "in-stock sofas" — walk-with-selector vs exact scan
    n_all = hnsw.ntotal
    rng = np.random.default_rng(args.seed + 7)
    ok = np.zeros(n_all, dtype=bool)
    n_ok = max(1, int(round(0.008 * n_all)))
    ok[rng.choice(n_all, size=n_ok, replace=False)] = True
    ok &= alive
    n_ok = int(ok.sum())
    print(f"\n# Filters: \"in-stock sofas only\"  {n_ok:,} of {n_all:,} = {100*n_ok/n_all:.1f}%")

    # pick a query whose true neighbors are mostly in the filter set
    matches = np.where(ok)[0]
    q = xb[matches[0] : matches[0] + 1] if len(matches) else q
    xb_all = np.vstack([base, extra])

    def filter_after(k):
        _, I = hnsw.search(q, k)
        return [int(i) for i in I[0] if i >= 0 and ok[int(i)]][:10]

    for k, label in ((10, "top 10, then filter"), (100, "top 100, then filter")):
        kept = filter_after(k)
        # "results of 10" is how many of the 10 slots were filled
        print(f"  {label}: results of 10 = {len(kept)}")

    # filter inside the walk vs exact scan of the few matches
    sel = pack_selector(ok)
    try:
        params = faiss.SearchParametersHNSW(sel=sel, efSearch=200)
        D_, I_ = hnsw.search(q, 10, params=params)
        walk_ids = [int(i) for i in I_[0] if i >= 0]
        print(f"  filter inside the walk (IDSelectorBitmap, efSearch=200): {walk_ids}")
    except Exception as exc:
        print(f"  filter inside the walk: Faiss selector API failed ({exc!r}); skipped")

    best = matches[np.argsort(-(xb_all[matches] @ q[0]))[:10]] if len(matches) else np.array([], dtype=int)
    print(f"  exact scan of the {len(matches)} matches: {best.tolist()[:10]}")


if __name__ == "__main__":
    main()
