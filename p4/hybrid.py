# Hybrid search over a toy furniture-store corpus: vector similarity
# (nomic task prefixes, unit-norm vectors so dot product = cosine), BM25
# keyword ranking, and the two fused with reciprocal rank fusion.

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from common import (
    STORE_DOCS,
    STORE_QUERIES,
    embed,
    require_embed_server,
    rrf,
    tokenize,
)


def bm25_ranking(query: str, corpus_tokens: list[list[str]]) -> list[int]:
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokenize(query))
    return list(sorted(range(len(scores)), key=lambda i: -scores[i]))


def argsort_top(D: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(D @ q))[:k]


def fmt_out(rank, ids):
    parts = []
    for j, i in enumerate(rank[:3]):
        mark = "*" if j == 0 else ""
        parts.append(f"{mark}{ids[i]}")
    return ", ".join(parts)


def main():
    require_embed_server()
    ids = [doc_id for doc_id, _ in STORE_DOCS]
    texts = [text for _, text in STORE_DOCS]
    tokens = [tokenize(f"{doc_id} {text}") for doc_id, text in STORE_DOCS]

    # --- cosine neighbors: nomic prefixes, unit norm so dot = cosine ---
    D = embed(["search_document: " + d for d in texts])
    meaning_q = STORE_QUERIES[0][1]  # "my cat scratched the sofa cushions, is that covered?"
    q = embed(["search_query: " + meaning_q])[0]
    top = list(argsort_top(D, q, 3))

    print("first 6 numbers of its vector:", [round(float(x), 3) for x in q[:6]])
    for i in top:
        print(f"  cos={float(D[i] @ q):.3f}  {ids[i]}: {texts[i][:60]}...")
    warranty = next(text for doc_id, text in STORE_DOCS if doc_id == "h-warranty-cushion")
    shared = sorted(set(tokenize(meaning_q)) & set(tokenize(warranty)))
    print(f"  words shared with the warranty chunk: {shared}")

    # --- hybrid: BM25 + vectors fused with rrf ---
    print("\n# hybrid comparison (* = expected doc ranked first)")
    print(f"{'question':<44}  {'vector':<22}  {'keyword':<22}  {'hybrid':<22}")
    lines = []
    for kind, query, expected in STORE_QUERIES:
        qv = embed(["search_query: " + query])[0]
        vec_rank = list(argsort_top(D, qv, len(ids)))
        kw_rank = bm25_ranking(query, tokens)
        hy_rank = rrf(vec_rank, kw_rank, k=60)
        def fmt(rank):
            top3 = [ids[i] for i in rank[:3]]
            star = "*" if rank and ids[rank[0]] == expected else " "
            return star + ",".join(top3)
        print(f"{query:<44}  {fmt(vec_rank):<22}  {fmt(kw_rank):<22}  {fmt(hy_rank):<22}")
        if query.lower().startswith("oak-2291"):
            lines = [
                f"OAK-2291 delivery time",
                f"  vector : {fmt_out(vec_rank, ids)}",
                f"  keyword: {fmt_out(kw_rank, ids)}",
                f"  hybrid : {fmt_out(hy_rank, ids)}",
            ]
    if lines:
        print("\n# per-query breakdown")
        for i, line in enumerate(lines, 1):
            print(f"{i} {line}" if i == 1 else line)


if __name__ == "__main__":
    main()
