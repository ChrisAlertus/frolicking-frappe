# Shared helpers for the vector-search demos.

from __future__ import annotations

import os
import time

import numpy as np
import requests

DIM = 768
EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8395/v1")


def embed(texts):
    r = requests.post(EMBED_URL + "/embeddings",
                      json={"input": texts},
                      timeout=300)
    r.raise_for_status()
    v = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    return v / np.linalg.norm(
        v, axis=1, keepdims=True)  # unit length: dot product = cosine


def rrf(*lists, k=60):
    s = {}
    for lst in lists:
        for rank, i in enumerate(lst, 1):
            s[i] = s.get(i, 0) + 1 / (k + rank)
    return sorted(s, key=lambda i: -s[i])


def require_embed_server():
    """Fail with a setup hint if llama-server is not reachable."""
    try:
        embed(["search_query: ping"])
    except Exception as exc:
        raise SystemExit("Could not reach the embedding server at "
                         f"{EMBED_URL}. Start it first:\n"
                         "  ./p4/setup.sh    # or: llama-server -hf "
                         "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0 "
                         "--embedding --pooling mean --port 8395 &\n"
                         f"Underlying error: {exc}") from exc


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def random_unit_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Synthetic stand-in for stored embeddings."""
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal((n, dim), dtype=np.float32))


def tokenize(text: str) -> list[str]:
    return [
        t for t in "".join(ch.lower() if ch.isalnum() else " "
                           for ch in text).split() if t
    ]


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    """Share of true top-k IDs recovered in each predicted row, then averaged."""
    hits = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k].tolist()) & set(t[:k].tolist()))
    return hits / (len(pred) * k)


def timed_ms(fn, repeats: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1000 / repeats


# Toy furniture-store corpus for the hybrid-search demo.
STORE_DOCS = [
    ("h-warranty-cushion",
     "Cushions and fabric carry a 1-year warranty. Pet scratches on sofa cushions may be covered under the fabric warranty if you file a claim."
     ),
    ("p-sof-1180",
     "SOF-1180 Juno three-seat sofa, performance velvet, soft cushions, living room seating."
     ),
    ("p-out-1502",
     "OUT-1502 Sol outdoor lounge chair, teak frame, weather-resistant cushions for the patio."
     ),
    ("p-oak-2290",
     "OAK-2290 solid oak dining bench, natural finish, seats two at the table."
     ),
    ("p-oak-2291",
     "OAK-2291 solid oak dining table, seats six. Delivery time is 2 to 4 weeks; white-glove delivery is available."
     ),
    ("h-ship-large",
     "Shipping large furniture: delivery time windows, curb-side drop-off, and scheduling a truck for sofas, tables, and benches."
     ),
    ("h-white-glove",
     "White-glove delivery: we carry the piece inside, place it, and remove packaging. Available on tables and sofas."
     ),
    ("h-redelivery",
     "Redelivery policy: if you miss the delivery window we will book a second drop at no extra charge."
     ),
    ("h-returns-window",
     "Returns window: most items can be returned within 30 days. Tables and benches must be unused and in original packaging."
     ),
    ("p-sof-2201",
     "SOF-2201 compact loveseat, stain-resistant fabric, in-stock sofa for small rooms."
     ),
    ("h-pet-damage",
     "Pet damage and accidental scratches: what the sofa warranty does and does not cover when a cat or dog claws the fabric."
     ),
    ("p-chr-0804",
     "CHR-0804 lounge chair, oak legs, linen upholstery, indoor use only."),
]

# Two dummy questions
STORE_QUERIES = [
    ("meaning", "my cat scratched the sofa cushions, is that covered?",
     "h-warranty-cushion"),
    ("code", "OAK-2291 delivery time", "p-oak-2291"),
]
