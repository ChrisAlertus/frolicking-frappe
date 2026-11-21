# Time embedding throughput against the local embedding server, then
# extrapolate: how long a million or a billion chunks would take at the
# measured rate, and the storage for a billion 768-dim float32 vectors
# (1e9 * 768 * 4 bytes = 3.07 TB).

from __future__ import annotations

import argparse
import time

from common import DIM, embed, require_embed_server

# Filler vocabulary for the synthetic ~45-word chunks.
_WORDS = (
    "warranty sofa cushion fabric velvet delivery table bench oak chair "
    "lounge indoor outdoor scratch pet cover claim shipping returns window "
    "white glove packaging stock living dining patio teak performance "
).split()


def make_chunks(n: int, words_per: int = 45) -> list[str]:
    chunks = []
    for i in range(n):
        toks = [_WORDS[(i * 3 + j) % len(_WORDS)] for j in range(words_per)]
        chunks.append(" ".join(toks))
    return chunks


def main():
    p = argparse.ArgumentParser(description="Time embedding throughput.")
    p.add_argument("--n", type=int, default=1024, help="chunks to embed")
    p.add_argument("--words", type=int, default=45)
    args = p.parse_args()

    require_embed_server()
    chunks = make_chunks(args.n, args.words)
    prefixed = ["search_document: " + c for c in chunks]

    t0 = time.perf_counter()
    _ = embed(prefixed)
    elapsed = time.perf_counter() - t0
    rate = args.n / elapsed if elapsed else float("inf")

    hours_1m = 1_000_000 / rate / 3600
    hours_1b = 1_000_000_000 / rate / 3600
    storage_1b_tb = 1_000_000_000 * DIM * 4 / 1e12  # 3.072 TB → printed 3.07 TB

    print(f"embedded {args.n:,} chunks of ~{args.words} words in {elapsed:.1f} s = {rate:.0f} chunks/s on this laptop")
    print(f"  1 million chunks at this rate: {hours_1m:.1f} hours = {hours_1m/24:.1f} days")
    print(f"  1 billion chunks at this rate: {hours_1b:,.1f} hours = {hours_1b/24:.1f} days")
    print(f"  storage for 1 billion vectors of {DIM} float32 numbers: {storage_1b_tb:.2f} TB")


if __name__ == "__main__":
    main()
