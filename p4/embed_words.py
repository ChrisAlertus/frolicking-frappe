# Embed a word list with the local embedding server and save the matrix as
# .npy for real_bench.py. Full scale is 235,476 words: 235,476 * 768 * 4
# bytes = 0.72 GB. Without --words-file, placeholder tokens are synthesized.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import DIM, embed, require_embed_server


def load_words(path: str | None, n: int) -> list[str]:
    if path:
        text = Path(path).read_text(errors="replace")
        words = [w.strip() for w in text.splitlines() if w.strip() and " " not in w.strip()]
        if not words:
            words = text.split()
        return words[:n] if n else words
    # placeholder stand-in for a real dictionary list
    return [f"word{i:06d}" for i in range(n)]


def main():
    p = argparse.ArgumentParser(description="Embed a word list and save the embedding matrix.")
    p.add_argument("--words-file", default=None, help="one word per line")
    p.add_argument("--n", type=int, default=235_476, help="how many words (use a small value for a quick demo)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", default="word_embeddings.npy")
    args = p.parse_args()

    n = args.n
    require_embed_server()
    words = load_words(args.words_file, n)
    print(f"embedding {len(words)} words in batches of {args.batch} -> {args.out}")

    parts = []
    for i in range(0, len(words), args.batch):
        batch = words[i : i + args.batch]
        parts.append(embed(["search_document: " + w for w in batch]))
        print(f"  {min(i + args.batch, len(words))}/{len(words)}")
    xb = np.vstack(parts).astype(np.float32)
    np.save(args.out, xb)
    gb = xb.nbytes / 1e9
    print(f"saved {xb.shape[0]} x {xb.shape[1]} float32 ({gb:.2f} GB) to {args.out}")
    print(f"(full scale: 235,476 x {DIM} = 0.72 GB)")


if __name__ == "__main__":
    main()
