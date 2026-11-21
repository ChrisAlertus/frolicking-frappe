# Vector-search demos

Embedding and ANN benchmarks over 768-dim vectors, plus a hybrid
BM25 + vector search over a toy furniture-store corpus.
`embed_rate`, `hybrid`, and `embed_words` need the embedding server.
`real_bench`, `ann_bench`, and `fresh_filter` run on synthetic vectors
and do not.

## What is linked

The image is a static binary (`CGO_ENABLED=0`), so these are the pieces
that fit:

- Exact search is a float32 inner-product scan.
- HNSW is [github.com/coder/hnsw](https://github.com/coder/hnsw). On unit
  vectors its cosine distance ranks the same as inner product. `EfSearch`
  is used while inserting and while searching; the benches insert at 128,
  then lower it for the ef sweep. That graph stops a search once a round
  stops improving the best hit, so on these random vectors a higher
  `efSearch` often does not move recall the way Faiss does.
- IVF-Flat is a small in-process index: spherical k-means on a sample of
  the rows, then `nprobe` lists rescored with the original vectors.
- BM25 is Okapi with the `rank_bm25` defaults (`k1=1.5`, `b=0.75`).

Faiss is not linked. IVF-PQ (96-byte codes and the full-vector re-rank)
and an in-graph bitmap selector (`IDSelectorBitmap`) need libfaiss.
`fresh_filter` keeps deletes as a tombstone mask, which is what Faiss
HNSW does; `coder/hnsw`'s `Delete` would remove the node, and the bench
does not call it. Filtered search is top-k-then-filter plus an exact scan
of the matches.

## Setup

`./p4/setup.sh` runs `go mod download` and starts the embedding server
(`--pooling mean`) with pid/log files in `p4/.demo/`.
`./p4/teardown.sh` stops it and cleans up.

Or by hand:

```bash
# embedding server — needed for embed_rate, hybrid, embed_words
llama-server -hf nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0 --embedding --pooling mean --port 8395 &
```

Or in Docker (the synthetic benches need no server; the rest reach the
laptop's server via `host.docker.internal`):

```bash
docker build -t llm-demos-p4 ./p4
docker run --rm llm-demos-p4 -n 20000
docker run --rm --entrypoint /usr/bin/hybrid \
  -e EMBED_URL=http://host.docker.internal:8395/v1 llm-demos-p4
```

Full-scale benches can exceed Docker Desktop's default ~2 GB VM and get
OOM-killed (exit 137) — `ann_bench` at 1M vectors alone needs ≈ 3.1 GB.
Run full-scale benches on the laptop instead
(`go run ./cmd/ann_bench`), pass a smaller `-n` in Docker, or raise the VM
limit under Settings → Resources → Memory.

`llama-server` comes from [llama.cpp](https://github.com/ggml-org/llama.cpp). The model is [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (GGUF Q8_0). Documents must be prefixed with `search_document: ` and questions with `search_query: `.

## How to run

From this directory, with the server up for the first three:

```bash
go run ./cmd/embed_rate
go run ./cmd/hybrid
go run ./cmd/embed_words -n 512 -out word_embeddings.f32emb
go run ./cmd/real_bench -n 5000 -queries 50
go run ./cmd/real_bench -vectors word_embeddings.f32emb
go run ./cmd/ann_bench -n 20000
go run ./cmd/fresh_filter -n 5000 -add 100 -delete 100
```

The benches default to full scale, so they are heavy on RAM and time:

```bash
go run ./cmd/embed_words                 # 235,476 words, 0.72 GB of vectors
go run ./cmd/real_bench                  # 235,476 synthetic vectors + 500 queries
go run ./cmd/ann_bench                   # 1,000,000 synthetic vectors ≈ 3.07 GB RAM
go run ./cmd/fresh_filter                # 200,000 products + 1,000 adds + 1,000 tombstones
```

`embed_words` writes `word_embeddings.f32emb` (a small header plus raw
float32 rows), which `real_bench -vectors` reads. It is not a `.npy`.
