# Vector-search demos

Embedding and ANN benchmarks over 768-dim vectors, plus a hybrid
BM25 + vector search over a toy furniture-store corpus.
`embed_rate.py`, `hybrid.py`, and `embed_words.py` need the embedding
server; `real_bench.py`, `ann_bench.py`, and `fresh_filter.py` run on
synthetic vectors offline.

## Setup

`./p4/setup.sh` creates a venv with `numpy faiss-cpu hnswlib rank_bm25
requests` in `p4/.demo/venv` and starts the embedding server
(`--pooling mean`) with pid/log files in `p4/.demo/`.
`./p4/teardown.sh` stops it and cleans up.

Or by hand:

```bash
python3 -m venv /tmp/sd03/venv
/tmp/sd03/venv/bin/pip install numpy faiss-cpu hnswlib rank_bm25 requests

# embedding server — needed for embed_rate.py, hybrid.py, embed_words.py
llama-server -hf nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0 --embedding --pooling mean --port 8395 &
```

Or in Docker (the synthetic benches need no server; the rest reach the
laptop's server via `host.docker.internal`):

```bash
docker build -t llm-demos-p4 ./p4
docker run --rm llm-demos-p4 ann_bench.py
docker run --rm -e EMBED_URL=http://host.docker.internal:8395/v1 llm-demos-p4 hybrid.py
```

`llama-server` comes from [llama.cpp](https://github.com/ggml-org/llama.cpp). The model is [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (GGUF Q8_0). Documents must be prefixed with `search_document: ` and questions with `search_query: `.

`real_bench.py`, `ann_bench.py`, and `fresh_filter.py` use synthetic unit vectors by default and do **not** need the server. `hnswlib` is installed for completeness, but the index code uses Faiss.

## How to run

From this directory, using the venv from `./p4/setup.sh` (and the server up
for the first three):

```bash
PY=.demo/venv/bin/python
$PY embed_rate.py                 # 1,024 chunks of ~45 words, extrapolated to 1M/1B
$PY hybrid.py                     # toy store corpus, vector vs keyword vs hybrid
$PY embed_words.py --out word_embeddings.npy  # 235,476 words, 0.72 GB of vectors
$PY real_bench.py                 # 235,476 synthetic vectors + 500 queries
$PY real_bench.py --vectors word_embeddings.npy
$PY ann_bench.py                  # 1,000,000 synthetic vectors ≈ 3.07 GB RAM
$PY fresh_filter.py               # 200,000 products + 1,000 adds + 1,000 tombstones
```

The benches default to full scale, so they are heavy on RAM and time. Pass
smaller sizes for a quick run:

```bash
python embed_words.py --n 512
python real_bench.py --n 5000 --queries 50
python ann_bench.py --n 20000        # default is 1,000,000 ≈ 3.07 GB
python fresh_filter.py --n 10000 --add 100 --delete 100
```

```bash
python embed_rate.py ; python hybrid.py ; python embed_words.py
python real_bench.py ; python ann_bench.py ; python fresh_filter.py
```
