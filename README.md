# LLM demos

Two sets of demos:

- `p1/` — Go programs for tokenization, KV-cache size, and streaming time-to-first-token. Tokenization uses [tiktoken-go](https://github.com/pkoukk/tiktoken-go), a Go port of OpenAI's tiktoken (`o200k_base` and `cl100k_base`, same BPE files). The other two programs use only the standard library.
- `p2/` — Python demos (support-triage themed): `tokens_demo.py`, `context_math.py`, and `cost_calc.py` run offline; `stream_meter.py`, `temperature_demo.py`, and `structured_output.py` need a running server.
- `p3/` — Python demos for the KV cache: `kv_math.py` (bytes per token, offline) and `prefix_cache.py` (llama.cpp prompt-prefix reuse; needs a single-slot server, see its docstring).
- `p4/` — Python vector-search demos (see `p4/README.md`): `embed_rate.py`, `hybrid.py`, and `embed_words.py` need the embedding server; `real_bench.py`, `ann_bench.py`, and `fresh_filter.py` run on synthetic vectors offline.

Each folder is self-contained: its own `setup.sh` / `teardown.sh`,
`requirements.txt` or `go.mod`, and `Dockerfile`, with pid/log files under
`pN/.demo/`. Shared shell helpers (venv creation, server start/stop, health
waits) live in `scripts/demo-lib.sh`, and every llama-server shares the
model cache in `.demo/models`. The root `setup.sh` / `teardown.sh` just run
every folder's script; run one folder's directly to set up only that part.

```bash
./setup.sh        # or just one part: ./p2/setup.sh
PY2=p2/.demo/venv/bin/python
PY3=p3/.demo/venv/bin/python
PY4=p4/.demo/venv/bin/python

$PY2 p2/tokens_demo.py          # chars vs tokens per sample
$PY2 p2/context_math.py         # how chat history inflates input tokens
$PY2 p2/cost_calc.py            # monthly cost per model / caching / batch

$PY2 p2/stream_meter.py         # TTFT and tokens/s for one triage call
$PY2 p2/temperature_demo.py     # temp 0 vs 1.2, 10 answers each
$PY2 p2/structured_output.py    # prompt-only vs server-enforced schema

$PY3 p3/kv_math.py              # KV cache bytes per token
$PY3 p3/prefix_cache.py         # prefix reuse; needs ./p3/setup.sh's server

$PY4 p4/embed_rate.py           # embedding throughput on this laptop
$PY4 p4/hybrid.py               # BM25 + vector search fused with RRF
$PY4 p4/real_bench.py           # Faiss exact/HNSW/IVF/PQ recall vs speed

./teardown.sh   # or just one part: ./p3/teardown.sh
```

The image compiles static binaries with Chainguard's Go toolchain, then copies them into `cgr.dev/chainguard/static:latest`, which has no shell. Run `llama-server` on the laptop; the streaming program calls it from the container.

## Build

```bash
docker build -t llm-demos ./p1
```

Each Python folder has its own `Dockerfile` on Chainguard's Python image
(`latest-dev` installs the venv, `latest` runs it):

```bash
docker build -t llm-demos-p2 ./p2
docker build -t llm-demos-p3 ./p3
docker run --rm llm-demos-p3 kv_math.py
```

Server-backed demos reach the laptop's llama-server via
`host.docker.internal`:

```bash
docker run --rm -e BASE_URL=http://host.docker.internal:8080/v1 \
  llm-demos-p2 stream_meter.py

docker run --rm -e BASE_URL=http://host.docker.internal:8393 \
  llm-demos-p3 prefix_cache.py
```

## tokenize_demo

The first run downloads tiktoken encoding files, so the container needs network access.

```bash
go -C p1 run ./cmd/tokenize_demo
docker run --rm llm-demos
```

## kv_cache_math

```bash
go -C p1 run ./cmd/kv_cache_math
docker run --rm --entrypoint /usr/bin/kv_cache_math llm-demos
```

## stream_ttft

Start llama.cpp's server on the laptop first. The helper scripts do this and
clean up afterwards:

```bash
./p2/setup.sh    # go mod download, venv, then start the chat llama-server
./p3/setup.sh    # venv, then start the single-slot prefix-cache server
./p4/setup.sh    # venv, then start the embedding server
./teardown.sh    # run every p*/teardown.sh, then delete the shared model cache
```

Each `setup.sh` waits for its servers to report healthy and prints the URLs
to use (`BASE_URL` for chat, `EMBED_URL` for embeddings). They only ever
manage the processes they started; a server you launched yourself is left
alone. Override the defaults with environment variables:

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M PORT=8392 CONTEXT=16384 ./p2/setup.sh
```

Or start it by hand, for example:

```bash
llama-server -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M \ # download GGUF model directly from Hugging Face ~ 4GB memory
  -ngl 99 \ # Offload 100% of the model layers to the M4's GPU via Metal
  -fa on \ # enable flash attention to cut memory bandwidth usage
  -c 16384 \ # Allocate a 16K token context window
  --port 8080
```

Inside the container, `127.0.0.1` is the container, not the laptop. On Docker Desktop the laptop is `host.docker.internal`.

```bash
BASE_URL=http://127.0.0.1:8080/v1 MODEL=local \
  go -C p1 run ./cmd/stream_ttft "Explain what a KV cache is in three sentences."

docker run --rm --entrypoint /usr/bin/stream_ttft \
  -e BASE_URL=http://host.docker.internal:8080/v1 \
  -e MODEL=local \
  llm-demos "Explain what a KV cache is in three sentences."
```

A hosted OpenAI-compatible API works the same way. Export `API_KEY` in the shell first. The program reads that variable, and Docker's `-e API_KEY` forwards it into the container without putting the key on the command line.

```bash
export API_KEY  # already set in the environment

BASE_URL=https://api.openai.com/v1 MODEL=gpt-4o-mini \
  go -C p1 run ./cmd/stream_ttft

docker run --rm --entrypoint /usr/bin/stream_ttft \
  -e BASE_URL=https://api.openai.com/v1 \
  -e MODEL=gpt-4o-mini \
  -e API_KEY \
  llm-demos
```
