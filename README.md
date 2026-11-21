# LLM demos

Small programs for tokenization, KV-cache size, and streaming time-to-first-token. Tokenization uses [tiktoken-go](https://github.com/pkoukk/tiktoken-go), a Go port of OpenAI's tiktoken (`o200k_base` and `cl100k_base`, same BPE files). The other two programs use only the standard library.

The image compiles static binaries with Chainguard's Go toolchain, then copies them into `cgr.dev/chainguard/static:latest`, which has no shell. Run `llama-server` on the laptop; the streaming program calls it from the container.

## Build

```bash
docker build -t llm-demos .
```

## tokenize_demo

The first run downloads tiktoken encoding files, so the container needs network access.

```bash
go run ./cmd/tokenize_demo
docker run --rm llm-demos
```

## kv_cache_math

```bash
go run ./cmd/kv_cache_math
docker run --rm --entrypoint /usr/bin/kv_cache_math llm-demos
```

## stream_ttft

Start llama.cpp's server on the laptop first, for example:

```bash
llama-server -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M \ # download GGUF model directly from Hugging Face ~ 4GB memory
  -ngl 99 \ # Offload 100% of the model layers to the M4's GPU via Metal
  -fa on \ # enable flash attention to cut memory bandwidth usage
  -c 32768 \ # Allocate a 32K token context window
  --port 8080
```

Inside the container, `127.0.0.1` is the container, not the laptop. On Docker Desktop the laptop is `host.docker.internal`.

```bash
BASE_URL=http://127.0.0.1:8080/v1 MODEL=local \
  go run ./cmd/stream_ttft "Explain what a KV cache is in three sentences."

docker run --rm --entrypoint /usr/bin/stream_ttft \
  -e BASE_URL=http://host.docker.internal:8080/v1 \
  -e MODEL=local \
  llm-demos "Explain what a KV cache is in three sentences."
```

A hosted OpenAI-compatible API works the same way. Export `API_KEY` in the shell first. The program reads that variable, and Docker's `-e API_KEY` forwards it into the container without putting the key on the command line.

```bash
export API_KEY  # already set in the environment

BASE_URL=https://api.openai.com/v1 MODEL=gpt-4o-mini \
  go run ./cmd/stream_ttft

docker run --rm --entrypoint /usr/bin/stream_ttft \
  -e BASE_URL=https://api.openai.com/v1 \
  -e MODEL=gpt-4o-mini \
  -e API_KEY \
  llm-demos
```
