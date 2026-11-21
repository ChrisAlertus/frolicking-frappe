# LLM demos

Two sets of demos:

- `p1/` — Go programs for tokenization, KV-cache size, and streaming time-to-first-token. Tokenization uses [tiktoken-go](https://github.com/pkoukk/tiktoken-go), a Go port of OpenAI's tiktoken (`o200k_base` and `cl100k_base`, same BPE files). The other two programs use only the standard library.
- `p2/` — support-triage themed demos in Go: `tokens_demo`, `context_math`, and `cost_calc` run offline; `stream_meter`, `temperature_demo`, and `structured_output` need a running server.
- `p3/` — Go demos for the KV cache, standard library only: `kv_math` (bytes per token, offline) and `prefix_cache` (llama.cpp prompt-prefix reuse; needs a single-slot server, see its doc comment).

```bash
./setup.sh

go -C p2 run ./cmd/tokens_demo          # chars vs tokens per sample
go -C p2 run ./cmd/context_math         # how chat history inflates input tokens
go -C p2 run ./cmd/cost_calc            # monthly cost per model / caching / batch

go -C p2 run ./cmd/stream_meter         # TTFT and tokens/s for one triage call
go -C p2 run ./cmd/temperature_demo     # temp 0 vs 1.2, 10 answers each
go -C p2 run ./cmd/structured_output    # prompt-only vs server-enforced schema

go -C p3 run ./cmd/kv_math              # KV cache bytes per token
go -C p3 run ./cmd/prefix_cache         # prefix reuse; needs the server from its doc comment

./teardown.sh
```

The image compiles static binaries with Chainguard's Go toolchain, then copies them into `cgr.dev/chainguard/static:latest`, which has no shell. Run `llama-server` on the laptop; the streaming program calls it from the container.

## Build

```bash
docker build -t llm-demos ./p1
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
./setup.sh    # go mod download, then start llama-server in the background
./teardown.sh # stop it, delete the model cache and logs in .demo/
```

`setup.sh` waits for the server to report healthy and prints the `BASE_URL`
to use. It only ever manages the process it started; a server you launched
yourself is left alone. Override the defaults with environment variables:

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M PORT=8392 CONTEXT=8192 ./setup.sh
```

Or start it by hand, for example:

```bash
llama-server -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M \ # download GGUF model directly from Hugging Face ~ 4GB memory
  -ngl 99 \ # Offload 100% of the model layers to the M4's GPU via Metal
  -fa on \ # enable flash attention to cut memory bandwidth usage
  -c 8192 \ # Allocate an 8K token context window
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
