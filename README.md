# LLM demos

Small scripts for tokenization, KV-cache size, and streaming time-to-first-token. Packages are installed with [uv](https://docs.astral.sh/uv/) in a build stage, then copied into Chainguard's hardened Python image (`cgr.dev/chainguard/python:latest`), which has no shell and no package manager. Run `llama-server` on the laptop; the streaming script calls it from the container.

## Build

```bash
docker build -t llm-demos .
```

## tokenize_demo.py

The first run downloads tiktoken encoding files, so the container needs network access.

```bash
docker run --rm llm-demos python tokenize_demo.py
```

## kv_cache_math.py

```bash
docker run --rm llm-demos python kv_cache_math.py
```

## stream_ttft.py

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
docker run --rm \
  -e BASE_URL=http://host.docker.internal:8080/v1 \
  -e MODEL=local \
  llm-demos python stream_ttft.py "Explain what a KV cache is in three sentences."
```

A hosted OpenAI-compatible API works the same way:

```bash
docker run --rm \
  -e BASE_URL=https://api.openai.com/v1 \
  -e MODEL=gpt-4o-mini \
  -e API_KEY=sk-... \
  llm-demos python stream_ttft.py
```
