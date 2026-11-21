"""Stream one chat request over SSE and measure TTFT and tokens/sec.

Works with any OpenAI-compatible /v1/chat/completions endpoint:
a local llama-server or vLLM (free), or a hosted API (costs money).
Usage:
  BASE_URL=http://127.0.0.1:8080/v1 MODEL=local python3 stream_ttft.py
  "your prompt"
  BASE_URL=https://api.openai.com/v1 MODEL=gpt-4o-mini API_KEY=sk-...
  python3 stream_ttft.py
"""
import json, os, sys, time
import requests

url = os.environ.get("BASE_URL",
                     "http://127.0.0.1:8080/v1") + "/chat/completions"
headers = {"Authorization": f"Bearer {os.environ.get('API_KEY', 'none')}"}
body = {
    "model":
    os.environ.get("MODEL", "local"),
    "stream":
    True,
    "stream_options": {
        "include_usage": True
    },  # last chunk carries token counts
    "messages": [{
        "role":
        "user",
        "content":
        sys.argv[1] if len(sys.argv) > 1 else
        "Explain what a KV cache is in three sentences."
    }]
}

t0 = time.perf_counter()
first, chunks, usage = None, 0, None
with requests.post(url, json=body, headers=headers, stream=True,
                   timeout=120) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue  # SSE: skip blanks/comments
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        event = json.loads(data)
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            text = choice.get("delta", {}).get("content")
            if text:
                first = first or time.perf_counter()  # time to first token
                chunks += 1
                print(text, end="", flush=True)
end = time.perf_counter()

out_tokens = (usage or {}).get("completion_tokens", chunks)
print(f"\n\nTTFT: {(first - t0) * 1000:.0f} ms")
print(f"total: {(end - t0):.2f} s, output tokens: {out_tokens}, "
      "text chunks: {chunks}")
print(f"decode speed: {(out_tokens - 1) / (end - first):.1f} tokens/s "
      "(after first token)")
if usage:
    print(f"input tokens: {usage.get('prompt_tokens')}")
