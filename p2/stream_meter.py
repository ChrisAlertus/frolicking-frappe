"""One streamed triage call, measured. This is where the 'measured'
SHORT_IN, OUT numbers in cost_calc.py come from. Needs a server running
(./setup.sh) or any OpenAI-compatible endpoint via BASE_URL/MODEL."""
import json
import time

import requests

from common import API_KEY, BASE_URL, MODEL, SYSTEM, TICKETS

body = {
    "model": MODEL,
    "stream": True,
    "stream_options": {"include_usage": True},
    "messages": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": TICKETS["broken_table"]}],
}

t0 = time.perf_counter()
first, usage = None, None
with requests.post(BASE_URL + "/chat/completions",
                   headers={"Authorization": f"Bearer {API_KEY}"},
                   json=body, stream=True, timeout=120) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        event = json.loads(data)
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            if choice.get("delta", {}).get("content"):
                first = first or time.perf_counter()
end = time.perf_counter()

out_tokens = usage.get("completion_tokens")
print("--- meter ---")
print("input tokens:", usage.get("prompt_tokens"))
print("output tokens:", out_tokens)
print(f"time to first token: {(first - t0) * 1000:.0f} ms")
print(f"total: {end - t0:.2f} s")
print(f"decode speed: {(out_tokens - 1) / (end - first):.0f} tokens/s")
