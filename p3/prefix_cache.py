"""Prefix-cache demo against llama.cpp's llama-server.

Server (single slot, so each request reuses or evicts the one cache):

    llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M \
        --port 8393 -c 16384 -np 1 --cache-ram 0
"""
import os
import time

import requests

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8393")
FILLER = open(os.path.join(os.path.dirname(__file__), "policy.txt")).read()


def tokenize(text):
    r = requests.post(f"{BASE}/tokenize", json={"content": text})
    return len(r.json()["tokens"])


def complete(prompt, n_predict=64, cache_prompt=True):
    t = time.time()
    r = requests.post(f"{BASE}/completion", json={
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": cache_prompt,
    }).json()
    r["wall_ms"] = (time.time() - t) * 1000
    return r


def prefix_of(n_tokens):
    """A prompt of exactly n_tokens, made by repeating the policy text.

    Trimming matters: with -c 16384 there are only ~380 tokens of headroom
    for the question and the answer, so an overshooting prefix would not
    fit the window."""
    text = FILLER
    while tokenize(text) < n_tokens:
        text += "\n\n" + FILLER
    tokens = requests.post(f"{BASE}/tokenize", json={"content": text}).json()["tokens"]
    r = requests.post(f"{BASE}/detokenize", json={"tokens": tokens[:n_tokens]})
    return r.json()["content"]


doc = prefix_of(16000)
qa = "\n\nQuestion: can a customer return an unwrapped mattress?\nAnswer:"
qb = "\n\nQuestion: who pays the stairs fee?\nAnswer:"


def show(label, r):
    t = r["timings"]
    print(f"{label:44s} computed {t['prompt_n']:>6,} tokens, "
          f"reused {t.get('cache_n', 0):>6,}, "
          f"prefill {t['prompt_ms']:>7,.0f} ms")


# empty the slot first
complete("Unrelated warm-up so the server holds nothing from the policy.", 1)
show("1. cold: 16K policy + question A", complete(doc + qa, 32))
show("2. same policy + question B (prefix hit)", complete(doc + qb, 32))
show("3. one word changed at the START + question B",
     complete("Note: " + doc + qb, 32))
show("4. same as 3 again", complete("Note: " + doc + qb, 32))
