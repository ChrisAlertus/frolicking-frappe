"""KV cache size per token = 2 (K and V) x layers x kv_heads x head_dim x bytes.
Configs from the models' published config.json files (Hugging Face).
"""

models = {  # name: (layers, kv_heads, head_dim)
    "Llama-3.1-8B": (32, 8, 128),
    "Llama-3.1-70B": (80, 8, 128),
}
BYTES = 2  # fp16 / bf16

for name, (layers, kv_heads, head_dim) in models.items():
    per_token = 2 * layers * kv_heads * head_dim * BYTES
    print(f"{name}: {per_token / 1024:.0f} KiB per token")
    for ctx in (1_000, 8_000, 128_000):
        print(f"   {ctx:>7,} tokens -> {per_token * ctx / 1024**3:6.2f} "
              "GiB for ONE conversation")
