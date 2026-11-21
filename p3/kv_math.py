# kv_math.py: KV cache bytes per token
LAYERS, KV_HEADS, HEAD_DIM = 32, 8, 128  # Llama-3.1-8B

per_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2   # K+V, fp16
# Llama-3.1-8B: 2*32*8*128*2 = 128 KiB/token -> 15.6 GiB at 128K tokens

KiB, GiB = 2**10, 2**30

print(f"Llama-3.1-8B KV cache: {per_token / KiB:.0f} KiB per token")
for tokens in (16_384, 32_768, 131_072):
    print(f"  {tokens:>7,} tokens: {per_token * tokens / GiB:5.1f} GiB")
