Request work from start to finish:
1. Browser: POST prompt / whole chat / steam: true
2. Gateway: auth (api key), token (rate) limits (input + max_tokens), route (load balance over region/ GPU pool)
3. Safety: cheap classifier (Jev?) for prompt injection/inappropriate
4. Tokenizer: determines token efficiency. Extra tokens get added to requests (template)
5. Scheduler: joins a running batch (continuous batching where requests join and leave batch at each step)
6. GPU: prefill (parallel over prompt tokens + compute bound) set TTFT, decode (memory bandwidth bound ) sets tokens/s
7. Stream back: SSE (because one way answer flows -> plain HTTP through every proxy). Check in chunks and cut the stream if flagged

TTFT: queue + prefill
Total latency: mostly decode
capacity = KV cache
usage recorded: input + cached input + output 
tokens determine bill + rate-limit bucket
