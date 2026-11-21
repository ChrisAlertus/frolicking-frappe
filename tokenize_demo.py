"""What the model actually sees: text -> token IDs. Runs offline after the
first download of the encoding files. pip install tiktoken"""
import tiktoken

enc = tiktoken.get_encoding("o200k_base")  # GPT-4o family encoding
old = tiktoken.get_encoding("cl100k_base")  # GPT-4 / GPT-3.5 encoding

prompt = "Plan a 3-day trip to Lisbon for two people on a budget."
ids = enc.encode(prompt)
print("prompt:", prompt)
print("token ids:", ids)
print("pieces:", [enc.decode([i]) for i in ids])
print(f"{len(prompt)} characters -> {len(ids)} tokens (o200k), "
      f"{len(old.encode(prompt))} tokens (cl100k)\n")

samples = {
    "English": "How are you today?",
    "Hindi": "आज आप कैसे हैं?",
    "Code": "for i in range(10): print(i)",
    "Number": "3.14159265358979",
    "Word": "strawberry",
}
for name, text in samples.items():
    t = enc.encode(text)
    print(
        f"{name:8} {len(text):3} chars -> {len(t):2} tokens (o200k), "
        f"{len(old.encode(text)):2} (cl100k)  {[enc.decode([i]) for i in t]}")

print("\nvocabulary sizes:", enc.n_vocab, "(o200k)", old.n_vocab, "(cl100k)")
