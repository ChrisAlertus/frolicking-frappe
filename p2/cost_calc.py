TICKETS_PER_DAY, DAYS = 2_000, 30
SHORT_IN, OUT = 142, 43           # measured
LONG_PROMPT = 3_000               # assumption: policy + examples, identical on every call

# model: (input, output, cache-read) in $ per 1M tokens
PRICES = {
    "claude-haiku-3-5":  (0.10, 0.50, 0.01),
    "claude-sonnet-3-5": (2.00, 10.00, 0.10),
    "claude-opus-3-5":   (4.00, 20.00, 0.20),
    "gpt-4-luna":        (0.10, 0.50, 0.01),
    "gpt-4.1-terra":     (1.00, 5.00, 0.05),
    "gpt-4.1-sol":       (2.00, 10.00, 0.10),
}
BATCH_DISCOUNT = 0.5


def month(model, tin, tout, cached=0, batch=False):
    pi, po, pc = PRICES[model]
    per = ((tin - cached) * pi + cached * pc + tout * po) / 1e6
    if batch:
        per *= 1 - BATCH_DISCOUNT
    return per * TICKETS_PER_DAY * DAYS


COLS = ("short prompt", "+3K prompt", "+3K cached", "cached+batch")

print(f"{TICKETS_PER_DAY * DAYS:,} tickets a month\n")
print(f"{'model':<18}" + "".join(f"{h:>16}" for h in COLS))
for model in PRICES:
    scenarios = (
        month(model, SHORT_IN, OUT),
        month(model, SHORT_IN + LONG_PROMPT, OUT),
        month(model, SHORT_IN + LONG_PROMPT, OUT, cached=LONG_PROMPT),
        month(model, SHORT_IN + LONG_PROMPT, OUT, cached=LONG_PROMPT,
              batch=True),
    )
    print(f"{model:<18}" + "".join(f"{f'${v:,.2f}':>16}" for v in scenarios))

print(f"""
cached: the identical {LONG_PROMPT:,}-token prefix is billed at the
cache-read price (cache writes are ignored here; they add a little on
the first call of each cache window).

batch: half price, answers within 24 hours. Fine for overnight triage,
not for live chat.
""")
