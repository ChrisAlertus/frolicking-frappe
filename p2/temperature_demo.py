"""Same question, same model, two temperatures. At 0 the answer is stable;
at 1.2 it wanders. Run it a few times — the 1.2 line keeps changing.
Needs a server running (./setup.sh)."""
from collections import Counter

from common import TICKETS, chat

Q = ("Which team should handle this ticket: shipping, returns, billing or account? "
     "Answer with one word.\n\n" + TICKETS["broken_table"])

for temp in (0, 1.2):
    answers = Counter()
    for _ in range(10):
        out = chat([{"role": "user", "content": Q}],
                   temperature=temp, max_tokens=5)
        answers[out["choices"][0]["message"]["content"]
                .strip().strip(".").lower()] += 1
    print(f"temperature {temp}: {dict(answers)}")
