"""Same information, different token counts. Characters are cheap to count
but tokens are what you pay for. pip install tiktoken; runs offline after
the first download of the encoding file."""
import json

import tiktoken

from common import SYSTEM, TICKETS

enc = tiktoken.get_encoding("o200k_base")

order = {"order": 4821, "customer": "Dana", "team": "shipping",
         "urgency": "high", "channel": "email",
         "draft": "Replacement ships Tuesday."}

samples = {
    "ticket, English (Dana)": TICKETS["broken_table"],
    "ticket, Spanish (Mateo)": TICKETS["lamp_spanish"],
    "ticket, Hindi (same as Dana's)": TICKETS["table_hindi"],
    "system prompt (episode 1)": SYSTEM,
    "JSON, pretty (indent=2)": json.dumps(order, indent=2),
    "JSON, minified": json.dumps(order, separators=(",", ":")),
    "Python code":
        'def urgency(ticket):\n'
        '    return "high" if "broken" in ticket else "low"',
    "a long number": "8675309867530902",
    "emoji": "😀🎉👍",
}

print(f"{'text':34} {'chars':>6} {'tokens':>7} {'chars/token':>12}")
for name, text in samples.items():
    n = len(enc.encode(text))
    print(f"{name:34}: {len(text):>6} {n:>7} {len(text) / n:>12.1f}")
