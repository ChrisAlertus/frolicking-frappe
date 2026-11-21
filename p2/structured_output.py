"""A schema beats asking nicely. The prompt is identical in both halves;
only B constrains the decoding, so B always parses. Note what the schema
cannot fix: the drafts still mention refunds the prompt told the model
not to promise. Needs a server running (./setup.sh)."""
import json

import jsonschema

from common import SYSTEM, TICKETS, chat

SCHEMA = {"type": "object",
          "properties": {
              "team": {"enum": ["shipping", "returns", "billing", "account"]},
              "urgency": {"enum": ["low", "normal", "high"]},
              "draft": {"type": "string"}},
          "required": ["team", "urgency", "draft"]}

MESSAGES = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TICKETS["angry_refund"]}]

N = 20


def run(label, schema_enforced=False):
    params = {}
    if schema_enforced:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "triage", "schema": SCHEMA}}
    valid, refunds, bad_example = 0, 0, None
    for _ in range(N):
        text = chat(MESSAGES, temperature=0.8,
                    **params)["choices"][0]["message"]["content"]
        try:
            reply = json.loads(text)
            if schema_enforced:
                jsonschema.validate(reply, SCHEMA)
            valid += 1
            draft = reply.get("draft", "") if isinstance(reply, dict) else ""
        except (json.JSONDecodeError, jsonschema.ValidationError):
            draft = text
            if bad_example is None:
                bad_example = text
        if "refund" in draft.lower():
            refunds += 1
    print(label)
    if N - valid:
        print(f"  not JSON: {N - valid}/{N}")
    print(f"  valid: {valid}/{N}")
    print(f"  replies that mention a refund anyway: {refunds}/{N}")
    if bad_example:
        snippet = bad_example[:160].replace("\n", "\\n")
        print(f"  failed reply (first 160 chars): {snippet}")


run("A: prompt only")                       # A: ask nicely
run("B: prompt + enforced schema",          # B: same prompt + a schema
    schema_enforced=True)                   #    the server enforces
