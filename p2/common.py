"""Shared bits for the support-triage demos: the system prompt, a few
customer tickets, and a tiny chat() wrapper for any OpenAI-compatible
/v1/chat/completions endpoint (local llama-server by default)."""
import os

import requests

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("MODEL", "local")
API_KEY = os.environ.get("API_KEY", "none")

SYSTEM = (
    "You are a support triage assistant for our online furniture store. "
    "Teams: shipping, returns, billing, account. "
    "Reply with three lines: 'team: <one team>, 'urgency: low|normal|high', "
    "then 'draft:' followed by a short, polite reply in the customer's language. "
    "Never promise refunds or credits; a human decides those."
)

TICKETS = {
    "broken_table":
        "My wooden dining table arrived yesterday with a deep crack across "
        "the top and one leg wobbles so badly. Order #4821. I have photos "
        "if needed. I'd like a replacement sent as soon as possible, "
        "please — it is our only table.",
    "lamp_spanish":
        "Hola buenas, la lámpara de pie que pedí llegó con la pantalla "
        "rota y no enciende. Pedido #7315. ¿Pueden enviarme un repuesto o "
        "decirme cómo devolverla? Es para el salón de mamá.",
    "table_hindi":
        "मेरी लकड़ी की डाइनिंग टेबल कल पहुँची, लेकिन उसके ऊपर एक गहरी दरार "
        "है और एक पैर बहुत हिलता है। ऑर्डर #4821। मेरे पास फोटो हैं, जरूरत "
        "हो तो मैं भेज सकता हूँ। कृपया जल्द से जल्द नई टेबल भिजवा दें, यह "
        "हमारी इकलौती डाइनिंग टेबल है।",
    "angry_refund":
        "This is the second time my order arrived damaged. I want my money "
        "back right now — refund me in full today or I am calling my bank "
        "and disputing the charge. Order #6610.",
}


def chat(messages, **params):
    """One chat completion. Extra keyword arguments (temperature,
    max_tokens, response_format, ...) go straight into the request body."""
    r = requests.post(
        BASE_URL + "/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages, **params},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()
