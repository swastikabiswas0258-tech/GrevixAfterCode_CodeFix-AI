"""CodeFix-AI configuration (Google Gemini).

GET A FREE KEY
--------------
Create one at https://aistudio.google.com/apikey (no credit card needed for the free
tier, which covers the Flash / Flash-Lite models). Free-tier requests are rate limited,
and Google may use free-tier data to improve its products, so avoid pasting private or
confidential code while on the free tier.

API KEY
-------
Replace the placeholder below with your own key, OR (recommended) put it in a `.env`
file as GEMINI_API_KEY=... so it never ends up in source control.

The placeholder is NOT a working key. While it is unchanged, CodeFix-AI runs in
"local mode": the built-in parser still isolates and fixes Python syntax errors and
bracket/quote errors in every language, but AI analysis of other bugs is off.

Lookup order:  1) GEMINI_API_KEY (environment variable / .env)
               2) GOOGLE_API_KEY (also read by the Google SDK)
               3) DEFAULT_API_KEY below
"""

import os

PLACEHOLDER_KEY = "YOUR_GEMINI_KEY_HERE"

# >>> Replace this value with your real Gemini key (or leave it and use .env) <<<
DEFAULT_API_KEY = "YOUR_DEFAULT_GEMINI_KEY_HERE"

# Model names change often. Override with GEMINI_MODEL in .env if this one is retired.
DEFAULT_MODEL = "gemini-3.5-flash"
# Tried automatically, in order, if the model above is not found for your key.
FALLBACK_MODELS = ["gemini-2.5-flash"]


def _clean(key):
    """Remove spaces and stray quotes that often sneak in when pasting a key."""
    return (key or "").strip().strip("\"'").strip()


def _usable(key):
    key = _clean(key)
    return bool(key) and key != PLACEHOLDER_KEY and not key.lower().startswith("your")


def key_shape_warning(key):
    """Gemini keys from AI Studio start with 'AIza'. Return a hint if the key looks wrong."""
    key = _clean(key)
    if key.startswith("sk-ant"):
        return "That looks like an Anthropic key, not a Gemini key."
    if key.startswith("sk-"):
        return "That looks like an OpenAI-style key, not a Gemini key."
    if key and not key.startswith("AIza"):
        return "Gemini keys from Google AI Studio normally start with 'AIza'."
    return ""


def get_api_key():
    """Return a usable key, or None if only the placeholder is configured."""
    for candidate in (os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY"), DEFAULT_API_KEY):
        if _usable(candidate):
            return _clean(candidate)
    return None


def get_models():
    """Models to try, in order (the configured one first, then fallbacks)."""
    first = (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    return [first] + [m for m in FALLBACK_MODELS if m != first]