import os
from pathlib import Path
from typing import Any


# ── YAML loader ───────────────────────────────────────────────────────────────

def _load_yaml() -> dict:
    """Load config.yaml from the project root, if present."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(config_path.read_text()) or {}
    except Exception as e:
        import logging
        logging.getLogger("proxy").warning(f"Failed to load config.yaml: {e}")
        return {}

_yaml = _load_yaml()


def _get(env_key: str, *yaml_path: str, default: Any = None) -> Any:
    """Return env var if set, else the YAML value at yaml_path, else default."""
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val
    node = _yaml
    for key in yaml_path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node if node is not None else default


def _get_bool(env_key: str, *yaml_path: str, default: bool = False) -> bool:
    val = _get(env_key, *yaml_path, default=default)
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("1", "true", "yes")


# ── Provider ──────────────────────────────────────────────────────────────────

# Possible values: openai | openai-compat | google-api | google-oauth | google-vertex | anthropic | kimi
PREFERRED_PROVIDER = (_get("PREFERRED_PROVIDER", "provider", default="openai") or "openai").lower()

_is_google = PREFERRED_PROVIDER.startswith("google")
_is_openai_compat = PREFERRED_PROVIDER == "openai-compat"
_is_kimi = PREFERRED_PROVIDER == "kimi"

# Google auth flags — derived from provider name; legacy env vars still override
USE_GEMINI_OAUTH = _get_bool("USE_GEMINI_OAUTH") or PREFERRED_PROVIDER == "google-oauth"
USE_VERTEX_AUTH  = _get_bool("USE_VERTEX_AUTH")  or PREFERRED_PROVIDER == "google-vertex"

# ── API keys ──────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY", "anthropic",   "api_key")
OPENAI_API_KEY    = _get("OPENAI_API_KEY",    PREFERRED_PROVIDER if _is_openai_compat else "openai", "api_key")
GEMINI_API_KEY    = _get("GEMINI_API_KEY",    "google-api",  "api_key")
OPENAI_BASE_URL   = _get("OPENAI_BASE_URL",   PREFERRED_PROVIDER if _is_openai_compat else "openai", "base_url")
KIMI_API_KEY      = _get("KIMI_API_KEY",      "kimi",        "api_key")
KIMI_BASE_URL     = _get("KIMI_BASE_URL",     "kimi",        "base_url", default="https://api.kimi.com/coding/v1")

# ── Model mapping ─────────────────────────────────────────────────────────────

# Defaults differ by provider family
if _is_google:
    _default_big, _default_medium, _default_small = "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash-lite"
elif _is_openai_compat:
    _default_big, _default_medium, _default_small = "llama3.3", "llama3.2", "llama3.2"
elif _is_kimi:
    _default_big, _default_medium, _default_small = "kimi-for-coding", "kimi-for-coding", "kimi-for-coding"
else:
    _default_big, _default_medium, _default_small = "gpt-5.3-codex", "gpt-5.2", "gpt-5-mini"

# Read models from the active provider's section in YAML, env vars override
BIG_MODEL    = _get("BIG_MODEL",    PREFERRED_PROVIDER, "models", "big",    default=_default_big)
MEDIUM_MODEL = _get("MEDIUM_MODEL", PREFERRED_PROVIDER, "models", "medium", default=_default_medium)
SMALL_MODEL  = _get("SMALL_MODEL",  PREFERRED_PROVIDER, "models", "small",  default=_default_small)

# ── Google OAuth credentials ──────────────────────────────────────────────────

GEMINI_OAUTH_CREDS_PATH = Path(
    _get("GEMINI_OAUTH_CREDS_PATH", "google-oauth", "creds_path",
         default="~/.gemini/oauth_creds.json")
).expanduser()
GEMINI_OAUTH_CLIENT_ID     = _get("GEMINI_OAUTH_CLIENT_ID",     "google-oauth", "client_id")
GEMINI_OAUTH_CLIENT_SECRET = _get("GEMINI_OAUTH_CLIENT_SECRET", "google-oauth", "client_secret")

# ── Vertex AI ─────────────────────────────────────────────────────────────────

VERTEX_PROJECT  = _get("VERTEX_PROJECT",  "google-vertex", "project",  default="unset")
VERTEX_LOCATION = _get("VERTEX_LOCATION", "google-vertex", "location", default="unset")

# ── Code Assist endpoint ──────────────────────────────────────────────────────

CODE_ASSIST_ENDPOINT    = _get("CODE_ASSIST_ENDPOINT",    default="https://cloudcode-pa.googleapis.com")
CODE_ASSIST_API_VERSION = _get("CODE_ASSIST_API_VERSION", default="v1internal")

# ── Debug ─────────────────────────────────────────────────────────────────────

DEBUG = _get_bool("DEBUG", "debug")

# ── Per-provider model lookup ─────────────────────────────────────────────────

def get_models_for_provider(provider: str) -> tuple:
    """Return (big, medium, small) model names for any provider, reading from YAML."""
    is_google = provider == "google" or provider.startswith("google-")
    if is_google:
        d_big, d_med, d_small = "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash-lite"
    elif provider == "openai-compat":
        d_big, d_med, d_small = "llama3.3", "llama3.2", "llama3.2"
    elif provider == "kimi":
        d_big, d_med, d_small = "kimi-for-coding", "kimi-for-coding", "kimi-for-coding"
    else:
        d_big, d_med, d_small = "gpt-5.3-codex", "gpt-5.2", "gpt-5-mini"

    section = _yaml.get(provider, {})
    models = section.get("models", {}) if isinstance(section, dict) else {}
    return (
        models.get("big",    d_big),
        models.get("medium", d_med),
        models.get("small",  d_small),
    )


# ── Model lists ───────────────────────────────────────────────────────────────

OPENAI_MODELS = [
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.2-chat",
    "gpt-5.2-pro",
    "gpt-5.2",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-chat",
    "gpt-5.1",
    "gpt-5-codex",
    "gpt-5-chat",
    "gpt-5-pro",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-audio-preview",
    "gpt-4-turbo",
    "o4-mini",
    "o4-mini-high",
    "o3",
    "o3-pro",
    "o3-mini",
    "o3-mini-high",
    "o1",
    "o1-pro",
]

GEMINI_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

KIMI_MODELS = [
    "kimi-for-coding",
    "kimi-k2.5",
    "kimi-k2-0905-preview",
    "kimi-k2-0711-preview",
    "kimi-k2-turbo-preview",
    "kimi-k2-thinking-turbo",
    "kimi-k2-thinking",
    "moonshot-v1-8k",
    "moonshot-v1-32k",
    "moonshot-v1-128k",
    "moonshot-v1-auto",
]
