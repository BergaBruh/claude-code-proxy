import os
from pathlib import Path

# Debug mode — set DEBUG=true to enable verbose logging (litellm debug included)
DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get Vertex AI project and location from environment (if set)
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")

# Option to use Gemini API key instead of ADC for Vertex AI
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# Get OpenAI base URL from environment (if set)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-5.3-codex")      # claude opus  → big
MEDIUM_MODEL = os.environ.get("MEDIUM_MODEL", "gpt-5.2")       # claude sonnet → medium
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-5-mini")      # claude haiku  → small

# Google Code Assist OAuth support
USE_GEMINI_OAUTH = os.environ.get("USE_GEMINI_OAUTH", "False").lower() == "true"
GEMINI_OAUTH_CREDS_PATH = Path(os.environ.get("GEMINI_OAUTH_CREDS_PATH", "~/.gemini/oauth_creds.json")).expanduser()

CODE_ASSIST_ENDPOINT = os.environ.get("CODE_ASSIST_ENDPOINT", "https://cloudcode-pa.googleapis.com")
CODE_ASSIST_API_VERSION = os.environ.get("CODE_ASSIST_API_VERSION", "v1internal")

# List of OpenAI models
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
    "o1-pro"
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
