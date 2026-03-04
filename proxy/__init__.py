"""Claude Code Proxy — Anthropic API gateway to OpenAI/Gemini/Anthropic backends."""

from fastapi import FastAPI

import litellm

from proxy.config import DEBUG, USE_GEMINI_OAUTH, GEMINI_OAUTH_CREDS_PATH
from proxy.logging_config import logger  # noqa: F401  (initialises logging as side-effect)

app = FastAPI()

# Silently drop params unsupported by the target provider
litellm.drop_params = True

# Register middleware & routes
from proxy.middleware import register_middleware  # noqa: E402
from proxy.routes import register_routes  # noqa: E402

register_middleware(app)
register_routes(app)

# OAuth startup validation
if USE_GEMINI_OAUTH:
    from proxy.auth.oauth import get_gemini_oauth_access_token
    from proxy.auth.code_assist import _get_code_assist_project

    logger.info(f"Gemini OAuth mode enabled, creds path: {GEMINI_OAUTH_CREDS_PATH}")
    _startup_token = get_gemini_oauth_access_token()
    if _startup_token:
        logger.info("Gemini OAuth: startup token validation OK")
        try:
            _get_code_assist_project(_startup_token)
        except Exception as e:
            logger.warning(f"Gemini OAuth: failed to get Code Assist project on startup: {e}")
    else:
        logger.warning("Gemini OAuth: could not get access token on startup — check ~/.gemini/oauth_creds.json")
