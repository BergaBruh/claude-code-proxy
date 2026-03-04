import json
import logging
import os
import time
from typing import Dict, Any, Optional

import httpx

from proxy.config import GEMINI_OAUTH_CREDS_PATH

logger = logging.getLogger("proxy")


def _load_oauth_creds() -> Optional[Dict[str, Any]]:
    """Load OAuth credentials from ~/.gemini/oauth_creds.json."""
    if not GEMINI_OAUTH_CREDS_PATH.exists():
        logger.warning(f"OAuth creds file not found: {GEMINI_OAUTH_CREDS_PATH}")
        return None
    try:
        return json.loads(GEMINI_OAUTH_CREDS_PATH.read_text())
    except Exception as e:
        logger.error(f"Failed to load OAuth creds: {e}")
        return None


def _extract_gemini_cli_oauth_creds() -> tuple:
    """Extract OAuth client_id and client_secret from gemini-cli's installed source."""
    import subprocess, re
    try:
        # Find gemini-cli's oauth2.js which contains the public client credentials
        result = subprocess.run(
            ["node", "-e", "console.log(require.resolve('@google/gemini-cli-core/dist/src/code_assist/oauth2.js'))"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            oauth_path = __import__('pathlib').Path(result.stdout.strip())
            if oauth_path.exists():
                content = oauth_path.read_text()
                cid = re.search(r'clientId\s*[:=]\s*["\']([^"\']+)["\']', content)
                csec = re.search(r'clientSecret\s*[:=]\s*["\']([^"\']+)["\']', content)
                if cid and csec:
                    logger.info("Extracted OAuth credentials from gemini-cli")
                    return cid.group(1), csec.group(1)
    except Exception as e:
        logger.debug(f"Could not extract creds from gemini-cli: {e}")
    return None, None


# Cache extracted credentials
_gemini_cli_client_id, _gemini_cli_client_secret = None, None


def _get_gemini_oauth_client_creds() -> tuple:
    """Get OAuth client_id and client_secret from env vars or gemini-cli."""
    global _gemini_cli_client_id, _gemini_cli_client_secret
    client_id = os.environ.get("GEMINI_OAUTH_CLIENT_ID")
    client_secret = os.environ.get("GEMINI_OAUTH_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret
    # Try extracting from gemini-cli once
    if _gemini_cli_client_id is None:
        _gemini_cli_client_id, _gemini_cli_client_secret = _extract_gemini_cli_oauth_creds()
    if _gemini_cli_client_id and _gemini_cli_client_secret:
        return _gemini_cli_client_id, _gemini_cli_client_secret
    return None, None


def _refresh_oauth_token(creds: Dict[str, Any]) -> Optional[str]:
    """Refresh the OAuth access token using the refresh_token. Returns new access_token."""
    refresh_token = creds.get("refresh_token")
    client_id = creds.get("client_id")
    client_secret = creds.get("client_secret")
    if not (client_id and client_secret):
        client_id, client_secret = _get_gemini_oauth_client_creds()
    if not (client_id and client_secret):
        logger.error("OAuth client_id/client_secret not found. Set GEMINI_OAUTH_CLIENT_ID and GEMINI_OAUTH_CLIENT_SECRET env vars, or install gemini-cli (npm i -g @google/gemini-cli)")
        return None
    if not refresh_token:
        logger.error("OAuth creds missing refresh_token — run `gemini` to re-authenticate")
        return None
    try:
        resp = httpx.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        new_access_token = data["access_token"]
        # Update the creds file with new token
        creds["access_token"] = new_access_token
        if "expires_in" in data:
            creds["expiry_date"] = int(time.time() * 1000) + data["expires_in"] * 1000
        GEMINI_OAUTH_CREDS_PATH.write_text(json.dumps(creds, indent=2))
        logger.info("OAuth token refreshed successfully")
        return new_access_token
    except Exception as e:
        logger.error(f"OAuth token refresh failed: {e}")
        return None


def get_gemini_oauth_access_token() -> Optional[str]:
    """Get a valid OAuth access token, refreshing if needed."""
    creds = _load_oauth_creds()
    if not creds:
        return None

    access_token = creds.get("access_token")
    expiry = creds.get("expiry_date")
    now_ms = int(time.time() * 1000)

    # If no expiry info, trust the token as-is (e.g. gemini-cli stores only access_token)
    if access_token and (not expiry or (expiry - now_ms) > 60_000):
        return access_token

    # Token expired — try to refresh
    refreshed = _refresh_oauth_token(creds)
    if refreshed:
        return refreshed
    if access_token:
        logger.warning("OAuth token may be expired and could not be refreshed. Run `gemini` to re-authenticate.")
    return access_token
