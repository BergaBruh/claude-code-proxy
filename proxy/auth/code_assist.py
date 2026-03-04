import logging
from typing import Optional

import httpx

from proxy.config import CODE_ASSIST_ENDPOINT, CODE_ASSIST_API_VERSION

logger = logging.getLogger("proxy")

_code_assist_project: Optional[str] = None


def _get_code_assist_project(token: str) -> str:
    """Get (and cache) the Code Assist project ID via loadCodeAssist."""
    global _code_assist_project
    if _code_assist_project:
        return _code_assist_project
    resp = httpx.post(
        f"{CODE_ASSIST_ENDPOINT}/{CODE_ASSIST_API_VERSION}:loadCodeAssist",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"metadata": {"ideType": "IDE_UNSPECIFIED", "platform": "PLATFORM_UNSPECIFIED", "pluginType": "GEMINI"}},
        timeout=15,
    )
    resp.raise_for_status()
    project = resp.json().get("cloudaicompanionProject")
    if not project:
        raise RuntimeError("Code Assist did not return a project ID. Ensure your Google account is set up for Gemini Code Assist.")
    _code_assist_project = project
    logger.info(f"Code Assist project ID: {project}")
    return project
