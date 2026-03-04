import json
import logging
import uuid
from types import SimpleNamespace
from typing import Dict, Any

import httpx

from proxy.config import CODE_ASSIST_ENDPOINT, CODE_ASSIST_API_VERSION

logger = logging.getLogger("proxy")


async def _code_assist_stream(token: str, project: str, model: str, gemini_request: Dict[str, Any]):
    """Call Code Assist streamGenerateContent and yield SSE chunks as litellm-style delta dicts."""
    url = f"{CODE_ASSIST_ENDPOINT}/{CODE_ASSIST_API_VERSION}:streamGenerateContent"
    body = {"model": model, "project": project, "request": gemini_request}

    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=15)) as client:
        async with client.stream(
            "POST", url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            content=json.dumps(body),
            params={"alt": "sse"},
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                raise RuntimeError(f"Code Assist stream error {resp.status_code}: {error_body.decode()}")
            buffer = ""
            async for raw_chunk in resp.aiter_text():
                buffer += raw_chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        yield data


async def _code_assist_stream_as_openai(token: str, project: str, model: str, gemini_request: Dict[str, Any]):
    """Wrap Code Assist streaming into OpenAI-compatible chunk objects for handle_streaming."""
    async for data in _code_assist_stream(token, project, model, gemini_request):
        inner = data.get("response", data)
        candidates = inner.get("candidates", [])
        usage_meta = inner.get("usageMetadata")

        for cand in candidates:
            content = cand.get("content", {})
            parts = content.get("parts", [])
            finish_reason_raw = cand.get("finishReason")

            for part in parts:
                delta = SimpleNamespace()
                delta.content = part.get("text")
                delta.tool_calls = None
                delta.role = None

                if "functionCall" in part:
                    fc = part["functionCall"]
                    tc = SimpleNamespace()
                    tc.index = 0
                    tc.id = f"call_{uuid.uuid4().hex[:8]}"
                    tc.type = "function"
                    tc.function = SimpleNamespace()
                    tc.function.name = fc.get("name", "")
                    tc.function.arguments = json.dumps(fc.get("args", {}))
                    delta.tool_calls = [tc]
                    delta.content = None

                finish_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "stop", "RECITATION": "stop"}
                choice = SimpleNamespace()
                choice.delta = delta
                choice.finish_reason = finish_map.get(finish_reason_raw) if finish_reason_raw else None
                choice.index = 0

                chunk = SimpleNamespace()
                chunk.choices = [choice]
                chunk.usage = None
                if usage_meta:
                    chunk.usage = SimpleNamespace()
                    chunk.usage.prompt_tokens = usage_meta.get("promptTokenCount", 0)
                    chunk.usage.completion_tokens = usage_meta.get("candidatesTokenCount", 0)
                    chunk.usage.total_tokens = usage_meta.get("totalTokenCount", 0)

                yield chunk


async def _code_assist_generate(token: str, project: str, model: str, gemini_request: Dict[str, Any]) -> Dict[str, Any]:
    """Call Code Assist generateContent (non-streaming)."""
    url = f"{CODE_ASSIST_ENDPOINT}/{CODE_ASSIST_API_VERSION}:generateContent"
    body = {"model": model, "project": project, "request": gemini_request}

    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=15)) as client:
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            content=json.dumps(body),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Code Assist error {resp.status_code}: {resp.text}")
        return resp.json()
