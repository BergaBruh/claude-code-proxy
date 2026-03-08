import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


HEADERS = {
    "x-api-key": "test-key",
    "anthropic-version": "2024-01-01",
    "content-type": "application/json",
}


@pytest.fixture
def client():
    """FastAPI TestClient — no real API keys needed."""
    from proxy import app
    return TestClient(app)


@pytest.fixture
def sample_messages_body():
    """Minimal valid Anthropic /v1/messages request body."""
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
    }


@pytest.fixture
def sample_tool():
    """Sample tool definition."""
    return {
        "name": "calculator",
        "description": "Evaluate math expressions",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"],
        },
    }


def make_litellm_response(content="Hello!", model="openai/gpt-5.2", tool_calls=None,
                           finish_reason="stop", input_tokens=10, output_tokens=5):
    """Build a fake litellm ModelResponse-like object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens)
    return SimpleNamespace(id="chatcmpl-test123", choices=[choice], usage=usage, model=model)


async def make_streaming_chunks(texts=None, finish_reason="stop",
                                 input_tokens=10, output_tokens=5):
    """Async generator yielding fake LiteLLM streaming chunks."""
    texts = texts or ["Hello", " world", "!"]
    for text in texts:
        delta = SimpleNamespace(content=text, tool_calls=None, role=None)
        choice = SimpleNamespace(delta=delta, finish_reason=None, index=0)
        chunk = SimpleNamespace(choices=[choice], usage=None)
        yield chunk

    # Final chunk with finish_reason and usage
    delta = SimpleNamespace(content=None, tool_calls=None, role=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens)
    chunk = SimpleNamespace(choices=[choice], usage=usage)
    yield chunk
