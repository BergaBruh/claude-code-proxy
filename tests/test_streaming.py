"""Unit tests for proxy/streaming.py — SSE streaming handler."""

import json

import pytest

from proxy.models import MessagesRequest
from proxy.streaming import handle_streaming
from tests.conftest import make_streaming_chunks


def _make_request(model="openai/gpt-5.2"):
    return MessagesRequest(
        model=model, max_tokens=100,
        messages=[{"role": "user", "content": "hi"}]
    )


async def _collect_events(gen):
    """Collect all SSE events from the streaming generator."""
    events = []
    async for chunk in gen:
        if chunk.strip():
            for part in chunk.split("\n\n"):
                part = part.strip()
                if not part:
                    continue
                if part.startswith("event: "):
                    lines = part.split("\n")
                    event_type = lines[0].replace("event: ", "")
                    data_line = next((l for l in lines if l.startswith("data: ")), None)
                    if data_line:
                        data_str = data_line.replace("data: ", "")
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            data = data_str
                        events.append({"event": event_type, "data": data})
                elif part.startswith("data: "):
                    data_str = part.replace("data: ", "")
                    events.append({"event": "data", "data": data_str})
    return events


# We need to wrap the sync generator into an async-compatible form
# handle_streaming is an async generator that takes an async iterable

@pytest.mark.asyncio
async def test_streaming_text_only():
    """Verify message_start, content_block_start, text deltas, stop events."""
    request = _make_request()
    chunks = make_streaming_chunks(texts=["Hello", " world"])
    events = await _collect_events(handle_streaming(chunks, request))

    event_types = [e["event"] for e in events]
    assert "message_start" in event_types
    assert "content_block_start" in event_types
    assert "content_block_delta" in event_types
    assert "content_block_stop" in event_types
    assert "message_delta" in event_types
    assert "message_stop" in event_types

    # Verify text deltas contain our text
    text_deltas = [e for e in events if e["event"] == "content_block_delta"
                   and isinstance(e["data"], dict)
                   and e["data"].get("delta", {}).get("type") == "text_delta"]
    full_text = "".join(d["data"]["delta"]["text"] for d in text_deltas)
    assert "Hello" in full_text
    assert "world" in full_text


@pytest.mark.asyncio
async def test_streaming_message_start_structure():
    """Verify message_start contains proper Anthropic format."""
    request = _make_request()
    chunks = make_streaming_chunks(texts=["Hi"])
    events = await _collect_events(handle_streaming(chunks, request))

    msg_start = next(e for e in events if e["event"] == "message_start")
    msg = msg_start["data"]["message"]
    assert msg["role"] == "assistant"
    assert msg["type"] == "message"
    assert "usage" in msg
    assert "id" in msg


@pytest.mark.asyncio
async def test_streaming_finish_reason_stop():
    """Verify stop -> end_turn mapping."""
    request = _make_request()
    chunks = make_streaming_chunks(texts=["Hi"], finish_reason="stop")
    events = await _collect_events(handle_streaming(chunks, request))

    msg_delta = next(e for e in events if e["event"] == "message_delta")
    assert msg_delta["data"]["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_streaming_finish_reason_length():
    """Verify length -> max_tokens mapping."""
    request = _make_request()
    chunks = make_streaming_chunks(texts=["Hi"], finish_reason="length")
    events = await _collect_events(handle_streaming(chunks, request))

    msg_delta = next(e for e in events if e["event"] == "message_delta")
    assert msg_delta["data"]["delta"]["stop_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_streaming_done_marker():
    """Verify [DONE] marker is sent at the end."""
    request = _make_request()
    chunks = make_streaming_chunks(texts=["Hi"])
    events = await _collect_events(handle_streaming(chunks, request))

    last_data_events = [e for e in events if e["event"] == "data"]
    assert any("[DONE]" in str(e["data"]) for e in last_data_events)


@pytest.mark.asyncio
async def test_streaming_ping_event():
    """Verify a ping event is emitted for keep-alive."""
    request = _make_request()
    chunks = make_streaming_chunks(texts=["Hi"])
    events = await _collect_events(handle_streaming(chunks, request))

    assert any(e["event"] == "ping" for e in events)
