"""Integration tests for POST /v1/messages using FastAPI TestClient."""

from unittest.mock import patch, AsyncMock

import pytest

from tests.conftest import HEADERS, make_litellm_response, make_streaming_chunks


class TestMessagesEndpoint:
    """Test the /v1/messages endpoint with mocked litellm."""

    @patch("litellm.completion")
    def test_non_streaming_simple(self, mock_completion, client, sample_messages_body):
        mock_completion.return_value = make_litellm_response(content="Hi there!")
        sample_messages_body["stream"] = False

        resp = client.post("/v1/messages", json=sample_messages_body, headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "assistant"
        assert data["type"] == "message"
        assert len(data["content"]) > 0
        assert data["content"][0]["type"] == "text"

    @patch("litellm.completion")
    def test_non_streaming_stop_reason(self, mock_completion, client, sample_messages_body):
        mock_completion.return_value = make_litellm_response(finish_reason="stop")
        sample_messages_body["stream"] = False

        resp = client.post("/v1/messages", json=sample_messages_body, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["stop_reason"] == "end_turn"

    @patch("litellm.completion")
    def test_non_streaming_usage(self, mock_completion, client, sample_messages_body):
        mock_completion.return_value = make_litellm_response(
            input_tokens=42, output_tokens=17
        )
        sample_messages_body["stream"] = False

        resp = client.post("/v1/messages", json=sample_messages_body, headers=HEADERS)
        data = resp.json()
        assert data["usage"]["input_tokens"] == 42
        assert data["usage"]["output_tokens"] == 17

    @patch("litellm.completion")
    def test_non_streaming_with_system(self, mock_completion, client, sample_messages_body):
        mock_completion.return_value = make_litellm_response()
        sample_messages_body["stream"] = False
        sample_messages_body["system"] = "Be concise"

        resp = client.post("/v1/messages", json=sample_messages_body, headers=HEADERS)
        assert resp.status_code == 200

    @patch("litellm.completion")
    def test_non_streaming_with_tools(self, mock_completion, client, sample_messages_body, sample_tool):
        mock_completion.return_value = make_litellm_response()
        sample_messages_body["stream"] = False
        sample_messages_body["tools"] = [sample_tool]
        sample_messages_body["tool_choice"] = {"type": "auto"}

        resp = client.post("/v1/messages", json=sample_messages_body, headers=HEADERS)
        assert resp.status_code == 200

    @patch("litellm.completion")
    def test_error_returns_http_error(self, mock_completion, client, sample_messages_body):
        mock_completion.side_effect = Exception("API error")
        sample_messages_body["stream"] = False

        resp = client.post("/v1/messages", json=sample_messages_body, headers=HEADERS)
        assert resp.status_code == 500

    @patch("litellm.completion")
    def test_provider_override_via_header(self, mock_completion, client, sample_messages_body):
        mock_completion.return_value = make_litellm_response()
        sample_messages_body["stream"] = False

        headers = {**HEADERS, "authorization": "Bearer openai"}
        resp = client.post("/v1/messages", json=sample_messages_body, headers=headers)
        assert resp.status_code == 200

    @patch("litellm.completion")
    def test_unknown_provider_override_ignored(self, mock_completion, client, sample_messages_body):
        mock_completion.return_value = make_litellm_response()
        sample_messages_body["stream"] = False

        headers = {**HEADERS, "authorization": "Bearer not-a-provider"}
        resp = client.post("/v1/messages", json=sample_messages_body, headers=headers)
        assert resp.status_code == 200


class TestStreamingEndpoint:
    """Test streaming responses from /v1/messages."""

    @patch("litellm.acompletion")
    def test_streaming_returns_sse(self, mock_acompletion, client, sample_messages_body):
        mock_acompletion.return_value = make_streaming_chunks(texts=["Hello"])
        sample_messages_body["stream"] = True

        resp = client.post("/v1/messages", json=sample_messages_body, headers=HEADERS)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Verify SSE events are present
        body = resp.text
        assert "event: message_start" in body
        assert "event: content_block_start" in body
        assert "event: message_stop" in body
