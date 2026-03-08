"""Integration tests for POST /v1/messages/count_tokens."""

from unittest.mock import patch

from tests.conftest import HEADERS


class TestTokenCountEndpoint:
    @patch("litellm.token_counter", return_value=42)
    def test_basic_token_count(self, mock_counter, client):
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello world"}],
        }
        resp = client.post("/v1/messages/count_tokens", json=body, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 42

    @patch("litellm.token_counter", return_value=100)
    def test_token_count_with_system(self, mock_counter, client):
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "Be helpful",
        }
        resp = client.post("/v1/messages/count_tokens", json=body, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 100

    @patch("litellm.token_counter", return_value=150)
    def test_token_count_with_tools(self, mock_counter, client, sample_tool):
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Calculate 2+2"}],
            "tools": [sample_tool],
        }
        resp = client.post("/v1/messages/count_tokens", json=body, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 150

    @patch("litellm.token_counter", return_value=50)
    def test_token_count_provider_override(self, mock_counter, client):
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        headers = {**HEADERS, "authorization": "Bearer openai"}
        resp = client.post("/v1/messages/count_tokens", json=body, headers=headers)
        assert resp.status_code == 200
