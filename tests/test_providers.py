"""Unit tests for provider abstraction — configure_request and get_provider."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from proxy.providers import get_provider
from proxy.providers.base import BaseProvider
from proxy.providers.openai import OpenAIProvider, _flatten_content_blocks
from proxy.providers.kimi import KimiProvider
from proxy.providers.anthropic import AnthropicProvider
from proxy.providers.gemini_api import GeminiAPIProvider
from proxy.providers.vertex import VertexProvider


def _mock_request(user_agent="claude-code/1.0.0"):
    req = MagicMock()
    req.headers = {"user-agent": user_agent}
    return req


# ── get_provider registry ────────────────────────────────────────────────

class TestGetProvider:
    def test_kimi_model(self):
        p = get_provider("kimi/kimi-for-coding")
        assert isinstance(p, KimiProvider)

    def test_openai_model(self):
        p = get_provider("openai/gpt-5.2")
        assert isinstance(p, OpenAIProvider)

    def test_anthropic_default(self):
        p = get_provider("anthropic/claude-sonnet-4-6")
        assert isinstance(p, AnthropicProvider)

    @patch("proxy.providers.USE_GEMINI_OAUTH", False)
    @patch("proxy.providers.USE_VERTEX_AUTH", False)
    def test_gemini_api(self):
        p = get_provider("gemini/gemini-2.5-pro")
        assert isinstance(p, GeminiAPIProvider)

    @patch("proxy.providers.USE_GEMINI_OAUTH", False)
    @patch("proxy.providers.USE_VERTEX_AUTH", True)
    def test_gemini_vertex(self):
        p = get_provider("gemini/gemini-2.5-pro")
        assert isinstance(p, VertexProvider)


# ── OpenAIProvider ───────────────────────────────────────────────────────

class TestOpenAIProvider:
    @patch("proxy.providers.openai.OPENAI_API_KEY", "sk-test")
    @patch("proxy.providers.openai.OPENAI_BASE_URL", None)
    def test_sets_api_key(self):
        p = OpenAIProvider()
        req = {"model": "openai/gpt-5.2", "messages": []}
        result = p.configure_request(req, _mock_request())
        assert result["api_key"] == "sk-test"

    @patch("proxy.providers.openai.OPENAI_API_KEY", "sk-test")
    @patch("proxy.providers.openai.OPENAI_BASE_URL", "http://localhost:1234/v1")
    def test_sets_base_url(self):
        p = OpenAIProvider()
        req = {"model": "openai/gpt-5.2", "messages": []}
        result = p.configure_request(req, _mock_request())
        assert result["base_url"] == "http://localhost:1234/v1"


# ── KimiProvider ─────────────────────────────────────────────────────────

class TestKimiProvider:
    @patch("proxy.providers.kimi.KIMI_API_KEY", "sk-kimi-test")
    @patch("proxy.providers.kimi.KIMI_BASE_URL", "https://api.kimi.com/coding/v1")
    def test_rewrites_model_name(self):
        p = KimiProvider()
        req = {"model": "kimi/kimi-for-coding", "messages": []}
        result = p.configure_request(req, _mock_request())
        assert result["model"] == "openai/kimi-for-coding"

    @patch("proxy.providers.kimi.KIMI_API_KEY", "sk-kimi-test")
    @patch("proxy.providers.kimi.KIMI_BASE_URL", "https://api.kimi.com/coding/v1")
    def test_sets_api_key_and_base_url(self):
        p = KimiProvider()
        req = {"model": "kimi/kimi-for-coding", "messages": []}
        result = p.configure_request(req, _mock_request())
        assert result["api_key"] == "sk-kimi-test"
        assert result["base_url"] == "https://api.kimi.com/coding/v1"

    @patch("proxy.providers.kimi.KIMI_API_KEY", "sk-kimi-test")
    @patch("proxy.providers.kimi.KIMI_BASE_URL", "https://api.kimi.com/coding/v1")
    def test_passes_user_agent(self):
        p = KimiProvider()
        req = {"model": "kimi/kimi-for-coding", "messages": []}
        result = p.configure_request(req, _mock_request(user_agent="test-agent/2.0"))
        assert result["extra_headers"]["User-Agent"] == "test-agent/2.0"


# ── AnthropicProvider ────────────────────────────────────────────────────

class TestAnthropicProvider:
    @patch("proxy.providers.anthropic.ANTHROPIC_API_KEY", "sk-ant-test")
    def test_sets_api_key(self):
        p = AnthropicProvider()
        req = {"model": "anthropic/claude-sonnet-4-6", "messages": []}
        result = p.configure_request(req, _mock_request())
        assert result["api_key"] == "sk-ant-test"


# ── GeminiAPIProvider ────────────────────────────────────────────────────

class TestGeminiAPIProvider:
    @patch("proxy.providers.gemini_api.GEMINI_API_KEY", "gemini-key")
    def test_sets_api_key(self):
        p = GeminiAPIProvider()
        req = {"model": "gemini/gemini-2.5-pro", "messages": []}
        result = p.configure_request(req, _mock_request())
        assert result["api_key"] == "gemini-key"


# ── VertexProvider ───────────────────────────────────────────────────────

class TestVertexProvider:
    @patch("proxy.providers.vertex.VERTEX_PROJECT", "my-project")
    @patch("proxy.providers.vertex.VERTEX_LOCATION", "us-central1")
    def test_sets_project_and_location(self):
        p = VertexProvider()
        req = {"model": "gemini/gemini-2.5-pro", "messages": []}
        result = p.configure_request(req, _mock_request())
        assert result["vertex_project"] == "my-project"
        assert result["vertex_location"] == "us-central1"
        assert result["custom_llm_provider"] == "vertex_ai"


# ── Content flattening ───────────────────────────────────────────────────

class TestFlattenContentBlocks:
    def test_text_blocks_to_string(self):
        req = {
            "model": "openai/gpt-5.2",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }],
        }
        _flatten_content_blocks(req)
        assert isinstance(req["messages"][0]["content"], str)
        assert "Hello" in req["messages"][0]["content"]
        assert "World" in req["messages"][0]["content"]

    def test_none_content_replaced(self):
        req = {
            "model": "openai/gpt-5.2",
            "messages": [{"role": "user", "content": None}],
        }
        _flatten_content_blocks(req)
        assert req["messages"][0]["content"] == "..."

    def test_tool_result_flattened(self):
        req = {
            "model": "openai/gpt-5.2",
            "messages": [{
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "result text"}],
            }],
        }
        _flatten_content_blocks(req)
        assert isinstance(req["messages"][0]["content"], str)
        assert "Tool Result" in req["messages"][0]["content"]

    def test_unsupported_fields_removed(self):
        req = {
            "model": "openai/gpt-5.2",
            "messages": [{"role": "user", "content": "hi", "extra_field": True}],
        }
        _flatten_content_blocks(req)
        assert "extra_field" not in req["messages"][0]

    def test_non_openai_model_skipped(self):
        req = {
            "model": "gemini/gemini-2.5-pro",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
        _flatten_content_blocks(req)
        # Content should remain as list for non-openai models
        assert isinstance(req["messages"][0]["content"], list)
