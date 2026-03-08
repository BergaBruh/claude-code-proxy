"""Unit tests for proxy/converters.py — format conversion logic."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from proxy.converters import (
    parse_tool_result_content,
    convert_anthropic_to_litellm,
    convert_litellm_to_anthropic,
)
from proxy.models import MessagesRequest, Message, ContentBlockText, ContentBlockToolResult
from tests.conftest import make_litellm_response


# ── parse_tool_result_content ────────────────────────────────────────────

class TestParseToolResultContent:
    def test_none_returns_default(self):
        assert parse_tool_result_content(None) == "No content provided"

    def test_string_passthrough(self):
        assert parse_tool_result_content("hello") == "hello"

    def test_list_of_text_blocks(self):
        content = [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]
        result = parse_tool_result_content(content)
        assert "line1" in result
        assert "line2" in result

    def test_dict_text_block(self):
        assert parse_tool_result_content({"type": "text", "text": "ok"}) == "ok"

    def test_dict_non_text_serialized(self):
        result = parse_tool_result_content({"key": "val"})
        assert "key" in result


# ── convert_anthropic_to_litellm ─────────────────────────────────────────

class TestConvertAnthropicToLitellm:
    def _make_request(self, **kwargs):
        defaults = {
            "model": "openai/gpt-5.2",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        defaults.update(kwargs)
        return MessagesRequest(**defaults)

    def test_simple_text_message(self):
        req = self._make_request()
        result = convert_anthropic_to_litellm(req)
        assert result["model"] == "openai/gpt-5.2"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "hi"

    def test_system_string(self):
        req = self._make_request(system="You are helpful")
        result = convert_anthropic_to_litellm(req)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful"

    def test_system_list(self):
        req = self._make_request(system=[{"type": "text", "text": "Be concise"}])
        result = convert_anthropic_to_litellm(req)
        assert result["messages"][0]["role"] == "system"
        assert "Be concise" in result["messages"][0]["content"]

    def test_max_tokens_capped_for_openai(self):
        req = self._make_request(max_tokens=999999)
        result = convert_anthropic_to_litellm(req)
        assert result["max_completion_tokens"] <= 16384

    def test_max_tokens_capped_for_gemini(self):
        req = self._make_request(model="gemini/gemini-2.5-pro", max_tokens=999999)
        result = convert_anthropic_to_litellm(req)
        assert result["max_completion_tokens"] <= 16384

    def test_tools_converted_to_openai_format(self):
        req = self._make_request(
            tools=[{
                "name": "calc",
                "description": "Calculate",
                "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}},
            }]
        )
        result = convert_anthropic_to_litellm(req)
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "calc"
        assert "parameters" in tool["function"]

    def test_tool_choice_auto(self):
        req = self._make_request(tool_choice={"type": "auto"})
        result = convert_anthropic_to_litellm(req)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        req = self._make_request(tool_choice={"type": "any"})
        result = convert_anthropic_to_litellm(req)
        assert result["tool_choice"] == "any"

    def test_tool_choice_specific(self):
        req = self._make_request(tool_choice={"type": "tool", "name": "calc"})
        result = convert_anthropic_to_litellm(req)
        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "calc"

    def test_stop_sequences(self):
        req = self._make_request(stop_sequences=["END"])
        result = convert_anthropic_to_litellm(req)
        assert result["stop"] == ["END"]

    def test_thinking_included_for_anthropic_prefix(self):
        """When model already has anthropic/ prefix, thinking is included."""
        req = self._make_request(thinking={"enabled": True})
        # Manually set model to anthropic-prefixed (bypassing validator remapping)
        req.model = "anthropic/claude-sonnet-4-6"
        result = convert_anthropic_to_litellm(req)
        assert "thinking" in result

    def test_thinking_omitted_for_openai(self):
        req = self._make_request(thinking={"enabled": True})
        result = convert_anthropic_to_litellm(req)
        assert "thinking" not in result


# ── convert_litellm_to_anthropic ─────────────────────────────────────────

class TestConvertLitellmToAnthropic:
    def _make_request(self, model="openai/gpt-5.2"):
        return MessagesRequest(
            model=model, max_tokens=100,
            messages=[{"role": "user", "content": "hi"}]
        )

    def test_text_response(self):
        resp = make_litellm_response(content="Hello!")
        result = convert_litellm_to_anthropic(resp, self._make_request())
        block = result.content[0]
        assert block.type == "text"
        assert block.text == "Hello!"
        assert result.role == "assistant"
        assert result.type == "message"

    def test_stop_reason_mapping_stop(self):
        resp = make_litellm_response(finish_reason="stop")
        result = convert_litellm_to_anthropic(resp, self._make_request())
        assert result.stop_reason == "end_turn"

    def test_stop_reason_mapping_length(self):
        resp = make_litellm_response(finish_reason="length")
        result = convert_litellm_to_anthropic(resp, self._make_request())
        assert result.stop_reason == "max_tokens"

    def test_stop_reason_mapping_tool_calls(self):
        resp = make_litellm_response(finish_reason="tool_calls")
        result = convert_litellm_to_anthropic(resp, self._make_request())
        assert result.stop_reason == "tool_use"

    def test_usage_extracted(self):
        resp = make_litellm_response(input_tokens=42, output_tokens=17)
        result = convert_litellm_to_anthropic(resp, self._make_request())
        assert result.usage.input_tokens == 42
        assert result.usage.output_tokens == 17

    def test_tool_calls_for_non_claude_become_text(self):
        tc = SimpleNamespace(
            id="tool_1",
            function=SimpleNamespace(name="calc", arguments='{"x": 1}'),
        )
        resp = make_litellm_response(content="Let me calculate", tool_calls=[tc])
        result = convert_litellm_to_anthropic(resp, self._make_request())
        # Non-claude model: tool calls should appear as text
        block = result.content[0]
        assert block.type == "text"
        assert "Tool" in block.text or "calc" in block.text

    def test_empty_content_gets_fallback(self):
        resp = make_litellm_response(content=None)
        result = convert_litellm_to_anthropic(resp, self._make_request())
        assert len(result.content) > 0

    def test_dict_response_format(self):
        resp = {
            "id": "test",
            "choices": [{"message": {"content": "hi", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        result = convert_litellm_to_anthropic(resp, self._make_request())
        assert result.content[0].text == "hi"
