"""Unit tests for proxy/providers/gemini_format.py — schema cleaning and format conversion."""

import copy
import json

from proxy.providers.gemini_format import (
    clean_gemini_schema,
    _openai_messages_to_gemini,
    _openai_tools_to_gemini,
    _gemini_response_to_litellm_format,
)


# ── clean_gemini_schema ──────────────────────────────────────────────────

class TestCleanGeminiSchema:
    def test_removes_unsupported_keys(self):
        schema = {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "additionalProperties": False,
            "default": "foo",
            "properties": {"x": {"type": "string"}},
        }
        result = clean_gemini_schema(schema)
        assert "$schema" not in result
        assert "additionalProperties" not in result
        assert "default" not in result
        assert result["properties"]["x"]["type"] == "string"

    def test_converts_anyof_const_to_enum(self):
        schema = {
            "anyOf": [{"const": "a"}, {"const": "b"}, {"const": "c"}]
        }
        result = clean_gemini_schema(schema)
        assert "anyOf" not in result
        assert result["enum"] == ["a", "b", "c"]

    def test_drops_anyof_without_const(self):
        schema = {
            "anyOf": [{"type": "string"}, {"type": "number"}]
        }
        result = clean_gemini_schema(schema)
        assert "anyOf" not in result

    def test_removes_unsupported_string_format(self):
        schema = {"type": "string", "format": "uri"}
        result = clean_gemini_schema(schema)
        assert "format" not in result

    def test_keeps_supported_string_format(self):
        schema = {"type": "string", "format": "date-time"}
        result = clean_gemini_schema(schema)
        assert result["format"] == "date-time"

    def test_recursive_cleaning(self):
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "$schema": "remove-me",
                    "additionalProperties": True,
                    "properties": {"x": {"type": "string", "default": "y"}},
                }
            },
        }
        result = clean_gemini_schema(schema)
        nested = result["properties"]["nested"]
        assert "$schema" not in nested
        assert "additionalProperties" not in nested
        assert "default" not in nested["properties"]["x"]

    def test_list_items_cleaned(self):
        schema = {
            "type": "array",
            "items": {"type": "string", "$schema": "remove"},
        }
        result = clean_gemini_schema(schema)
        assert "$schema" not in result["items"]


# ── _openai_messages_to_gemini ───────────────────────────────────────────

class TestOpenaiMessagesToGemini:
    def test_system_becomes_system_instruction(self):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = _openai_messages_to_gemini(messages)
        assert "systemInstruction" in result
        assert result["systemInstruction"]["parts"][0]["text"] == "Be helpful"
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"

    def test_user_and_assistant_roles(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = _openai_messages_to_gemini(messages)
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][1]["role"] == "model"

    def test_consecutive_same_role_merged(self):
        messages = [
            {"role": "user", "content": "Part 1"},
            {"role": "user", "content": "Part 2"},
        ]
        result = _openai_messages_to_gemini(messages)
        assert len(result["contents"]) == 1
        assert len(result["contents"][0]["parts"]) == 2

    def test_tool_calls_become_function_call(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "function": {"name": "calc", "arguments": '{"x": 1}'},
                }],
            },
        ]
        result = _openai_messages_to_gemini(messages)
        parts = result["contents"][0]["parts"]
        assert "functionCall" in parts[0]
        assert parts[0]["functionCall"]["name"] == "calc"

    def test_tool_role_becomes_function_response(self):
        messages = [
            {"role": "tool", "content": '{"result": 42}', "tool_call_id": "call_1", "name": "calc"},
        ]
        result = _openai_messages_to_gemini(messages)
        parts = result["contents"][0]["parts"]
        assert "functionResponse" in parts[0]
        assert parts[0]["functionResponse"]["name"] == "calc"


# ── _openai_tools_to_gemini ──────────────────────────────────────────────

class TestOpenaiToolsToGemini:
    def test_none_returns_none(self):
        assert _openai_tools_to_gemini(None) is None

    def test_empty_returns_none(self):
        assert _openai_tools_to_gemini([]) is None

    def test_converts_function_tools(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Calculate",
                "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
            },
        }]
        result = _openai_tools_to_gemini(tools)
        assert result is not None
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "calc"

    def test_non_function_tools_skipped(self):
        tools = [{"type": "retrieval"}]
        assert _openai_tools_to_gemini(tools) is None


# ── _gemini_response_to_litellm_format ───────────────────────────────────

class TestGeminiResponseToLitellm:
    def test_text_response(self):
        gemini_resp = {
            "response": {
                "candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
            }
        }
        result = _gemini_response_to_litellm_format(gemini_resp, "gemini-2.5-pro")
        assert result["choices"][0]["message"]["content"] == "Hello"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 5

    def test_function_call_response(self):
        gemini_resp = {
            "response": {
                "candidates": [{
                    "content": {"parts": [{"functionCall": {"name": "calc", "args": {"x": 1}}}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
            }
        }
        result = _gemini_response_to_litellm_format(gemini_resp, "gemini-2.5-pro")
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "calc"
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_max_tokens_finish_reason(self):
        gemini_resp = {
            "response": {
                "candidates": [{"content": {"parts": [{"text": "cut off"}]}, "finishReason": "MAX_TOKENS"}],
                "usageMetadata": {},
            }
        }
        result = _gemini_response_to_litellm_format(gemini_resp, "gemini-2.5-pro")
        assert result["choices"][0]["finish_reason"] == "length"
