import json
import logging
import copy
import time
import uuid
from typing import List, Dict, Any, Optional

logger = logging.getLogger("proxy")


def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini / Code Assist."""
    if isinstance(schema, dict):
        # Remove all keys unsupported by Gemini tool parameters
        unsupported_keys = {
            "$schema", "additionalProperties", "default",
            "exclusiveMinimum", "exclusiveMaximum",
            "propertyNames", "const", "oneOf",
            "patternProperties", "dependencies", "if", "then", "else",
            "allOf", "not", "minProperties", "maxProperties",
            "minContains", "maxContains", "contentMediaType", "contentEncoding",
            "$id", "$ref", "$comment", "$defs", "definitions",
            "examples", "readOnly", "writeOnly", "deprecated",
            "uniqueItems", "additionalItems",
        }
        for key in unsupported_keys:
            schema.pop(key, None)

        # Convert anyOf to enum where possible (e.g. anyOf with const values)
        if "anyOf" in schema:
            # Check if it's a simple type union like anyOf: [{type: "string"}, {type: "number"}]
            # or const-based enum like anyOf: [{const: "a"}, {const: "b"}]
            any_of = schema["anyOf"]
            if isinstance(any_of, list):
                const_values = [item.get("const") for item in any_of if isinstance(item, dict) and "const" in item]
                if const_values and len(const_values) == len(any_of):
                    # All entries are const — convert to enum
                    schema.pop("anyOf")
                    schema["enum"] = const_values
                    if not schema.get("type"):
                        # Infer type from values
                        if all(isinstance(v, str) for v in const_values):
                            schema["type"] = "string"
                else:
                    # Just drop anyOf — Gemini doesn't support it
                    schema.pop("anyOf")

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()):
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
    return schema


def _openai_messages_to_gemini(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert OpenAI-format messages to Gemini native format (contents + systemInstruction)."""
    system_parts: List[Dict[str, Any]] = []
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # System messages → systemInstruction
            if isinstance(content, str):
                system_parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        system_parts.append({"text": item})
                    elif isinstance(item, dict) and item.get("type") == "text":
                        system_parts.append({"text": item.get("text", "")})
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts: List[Dict[str, Any]] = []

        # Handle tool calls from assistant
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                parts.append({
                    "functionCall": {
                        "name": func.get("name", ""),
                        "args": args,
                    }
                })

        # Handle tool role (function result)
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            response_data = content
            if isinstance(response_data, str):
                try:
                    response_data = json.loads(response_data)
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": response_data}
            parts.append({
                "functionResponse": {
                    "name": msg.get("name", tool_call_id),
                    "response": response_data if isinstance(response_data, dict) else {"result": str(response_data)},
                }
            })
            gemini_role = "user"  # Gemini expects function responses as user role

        # Handle regular content
        if content and role != "tool":
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append({"text": item})
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            url_data = item.get("image_url", {})
                            url = url_data.get("url", "") if isinstance(url_data, dict) else str(url_data)
                            if url.startswith("data:"):
                                # data URI: data:image/png;base64,<data>
                                mime_end = url.index(";")
                                mime_type = url[5:mime_end]
                                b64_data = url[url.index(",") + 1:]
                                parts.append({"inlineData": {"mimeType": mime_type, "data": b64_data}})

        if parts:
            # Gemini requires alternating user/model roles; merge consecutive same-role
            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": gemini_role, "parts": parts})

    result: Dict[str, Any] = {"contents": contents}
    if system_parts:
        result["systemInstruction"] = {"role": "user", "parts": system_parts}
    return result


def _openai_tools_to_gemini(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI-format tool definitions to Gemini format."""
    if not tools:
        return None
    declarations = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        decl: Dict[str, Any] = {"name": func.get("name", ""), "description": func.get("description", "")}
        params = func.get("parameters")
        if params:
            cleaned = clean_gemini_schema(copy.deepcopy(params))
            decl["parameters"] = cleaned
        declarations.append(decl)
    if not declarations:
        return None
    logger.debug(f"Converted {len(declarations)} tools to Gemini format")
    return [{"functionDeclarations": declarations}]


def _gemini_response_to_litellm_format(gemini_resp: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Convert a Code Assist response to a litellm/OpenAI-like object for convert_litellm_to_anthropic."""
    inner = gemini_resp.get("response", gemini_resp)
    candidates = inner.get("candidates", [])
    usage_meta = inner.get("usageMetadata", {})

    choices = []
    for cand in candidates:
        content = cand.get("content", {})
        parts = content.get("parts", [])
        text_parts = []
        tool_calls = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    }
                })

        finish_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "stop",
                      "RECITATION": "stop", "OTHER": "stop"}
        finish_reason = finish_map.get(cand.get("finishReason", "STOP"), "stop")

        message: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) if text_parts else None}
        if tool_calls:
            message["tool_calls"] = tool_calls
            if not message["content"]:
                finish_reason = "tool_calls"

        choices.append({"index": 0, "message": message, "finish_reason": finish_reason})

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": usage_meta.get("promptTokenCount", 0),
            "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
            "total_tokens": usage_meta.get("totalTokenCount", 0),
        }
    }
