import logging
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal

from proxy.config import (
    PREFERRED_PROVIDER, BIG_MODEL, MEDIUM_MODEL, SMALL_MODEL,
    OPENAI_MODELS, GEMINI_MODELS, get_models_for_provider,
)

logger = logging.getLogger("proxy")


def map_model_for_provider(model_name: str, provider: str) -> str:
    """Map an Anthropic model name to the correct backend model for *provider*.

    This is the core mapping logic, usable both from Pydantic validators and
    from route handlers (for per-request provider overrides).
    """
    new_model = model_name

    # Remove provider prefixes for easier matching
    clean_v = model_name
    if clean_v.startswith('anthropic/'):
        clean_v = clean_v[10:]
    elif clean_v.startswith('openai/'):
        clean_v = clean_v[7:]
    elif clean_v.startswith('gemini/'):
        clean_v = clean_v[7:]

    is_google = provider == "google" or provider.startswith("google-")
    p_big, p_medium, p_small = get_models_for_provider(provider)

    mapped = False
    if provider == "anthropic":
        new_model = f"anthropic/{clean_v}"
        mapped = True
    elif 'haiku' in clean_v.lower():
        if is_google:
            new_model = f"gemini/{p_small}"
        else:
            new_model = f"openai/{p_small}"
        mapped = True
    elif 'sonnet' in clean_v.lower():
        if is_google:
            new_model = f"gemini/{p_medium}"
        else:
            new_model = f"openai/{p_medium}"
        mapped = True
    elif 'opus' in clean_v.lower():
        if is_google:
            new_model = f"gemini/{p_big}"
        else:
            new_model = f"openai/{p_big}"
        mapped = True
    else:
        if clean_v in GEMINI_MODELS and not model_name.startswith('gemini/'):
            new_model = f"gemini/{clean_v}"
            mapped = True
        elif clean_v in OPENAI_MODELS and not model_name.startswith('openai/'):
            new_model = f"openai/{clean_v}"
            mapped = True

    if not mapped:
        new_model = model_name

    return new_model


def _map_model(v, info, *, label="MODEL"):
    """Shared model-mapping logic used by MessagesRequest and TokenCountRequest validators."""
    original_model = v

    logger.debug(
        f"📋 {label} VALIDATION: Original='{original_model}', "
        f"Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', "
        f"MEDIUM='{MEDIUM_MODEL}', SMALL='{SMALL_MODEL}'"
    )

    new_model = map_model_for_provider(v, PREFERRED_PROVIDER)

    if new_model != v:
        logger.debug(f"📌 {label} MAPPING: '{original_model}' ➡️ '{new_model}'")
    else:
        if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
            logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")

    # Store the original model in the values dictionary
    values = info.data
    if isinstance(values, dict):
        values['original_model'] = original_model

    return new_model


# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = True

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model_field(cls, v, info):
        return _map_model(v, info, label="MODEL")

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model_token_count(cls, v, info):
        return _map_model(v, info, label="TOKEN COUNT")

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage
