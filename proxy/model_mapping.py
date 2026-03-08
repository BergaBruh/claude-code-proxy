import logging

from proxy.config import (
    PREFERRED_PROVIDER, BIG_MODEL, MEDIUM_MODEL, SMALL_MODEL,
    OPENAI_MODELS, GEMINI_MODELS, KIMI_MODELS, get_models_for_provider,
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
    elif clean_v.startswith('kimi/'):
        clean_v = clean_v[5:]

    is_google = provider == "google" or provider.startswith("google-")
    is_kimi = provider == "kimi"
    p_big, p_medium, p_small = get_models_for_provider(provider)

    mapped = False
    if provider == "anthropic":
        new_model = f"anthropic/{clean_v}"
        mapped = True
    elif 'haiku' in clean_v.lower():
        if is_google:
            new_model = f"gemini/{p_small}"
        elif is_kimi:
            new_model = f"kimi/{p_small}"
        else:
            new_model = f"openai/{p_small}"
        mapped = True
    elif 'sonnet' in clean_v.lower():
        if is_google:
            new_model = f"gemini/{p_medium}"
        elif is_kimi:
            new_model = f"kimi/{p_medium}"
        else:
            new_model = f"openai/{p_medium}"
        mapped = True
    elif 'opus' in clean_v.lower():
        if is_google:
            new_model = f"gemini/{p_big}"
        elif is_kimi:
            new_model = f"kimi/{p_big}"
        else:
            new_model = f"openai/{p_big}"
        mapped = True
    else:
        if clean_v in GEMINI_MODELS and not model_name.startswith('gemini/'):
            new_model = f"gemini/{clean_v}"
            mapped = True
        elif clean_v in KIMI_MODELS and not model_name.startswith('kimi/'):
            new_model = f"kimi/{clean_v}"
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
        if not v.startswith(('openai/', 'gemini/', 'anthropic/', 'kimi/')):
            logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")

    # Store the original model in the values dictionary
    values = info.data
    if isinstance(values, dict):
        values['original_model'] = original_model

    return new_model
