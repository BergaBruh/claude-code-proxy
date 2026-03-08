"""Unit tests for proxy/model_mapping.py — pure functions, no I/O."""

from unittest.mock import patch

from proxy.model_mapping import map_model_for_provider


class TestMapModelForProvider:
    """Tests for map_model_for_provider()."""

    # ── Haiku (small) mapping ────────────────────────────────────────────

    def test_haiku_maps_to_openai_small(self):
        result = map_model_for_provider("claude-3-haiku-20240307", "openai")
        assert result.startswith("openai/")

    def test_haiku_maps_to_gemini_small(self):
        result = map_model_for_provider("claude-3-haiku-20240307", "google-api")
        assert result.startswith("gemini/")

    def test_haiku_maps_to_kimi_small(self):
        result = map_model_for_provider("claude-3-haiku-20240307", "kimi")
        assert result.startswith("kimi/")

    # ── Sonnet (medium) mapping ──────────────────────────────────────────

    def test_sonnet_maps_to_openai_medium(self):
        result = map_model_for_provider("claude-sonnet-4-6", "openai")
        assert result.startswith("openai/")

    def test_sonnet_maps_to_gemini_medium(self):
        result = map_model_for_provider("claude-sonnet-4-6", "google-oauth")
        assert result.startswith("gemini/")

    def test_sonnet_maps_to_kimi_medium(self):
        result = map_model_for_provider("claude-sonnet-4-6", "kimi")
        assert result.startswith("kimi/")

    # ── Opus (big) mapping ───────────────────────────────────────────────

    def test_opus_maps_to_openai_big(self):
        result = map_model_for_provider("claude-opus-4-6", "openai")
        assert result.startswith("openai/")

    def test_opus_maps_to_gemini_big(self):
        result = map_model_for_provider("claude-opus-4-6", "google-vertex")
        assert result.startswith("gemini/")

    def test_opus_maps_to_kimi_big(self):
        result = map_model_for_provider("claude-opus-4-6", "kimi")
        assert result.startswith("kimi/")

    # ── Anthropic passthrough ────────────────────────────────────────────

    def test_anthropic_provider_passthrough(self):
        result = map_model_for_provider("claude-sonnet-4-6", "anthropic")
        assert result == "anthropic/claude-sonnet-4-6"

    def test_anthropic_preserves_full_model_name(self):
        result = map_model_for_provider("claude-3-opus-20240229", "anthropic")
        assert result == "anthropic/claude-3-opus-20240229"

    # ── Prefix stripping ─────────────────────────────────────────────────

    def test_strips_anthropic_prefix_before_mapping(self):
        result = map_model_for_provider("anthropic/claude-sonnet-4-6", "openai")
        assert result.startswith("openai/")
        assert "anthropic" not in result

    def test_strips_openai_prefix_before_mapping(self):
        result = map_model_for_provider("openai/claude-sonnet-4-6", "google-api")
        assert result.startswith("gemini/")

    # ── Known model detection ────────────────────────────────────────────

    def test_known_gemini_model_gets_prefix(self):
        result = map_model_for_provider("gemini-2.5-pro", "openai")
        assert result == "gemini/gemini-2.5-pro"

    def test_known_openai_model_gets_prefix(self):
        result = map_model_for_provider("gpt-4o", "google-api")
        assert result == "openai/gpt-4o"

    def test_known_kimi_model_gets_prefix(self):
        result = map_model_for_provider("kimi-for-coding", "openai")
        assert result == "kimi/kimi-for-coding"

    # ── Unknown / fallback ───────────────────────────────────────────────

    def test_unknown_model_returned_as_is(self):
        result = map_model_for_provider("some-random-model", "openai")
        assert result == "some-random-model"

    def test_already_prefixed_gemini_not_doubled(self):
        result = map_model_for_provider("gemini/gemini-2.5-pro", "google-api")
        assert result == "gemini/gemini-2.5-pro"

    # ── Google provider variants ─────────────────────────────────────────

    def test_google_api_is_google(self):
        result = map_model_for_provider("claude-sonnet-4-6", "google-api")
        assert result.startswith("gemini/")

    def test_google_oauth_is_google(self):
        result = map_model_for_provider("claude-sonnet-4-6", "google-oauth")
        assert result.startswith("gemini/")

    def test_google_vertex_is_google(self):
        result = map_model_for_provider("claude-sonnet-4-6", "google-vertex")
        assert result.startswith("gemini/")
