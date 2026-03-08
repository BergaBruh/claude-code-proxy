# Anthropic API Proxy for Gemini & OpenAI Models

**Use Anthropic clients (like Claude Code) with Gemini, OpenAI, local LLMs, Kimi or direct Anthropic backends.**

A proxy server that lets you use Anthropic clients with Gemini, OpenAI, OpenAI-compatible APIs (Ollama, LM Studio, vLLM, etc.), Kimi or Anthropic models themselves, all via LiteLLM.

![Anthropic API Proxy](pic.png)

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) installed (for running from source)
- At least one of the following:
  - OpenAI API key
  - OpenAI-compatible API (Ollama, LM Studio, vLLM, llama.cpp, etc.)
  - Google AI Studio (Gemini) API key
  - Google Code Assist via OAuth (free, no API key needed — requires [gemini-cli](https://github.com/google-gemini/gemini-cli))
  - Google Cloud Project with Vertex AI API enabled (for ADC auth)
  - Kimi Code API key (requires [Kimi Code subscription](https://www.kimi.com/code))
  - Anthropic API key (for transparent proxy mode)

### Setup

#### From source

1. **Clone this repository**:

   ```bash
   git clone https://github.com/bergabruh/claude-code-proxy.git
   cd claude-code-proxy
   ```

2. **Install uv** (if you haven't already):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Configure**:

   ```bash
   cp config.example.yaml config.yaml
   ```

   Edit `config.yaml` — see [Configuration](#configuration) below.

4. **Run the server**:

   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```

#### Docker

Download the example config and edit it:

```bash
curl -O https://raw.githubusercontent.com/bergabruh/claude-code-proxy/refs/heads/main/config.example.yaml
mv config.example.yaml config.yaml
```

Then start with [docker compose](https://docs.docker.com/compose/) (preferred):

```yml
services:
  proxy:
    image: ghcr.io/bergabruh/claude-code-proxy:latest
    restart: unless-stopped
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    ports:
      - 8082:8082
```

> Environment variables always override `config.yaml` values, so you can use `env_file` or `environment:` in Docker Compose instead of mounting the config file.
> **Note:** For Google OAuth mode, also mount the credentials file:
>
> ```yml
>     volumes:
>       - ./config.yaml:/app/config.yaml:ro
>       - ~/.gemini/oauth_creds.json:/root/.gemini/oauth_creds.json:ro
> ```

### Using with Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it — your Claude Code client will now use the configured backend models through the proxy.

#### Per-request provider override

If you have multiple providers configured, you can select one per-request by setting `ANTHROPIC_AUTH_TOKEN` to a provider name:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=google-oauth claude  # → Gemini via OAuth
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=openai claude         # → OpenAI
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=openai-compat claude  # → local LLM
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=anthropic claude      # → Anthropic
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=kimi claude      # → Kimi Code
```

This overrides `provider` for that session. If `ANTHROPIC_AUTH_TOKEN` is not set or is not a known provider name, the global `provider` setting is used.

## Configuration

Configuration lives in `config.yaml` (copied from `config.example.yaml`). All values can also be set via environment variables — env vars always take priority.

### Providers

| Provider | Description |
| --- | --- |
| `openai` | OpenAI API |
| `openai-compat` | Any OpenAI-compatible API (Ollama, LM Studio, vLLM, llama.cpp, …) |
| `google-api` | Google AI Studio (API key) |
| `google-oauth` | Google Code Assist via OAuth (free, no billing) |
| `google-vertex` | Google Cloud Vertex AI (ADC) |
| `kimi` | Kimi Console API (API key) |
| `anthropic` | Anthropic API (transparent proxy, no model remapping) |

### Configuration reference

| YAML key | Description |
| --- | --- |
| `provider` | Active provider (see table above) |
| `debug` | Set `true` for verbose logging |
| `<provider>.models.big` | Model for opus requests |
| `<provider>.models.medium` | Model for sonnet requests |
| `<provider>.models.small` | Model for haiku requests |
| `openai.api_key` | OpenAI API key |
| `openai-compat.base_url` | Base URL for `openai-compat` |
| `google-api.api_key` | Google AI Studio API key |
| `google-oauth.creds_path` | Path to OAuth creds (default: `~/.gemini/oauth_creds.json`) |
| `google-oauth.client_id` | OAuth client ID (auto-extracted from gemini-cli if not set) |
| `google-oauth.client_secret` | OAuth client secret (auto-extracted from gemini-cli if not set) |
| `google-vertex.project` | Google Cloud project ID |
| `google-vertex.location` | Google Cloud region (e.g. `us-central1`) |
| `kimi.api_key` | Kimi Code API key |
| `kimi.base_url` | Kimi API base URL (default: `https://api.kimi.com/coding/v1`) |
| `anthropic.api_key` | Anthropic API key |

## Model Mapping

The proxy maps Claude model families to configurable backend models:

| Claude Model | Config key | Default (OpenAI) | Default (Google) | Default (Kimi Code) |
| --- | --- | --- | --- | --- |
| `haiku` | `models.small` | `gpt-5-mini` | `gemini-2.5-flash-lite` | `kimi-for-coding` |
| `sonnet` | `models.medium` | `gpt-5.2` | `gemini-3-flash-preview` | `kimi-for-coding` |
| `opus` | `models.big` | `gpt-5.3-codex` | `gemini-3.1-pro-preview` | `kimi-for-coding` |

Models are configured per-provider in `config.yaml`. `anthropic` provider passes model names through unchanged.

### Supported Models

#### OpenAI

`gpt-5.3-codex`, `gpt-5.2`, `gpt-5.1-codex`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o4-mini`, `o4-mini-high`, `o3`, `o3-pro`, `o3-mini`, `o1`, `o1-pro`

#### Gemini

`gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`

#### Kimi Code

`kimi-for-coding`

#### OpenAI-compatible

Any model name supported by your local API (e.g. `llama3.3`, `qwen3.5`, `mistral`).

### Example Configurations

See [config.example.yaml](config.example.yaml) for all providers:

| Provider | Config section |
| --- | --- |
| OpenAI | [lines 13–19](config.example.yaml#L13) |
| Local LLM (Ollama / LM Studio / vLLM) | [lines 21–28](config.example.yaml#L21) |
| Google AI Studio | [lines 30–36](config.example.yaml#L30) |
| Google Code Assist OAuth (free) | [lines 38–48](config.example.yaml#L38) |
| Google Vertex AI | [lines 50–57](config.example.yaml#L50) |
| Kimi Code | [lines 59–68](config.example.yaml#L59) |
| Anthropic | [lines 70–72](config.example.yaml#L70) |

## How It Works

1. Receives requests in Anthropic Messages API format
2. Translates to OpenAI format via LiteLLM (or directly to Gemini format for OAuth mode)
3. Sends to the configured backend provider
4. Converts the response back to Anthropic format
5. Returns to the client

Supports both streaming and non-streaming responses.

> **Note:** When using OpenAI or Gemini backends, tool calls are returned as text content rather than native `tool_use` blocks. Native tool_use is only supported in Anthropic transparent proxy mode.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
