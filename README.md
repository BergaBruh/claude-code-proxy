# Anthropic API Proxy for Gemini & OpenAI Models

**Use Anthropic clients (like Claude Code) with Gemini, OpenAI, or direct Anthropic backends.**

A proxy server that lets you use Anthropic clients with Gemini, OpenAI, or Anthropic models themselves (a transparent proxy of sorts), all via LiteLLM.

![Anthropic API Proxy](pic.png)

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) installed (for running from source)
- At least one of the following:
  - OpenAI API key
  - Google AI Studio (Gemini) API key
  - Google Code Assist via OAuth (free, no API key needed — requires [gemini-cli](https://github.com/google-gemini/gemini-cli))
  - Google Cloud Project with Vertex AI API enabled (for ADC auth)
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

3. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` — see [Configuration](#configuration) below.

4. **Run the server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```

#### Docker

Download the example environment file and edit it:
```bash
curl -O .env https://raw.githubusercontent.com/bergabruh/claude-code-proxy/refs/heads/main/.env.example
```

Then start with [docker compose](https://docs.docker.com/compose/) (preferred):

```yml
services:
  proxy:
    image: ghcr.io/BergaBruh/claude-code-proxy:latest
    restart: unless-stopped
    env_file: .env
    ports:
      - 8082:8082
```

Or with a command:

```bash
docker run -d --env-file .env -p 8082:8082 ghcr.io/bergabruh/claude-code-proxy:latest
```

> **Note:** For Google OAuth mode, mount the credentials file into the container:
> ```bash
> docker run -d --env-file .env -p 8082:8082 \
>   -v ~/.gemini/oauth_creds.json:/root/.gemini/oauth_creds.json:ro \
>   ghcr.io/BergaBruh/claude-code-proxy:latest
> ```

### Using with Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it — your Claude Code client will now use the configured backend models through the proxy.

#### Per-request provider override

If you have multiple providers configured, you can select one per-request by setting `ANTHROPIC_AUTH_TOKEN` to a provider name:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=google claude   # → Gemini
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=openai claude   # → OpenAI
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=anthropic claude # → Anthropic
```

This overrides `PREFERRED_PROVIDER` for that session. If `ANTHROPIC_AUTH_TOKEN` is not set or is not a known provider name, the global `PREFERRED_PROVIDER` is used.

## Configuration

| Variable | Description | Required |
|---|---|---|
| `PREFERRED_PROVIDER` | `openai` (default), `google`, or `anthropic` | No |
| `BIG_MODEL` | Model for opus requests | No |
| `MEDIUM_MODEL` | Model for sonnet requests | No |
| `SMALL_MODEL` | Model for haiku requests | No |
| `OPENAI_API_KEY` | OpenAI API key | If using OpenAI |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible base URL | No |
| `GEMINI_API_KEY` | Google AI Studio API key | If using Google without OAuth/Vertex |
| `ANTHROPIC_API_KEY` | Anthropic API key | If using Anthropic |
| `USE_GEMINI_OAUTH` | Set `true` for Google Code Assist OAuth | No |
| `GEMINI_OAUTH_CREDS_PATH` | Path to OAuth creds JSON (default: `~/.gemini/oauth_creds.json`) | No |
| `USE_VERTEX_AUTH` | Set `true` for Vertex AI ADC auth | No |
| `VERTEX_PROJECT` | Google Cloud project ID | If using Vertex |
| `VERTEX_LOCATION` | Google Cloud region (e.g. `us-central1`) | If using Vertex |
| `DEBUG` | Set `true` for verbose logging | No |

## Model Mapping

The proxy maps Claude model families to configurable backend models:

| Claude Model | Env Var | Default (OpenAI) | Default (Google) |
|---|---|---|---|
| `*haiku*` | `SMALL_MODEL` | `gpt-5-mini` | `gemini-2.5-flash-lite` |
| `*sonnet*` | `MEDIUM_MODEL` | `gpt-5.2` | `gemini-3-flash-preview` |
| `*opus*` | `BIG_MODEL` | `gpt-5.3-codex` | `gemini-3.1-pro-preview` |

### Supported Models

#### OpenAI
`gpt-5.3-codex`, `gpt-5.2`, `gpt-5.1-codex`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o4-mini`, `o4-mini-high`, `o3`, `o3-pro`, `o3-mini`, `o1`, `o1-pro`

#### Gemini
`gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`

### Example Configurations

**OpenAI (default)**
```dotenv
OPENAI_API_KEY="sk-..."
# PREFERRED_PROVIDER="openai"       # default
# BIG_MODEL="gpt-5.3-codex"         # default
# MEDIUM_MODEL="gpt-5.2"            # default
# SMALL_MODEL="gpt-5-mini"          # default
```

**Google with API key**
```dotenv
GEMINI_API_KEY="your-key"
PREFERRED_PROVIDER="google"
BIG_MODEL="gemini-3.1-pro-preview"
MEDIUM_MODEL="gemini-3-flash-preview"
SMALL_MODEL="gemini-2.5-flash-lite"
```

**Google with OAuth (free, no API key)**
```dotenv
PREFERRED_PROVIDER="google"
USE_GEMINI_OAUTH=true
BIG_MODEL="gemini-3.1-pro-preview"
MEDIUM_MODEL="gemini-3-flash-preview"
SMALL_MODEL="gemini-2.5-flash-lite"
```
> Requires [gemini-cli](https://github.com/google-gemini/gemini-cli): `npm i -g @google/gemini-cli && gemini` (run once to authenticate). Uses the Google Code Assist endpoint — no API key or GCP billing needed. OAuth tokens are refreshed automatically — the proxy extracts `client_id`/`client_secret` from gemini-cli (works with both user and sudo installs).

**Google with Vertex AI (ADC)**
```dotenv
PREFERRED_PROVIDER="google"
USE_VERTEX_AUTH=true
VERTEX_PROJECT="your-gcp-project-id"
VERTEX_LOCATION="us-central1"
BIG_MODEL="gemini-3.1-pro-preview"
MEDIUM_MODEL="gemini-3-flash-preview"
SMALL_MODEL="gemini-2.5-flash-lite"
```

**Anthropic (transparent proxy)**
```dotenv
ANTHROPIC_API_KEY="sk-ant-..."
PREFERRED_PROVIDER="anthropic"
# BIG_MODEL/MEDIUM_MODEL/SMALL_MODEL are ignored in this mode
```

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