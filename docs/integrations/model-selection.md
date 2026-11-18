# Model IDs used in provider examples

OpenAI and Anthropic entries were reviewed November 13, 2026. Pass the full
identifier below to `from_provider()`. Direct SDK calls take the model ID after
the first slash.

| Provider | Instructor model string | Provider reference |
| --- | --- | --- |
| OpenAI | `openai/gpt-6-luna` | [Model](https://developers.openai.com/api/docs/models/gpt-6-luna) |
| Anthropic | `anthropic/claude-haiku-5-5` | [Model](https://platform.claude.com/docs/en/models/haiku-5-5/overview) |
| Google | `google/gemini-3.8-flash` | [Model](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) |
| Cohere | `cohere/command-a-03-2025` | [Models](https://docs.cohere.com/v1/docs/models) |
| Groq | `groq/llama-3.3-70b-versatile` | [Model](https://console.groq.com/docs/model/llama-3.3-70b-versatile) |
| Mistral | `mistral/mistral-small-latest` | [Vision example](https://docs.mistral.ai/studio/conversations/vision) |
| Fireworks | `fireworks/accounts/fireworks/models/kimi-k2p5` | [Model example](https://docs.fireworks.ai/guides/querying-vision-language-models) |
| Cerebras | `cerebras/gpt-oss-120b` | [Models](https://inference-docs.cerebras.ai/api-reference/models/public-models) |
| Writer | `writer/palmyra-x5` | [Models](https://dev.writer.com/home/models) |
| xAI | `xai/grok-4.20-reasoning` | [Model](https://docs.x.ai/developers/models/grok-4.20-reasoning) |
| Perplexity | `perplexity/sonar` | [Model](https://docs.perplexity.ai/docs/sonar/models/sonar) |
| DeepSeek | `deepseek/deepseek-v4-flash` | [Migration](https://api-docs.deepseek.com/updates) |
| OpenRouter | `openrouter/google/gemini-3.8-flash` | [Model](https://openrouter.ai/google/gemini-3.8-flash) |
| Together | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` | [Models](https://docs.together.ai/docs/serverless/models) |
| Bedrock | `bedrock/global.anthropic.claude-haiku-5-5` | [Model and regions](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-5-5.html) |
| Vertex AI | `vertexai/gemini-3.8-flash` | [Model guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash) |

## Request differences

- Luna Chat Completions tool calls need `reasoning_effort="none"`. Set output
  limits with `max_completion_tokens`. Use Responses for reasoning with tools.
- Luna does not accept audio. Transcribe audio with `gpt-transcribe` first.
- Haiku 5.5 skips thinking when the request forces a tool call. To use adaptive
  thinking before tool calls, set `tool_choice={"type": "auto"}`. Omit
  `temperature`, `top_p`, and `top_k`.
- Fireworks requires the full `accounts/fireworks/models/...` name.
- Bedrock and Vertex availability depends on region and account permissions.
  Check the provider reference before using a different region or inference profile.

## Deployment-specific IDs

Use the configured model or route ID for Databricks, Anyscale, Azure OpenAI,
local Ollama, and OpenAI-compatible gateways.
