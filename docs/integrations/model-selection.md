# Model IDs used in provider examples

Reviewed September 5, 2026. These are API identifiers, not display names.
`from_provider()` adds the Instructor provider prefix shown below; direct SDK
calls use the provider's model ID without that prefix.

| Provider | Instructor model string | Provider reference |
| --- | --- | --- |
| OpenAI | `openai/gpt-5.6-luna` | [Model](https://developers.openai.com/api/docs/models/gpt-5.6-luna) |
| Anthropic | `anthropic/claude-sonnet-5` | [Models](https://platform.claude.com/docs/en/models/overview) |
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
| Bedrock | `bedrock/anthropic.claude-sonnet-5` | [Model and regions](https://docs.aws.amazon.com/en_en/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html) |
| Vertex AI | `vertexai/gemini-3.8-flash` | [Model guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash) |

## Request differences

- Luna Chat Completions tool calls need `reasoning_effort="none"`. Set output
  limits with `max_completion_tokens`. Use Responses for reasoning with tools.
- Luna does not accept audio. Transcribe audio with `gpt-4o-transcribe` first.
- Sonnet 5 adaptive thinking uses `thinking={"type": "adaptive"}` and
  `tool_choice={"type": "auto"}`; do not pass an extended-thinking token budget.
- Fireworks requires the full `accounts/fireworks/models/...` name.
- Bedrock and Vertex availability depends on region and account permissions.
  Check the provider reference before using a different region or inference profile.

## Test coverage and older examples

`tests/llm/test_current_doc_models.py` makes real extraction requests. CI reports
missing credentials as skips, not successful model validation. The Google test
suite defaults to Gemini 3.8 Flash and accepts either a bare model ID or a
`google/`-prefixed `GOOGLE_GENAI_MODEL` override.

Historical blog posts, recorded responses, and model-specific regression fixtures
retain their original IDs. Haiku and GPT-4.1 tests remain as compatibility coverage.
Databricks and Anyscale deployments, local Ollama models, and LiteLLM routes are
deployment-specific; their names are not interchangeable with direct-provider IDs.
This update does not certify those deployments or every provider mode.
