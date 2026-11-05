# OpenAI-compatible provider clients

Use `instructor.from_provider("provider/model")` with a key issued for that provider. Model IDs can contain additional slashes. The SDK sends each request to the selected endpoint.

| Provider | Endpoint | Environment key | Default mode |
| --- | --- | --- | --- |
| [ZhipuAI](https://docs.bigmodel.cn/cn/best-practice/case/ai-search-engine) (`zhipuai`) | `https://open.bigmodel.cn/api/paas/v4/` | `ZHIPUAI_API_KEY`, `BIGMODEL_API_KEY` | `TOOLS` |
| [Z.ai](https://docs.z.ai/guides/overview/quick-start) (`zai`) | `https://api.z.ai/api/paas/v4/` | `ZAI_API_KEY` | `TOOLS` |
| [Hubris](https://hubris.pw/docs/api/chat-completions) (`hubris`) | `https://api.hubris.pw/v1` | `HUBRIS_API_KEY` | `TOOLS` |
| [Atlas Cloud](https://www.atlascloud.ai/docs/llm-protocols) (`atlascloud`) | `https://api.atlascloud.ai/v1` | `ATLASCLOUD_API_KEY` | `TOOLS` |
| [Requesty](https://docs.requesty.ai/) (`requesty`) | `https://router.requesty.ai/v1` | `REQUESTY_API_KEY` | `TOOLS` |
| [Sarvam](https://docs.sarvam.ai/api/token-factory) (`sarvam`) | `https://api.sarvam.ai/v2` | `SARVAM_API_KEY` | `MD_JSON` |
| [OrcaRouter](https://docs.orcarouter.ai/) (`orcarouter`) | `https://api.orcarouter.ai/v1` | `ORCAROUTER_API_KEY` | `TOOLS` |
| [MiniMax](https://platform.minimax.io/docs/api-reference/text-openai-api) (`minimax`) | `https://api.minimax.io/v1` | `MINIMAX_API_KEY` | `TOOLS` |
| [Manifest](https://manifest.build/llm-gateway/docs/introduction/) (`manifest`) | `https://app.manifest.build/v1` | `MANIFEST_API_KEY` | `TOOLS` |

Set `INSTRUCTOR_MODEL` to your provider and model, for example `manifest/auto` or `requesty/openai/gpt-4o-mini`. Set the matching credential variable from the table.

```python
import os

import instructor
from pydantic import BaseModel


class Contact(BaseModel):
    name: str


client = instructor.from_provider(os.environ["INSTRUCTOR_MODEL"])
contact = client.chat.completions.create(
    response_model=Contact,
    messages=[{"role": "user", "content": "My name is Ada."}],
)
print(contact.name)
```

Pass `async_client=True` and await `client.chat.completions.create(...)` for asynchronous calls. Each provider also exposes `instructor.from_<provider>` for an existing `OpenAI` or `AsyncOpenAI` client. Use `extra_body` for router-specific request options.

## Modes

The available modes depend on the route and its upstream model. `MD_JSON` validates JSON returned in the message. `TOOLS` requires tool calling. `JSON_SCHEMA` requires native structured outputs. `PARALLEL_TOOLS` requires multiple tool calls in a single response.

| Provider | Registered modes |
| --- | --- |
| `zhipuai` | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` |
| `zai` | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` |
| `hubris` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` |
| `atlascloud` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` |
| `requesty` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` |
| `sarvam` | `TOOLS`, `JSON`, `JSON_SCHEMA`, `MD_JSON` |
| `orcarouter` | `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` |
| `minimax` | `TOOLS`, `MD_JSON` |
| `manifest` | `TOOLS`, `JSON`, `MD_JSON` |

Z.ai and ZhipuAI have separate regional endpoints. Use the key for the selected region, or supply its key explicitly with `api_key=`. Sarvam uses its v2 Token Factory endpoint and defaults to `MD_JSON`. MiniMax reasoning models can include reasoning before the final JSON. Use `MD_JSON` for a fenced answer or `TOOLS` for a function call.
