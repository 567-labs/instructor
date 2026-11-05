# OpenAI-compatible gateways

Runcrate, AI Stupid Level, and Tsubasa accept Chat Completions through the OpenAI Python SDK. Pass the gateway’s own key when you construct the SDK client. Each request sends its messages and response schema to the selected gateway.

| Gateway | Base URL | Credential | Model selection |
| --- | --- | --- | --- |
| Runcrate | `https://api.runcrate.ai/v1` | `RUNCRATE_API_KEY` | Use an enabled model ID, including its organization prefix. [Chat API](https://www.runcrate.ai/chat-api) |
| AI Stupid Level | `https://aistupidlevel.info/v1` | `ASL_ROUTER_KEY` | Use a routing mode such as `auto-coding` or `auto-reasoning`. [Router API](https://aistupidlevel.info/router/docs) |
| Tsubasa | `https://api.tsubasa.sh/v1` | `TSUBASA_API_KEY` | Use a model from the gateway’s catalog, such as `tsubasa-fast`. [OpenAI SDK integration](https://tsubasa.sh/integrations/openai-sdk) |

Install `instructor` and set the credential for your gateway. The following example pairs each base URL with its own credential variable. Set `INSTRUCTOR_GATEWAY` to `runcrate`, `asl`, or `tsubasa`, and `GATEWAY_MODEL` to the model or routing mode you want to call.

```python
import os

import instructor
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel


class Contact(BaseModel):
    name: str
    team: str


gateways = {
    "runcrate": ("https://api.runcrate.ai/v1", "RUNCRATE_API_KEY"),
    "asl": ("https://aistupidlevel.info/v1", "ASL_ROUTER_KEY"),
    "tsubasa": ("https://api.tsubasa.sh/v1", "TSUBASA_API_KEY"),
}
base_url, key_variable = gateways[os.environ["INSTRUCTOR_GATEWAY"]]
sdk_options = {"base_url": base_url, "api_key": os.environ[key_variable]}
request = {
    "model": os.environ["GATEWAY_MODEL"],
    "messages": [{"role": "user", "content": "Ada works on the maps team."}],
}

with OpenAI(**sdk_options) as sdk:
    client = instructor.from_openai(sdk, mode=instructor.Mode.TOOLS)
    contact = client.chat.completions.create(response_model=Contact, **request)
    print(contact.name, contact.team)
```

For asynchronous code, reuse `Contact`, `sdk_options`, and `request` from the example above.

```python
import asyncio


async def main() -> None:
    async with AsyncOpenAI(**sdk_options) as sdk:
        client = instructor.from_openai(sdk, mode=instructor.Mode.TOOLS)
        contact = await client.chat.completions.create(response_model=Contact, **request)
        print(contact.name, contact.team)


asyncio.run(main())
```

`Mode.TOOLS` requires a model and route that support function calls. Choose `Mode.JSON_SCHEMA` when the route supports native structured outputs. For a text-only chat model, `Mode.MD_JSON` asks for JSON in the prompt and validates the response locally.

AI Stupid Level can route a request to different upstream models. Its response headers identify the selected route. `create_with_completion` returns both the validated model and the SDK completion when you need the completion’s model identifier.
