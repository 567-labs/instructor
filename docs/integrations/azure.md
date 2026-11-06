# Azure OpenAI and Microsoft Foundry

Use the deployment name from your Azure resource as the model. The installed OpenAI SDK supports both the dated Azure OpenAI API and the Foundry v1 API.

## Connect to an Azure OpenAI deployment

Set `AZURE_OPENAI_ENDPOINT` to your resource URL and `AZURE_OPENAI_API_KEY` to its key. Select an API version that supports your deployment and the features you use.

```python
import instructor
from pydantic import BaseModel


class Contact(BaseModel):
    name: str


client = instructor.from_provider(
    "azure/my-deployment",
    api_version="2024-10-21",
)
contact = client.create(
    response_model=Contact,
    messages=[{"role": "user", "content": "The contact is Avery."}],
)
```

`azure_openai/my-deployment` is an equivalent provider string. Pass `async_client=True` and await `client.create(...)` for asynchronous calls. You can also pass an existing `AzureOpenAI` or `AsyncAzureOpenAI` client to `instructor.from_azure`.

## Use a Foundry v1 endpoint

Pass the resource's OpenAI-compatible v1 URL and a deployment name. The v1 endpoint uses the standard OpenAI client.

```python
import os

import instructor
from openai import OpenAI

sdk = OpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
client = instructor.from_azure(sdk, model="my-deployment")
```

## Authenticate with Microsoft Entra ID

Install `azure-identity` and assign the identity an appropriate role on the Azure resource. For the dated Azure API, supply a token provider for the Cognitive Services scope.

```python
import instructor
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)
client = instructor.from_provider(
    "azure/my-deployment",
    azure_ad_token_provider=token_provider,
    api_version="2024-10-21",
)
```

## Choose a mode

Instructor defaults to `Mode.TOOLS`. It also provides `JSON`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS` and `RESPONSES_TOOLS`. Available service features depend on the deployment and API version. For example, choose a Foundry v1 deployment that supports the Responses API before using `RESPONSES_TOOLS`.

Microsoft documents the supported endpoints, API versions and identity setup. [API version lifecycle](https://learn.microsoft.com/azure/ai-foundry/openai/api-version-lifecycle) · [Keyless connections](https://learn.microsoft.com/azure/developer/ai/keyless-connections)
