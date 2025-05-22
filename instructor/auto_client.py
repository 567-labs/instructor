from __future__ import annotations
from typing import Any, Union, Literal, overload
from instructor.client import AsyncInstructor, Instructor
import instructor
from instructor.models import KnownModelName

# Type alias for the return type
InstructorType = Union[Instructor, AsyncInstructor]


# List of supported providers
supported_providers = [
    "openai",
    "anthropic",
    "google",
    "mistral",
    "cohere",
    "perplexity",
    "groq",
    "writer",
    "bedrock",
    "cerebras",
    "fireworks",
    "vertexai",
    "generative-ai",
]


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_provider(
    model: str, async_client: Literal[True] = True, **kwargs: Any
) -> AsyncInstructor: ...


@overload
def from_provider(
    model: str, async_client: Literal[False] = False, **kwargs: Any
) -> Instructor: ...


def from_provider(
    model: Union[str, KnownModelName],  # noqa: UP007
    async_client: bool = False,
    mode: Union[instructor.Mode, None] = None,  # noqa: ARG001, UP007
    **kwargs: Any,
) -> Union[Instructor, AsyncInstructor]:  # noqa: UP007
    """Create an Instructor client from a model string.

    Args:
        model: String in format "provider/model-name"
              (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet", "google/gemini-pro")
        async_client: Whether to return an async client
        mode: Override the default mode for the provider. If not specified, uses the
              recommended default mode for each provider.
        **kwargs: Additional arguments passed to the client constructor

    Returns:
        Instructor or AsyncInstructor instance

    Raises:
        ValueError: If provider is not supported or model string is invalid
        ImportError: If required package for provider is not installed

    Examples:
        >>> import instructor
        >>> # Sync clients
        >>> client = instructor.from_provider("openai/gpt-4")
        >>> client = instructor.from_provider("anthropic/claude-3-sonnet")
        >>> # Async clients
        >>> async_client = instructor.from_provider("openai/gpt-4", async_client=True)
    """
    try:
        provider, model_name = model.split("/", 1)
    except ValueError:
        from instructor.exceptions import ConfigurationError

        raise ConfigurationError(
            'Model string must be in format "provider/model-name" '
            '(e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")'
        ) from None

    if provider == "openai":
        try:
            import openai
            from instructor import from_openai

            client = openai.AsyncOpenAI() if async_client else openai.OpenAI()
            return from_openai(
                client,
                model=model_name,
                mode=mode if mode else instructor.Mode.TOOLS,
                **kwargs,
            )
        except ImportError:
            from instructor.exceptions import ConfigurationError

            raise ConfigurationError(
                "The openai package is required to use the OpenAI provider. "
                "Install it with `pip install openai`."
            ) from None

    elif provider == "anthropic":
        try:
            import anthropic
            from instructor import from_anthropic

            client = (
                anthropic.AsyncAnthropic() if async_client else anthropic.Anthropic()
            )
            max_tokens = kwargs.pop("max_tokens", 4096)
            return from_anthropic(
                client,
                model=model_name,
                mode=mode if mode else instructor.Mode.ANTHROPIC_TOOLS,
                max_tokens=max_tokens,
                **kwargs,
            )
        except ImportError:
            import_err = ImportError(
                "The anthropic package is required to use the Anthropic provider. "
                "Install it with `pip install anthropic`."
            )
            raise import_err from None

    elif provider == "google":
        try:
            import google.genai as genai  # type: ignore
            from instructor import from_genai

            client = genai.Client(
                vertexai=False
                if kwargs.get("vertexai") is None
                else kwargs.get("vertexai"),
                **kwargs,
            )  # type: ignore
            if async_client:
                return from_genai(client, use_async=True, model=model_name, **kwargs)  # type: ignore
            else:
                return from_genai(client, model=model_name, **kwargs)  # type: ignore
        except ImportError:
            import_err = ImportError(
                "The google-genai package is required to use the Google provider. "
                "Install it with `pip install google-genai`."
            )
            raise import_err from None

    elif provider == "mistral":
        try:
            from mistralai import Mistral  # type: ignore
            from instructor import from_mistral
            import os

            if os.environ.get("MISTRAL_API_KEY"):
                client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
            else:
                raise ValueError(
                    "MISTRAL_API_KEY is not set. "
                    "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
                )

            if async_client:
                return from_mistral(client, model=model_name, use_async=True, **kwargs)
            else:
                return from_mistral(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The mistralai package is required to use the Mistral provider. "
                "Install it with `pip install mistralai`."
            )
            raise import_err from None

    elif provider == "cohere":
        try:
            import cohere
            from instructor import from_cohere

            client = cohere.AsyncClient() if async_client else cohere.Client()
            return from_cohere(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The cohere package is required to use the Cohere provider. "
                "Install it with `pip install cohere`."
            )
            raise import_err from None

    elif provider == "perplexity":
        try:
            import openai
            from instructor import from_perplexity
            import os

            if os.environ.get("PERPLEXITY_API_KEY"):
                api_key = os.environ.get("PERPLEXITY_API_KEY")
            elif kwargs.get("api_key"):
                api_key = kwargs.get("api_key")
            else:
                raise ValueError(
                    "PERPLEXITY_API_KEY is not set. "
                    "Set it with `export PERPLEXITY_API_KEY=<your-api-key>` or pass it as a kwarg api_key=<your-api-key>"
                )

            client = (
                openai.AsyncOpenAI(
                    api_key=api_key, base_url="https://api.perplexity.ai"
                )
                if async_client
                else openai.OpenAI(
                    api_key=api_key, base_url="https://api.perplexity.ai"
                )
            )
            return from_perplexity(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The openai package is required to use the Perplexity provider. "
                "Install it with `pip install openai`."
            )
            raise import_err from None

    elif provider == "groq":
        try:
            import groq
            from instructor import from_groq

            client = groq.AsyncGroq() if async_client else groq.Groq()
            return from_groq(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The groq package is required to use the Groq provider. "
                "Install it with `pip install groq`."
            )
            raise import_err from None

    elif provider == "writer":
        try:
            from writerai import AsyncWriter, Writer
            from instructor import from_writer

            client = AsyncWriter() if async_client else Writer()
            return from_writer(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The writerai package is required to use the Writer provider. "
                "Install it with `pip install writerai`."
            )
            raise import_err from None

    elif provider == "bedrock":
        try:
            import boto3
            from instructor import from_bedrock

            client = boto3.client("bedrock-runtime")
            return from_bedrock(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The boto3 package is required to use the AWS Bedrock provider. "
                "Install it with `pip install boto3`."
            )
            raise import_err from None

    elif provider == "cerebras":
        try:
            from cerebras.cloud.sdk import AsyncCerebras, Cerebras
            from instructor import from_cerebras

            client = AsyncCerebras() if async_client else Cerebras()
            return from_cerebras(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The cerebras package is required to use the Cerebras provider. "
                "Install it with `pip install cerebras`."
            )
            raise import_err from None

    elif provider == "fireworks":
        try:
            from fireworks.client import AsyncFireworks, Fireworks
            from instructor import from_fireworks

            client = AsyncFireworks() if async_client else Fireworks()
            return from_fireworks(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The fireworks-ai package is required to use the Fireworks provider. "
                "Install it with `pip install fireworks-ai`."
            )
            raise import_err from None

    elif provider == "vertexai":
        try:
            import vertexai.generative_models as gm
            from instructor import from_vertexai

            client = gm.GenerativeModel(model_name=model_name)
            return from_vertexai(client, use_async=async_client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The google-cloud-aiplatform package is required to use the VertexAI provider. "
                "Install it with `pip install google-cloud-aiplatform`."
            )
            raise import_err from None

    elif provider == "generative-ai":
        try:
            from google.generativeai import GenerativeModel
            from instructor import from_gemini

            client = GenerativeModel(model_name=model_name)
            if async_client:
                return from_gemini(client, use_async=True, **kwargs)  # type: ignore
            else:
                return from_gemini(client, **kwargs)  # type: ignore
        except ImportError:
            import_err = ImportError(
                "The google-generativeai package is required to use the Google GenAI provider. "
                "Install it with `pip install google-genai`."
            )
            raise import_err from None

    else:
        from instructor.exceptions import ConfigurationError

        raise ConfigurationError(
            f"Unsupported provider: {provider}. "
            f"Supported providers are: {supported_providers}"
        )
