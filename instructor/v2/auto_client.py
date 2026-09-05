from __future__ import annotations

import importlib
from typing import Any, Callable, Literal, Optional, Union, cast, overload
from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor import __version__
from instructor.v2.core.mode import Mode
from instructor.models import KnownModelName
from instructor.cache import BaseCache
from instructor.v2.core.provider_specs import ALIAS_TO_PROVIDER
import warnings
import logging

# Type alias for the return type
InstructorType = Union[Instructor, AsyncInstructor]

logger = logging.getLogger("instructor.auto_client")


# Canonical strings and compatibility aliases accepted by from_provider().
supported_providers = list(ALIAS_TO_PROVIDER)


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[False] = False,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[True] = True,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_provider(
    model: str,
    async_client: Literal[False] = False,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_provider(
    model: str,
    async_client: Literal[True] = True,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_provider(
    model: Union[str, KnownModelName],  # noqa: UP007
    async_client: bool = False,
    cache: BaseCache | None = None,
    mode: Union[Mode, None] = None,  # noqa: ARG001, UP007
    **kwargs: Any,
) -> Union[Instructor, AsyncInstructor]:  # noqa: UP007
    """Create an Instructor client from a model string.

    Args:
        model: String in format "provider/model-name"
              (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet", "google/gemini-pro")
        async_client: Whether to return an async client
        cache: Optional cache adapter (e.g., ``AutoCache`` or ``RedisCache``)
               to enable transparent response caching. Automatically flows through
               **kwargs to all provider implementations.
        mode: Override the default mode for the provider. If not specified, uses the
              recommended default mode for each provider.
        **kwargs: Additional arguments passed to the provider client functions.
                 This includes the cache parameter and any provider-specific options.

    Returns:
        Instructor or AsyncInstructor instance

    Raises:
        ValueError: If provider is not supported or model string is invalid
        ImportError: If required package for provider is not installed

    Examples:
        >>> import instructor
        >>> from instructor.cache import AutoCache
        >>>
        >>> # Basic usage
        >>> client = instructor.from_provider("openai/gpt-4")
        >>> client = instructor.from_provider("anthropic/claude-3-sonnet")
        >>>
        >>> # With caching
        >>> cache = AutoCache(maxsize=1000)
        >>> client = instructor.from_provider("openai/gpt-4", cache=cache)
        >>>
        >>> # Async clients
        >>> async_client = instructor.from_provider("openai/gpt-4", async_client=True)
    """
    # Add cache to kwargs if provided so it flows through to provider functions
    if cache is not None:
        kwargs["cache"] = cache

    try:
        provider, model_name = model.split("/", 1)
    except ValueError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            'Model string must be in format "provider/model-name" '
            '(e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")'
        ) from None

    if not provider or not model_name:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            'Model string must be in format "provider/model-name" '
            'with both parts non-empty (e.g. "openai/gpt-4")'
        )

    provider_info = {"provider": provider, "operation": "initialize"}
    logger.info(
        "Initializing %s provider with model %s",
        provider,
        model_name,
        extra=provider_info,
    )
    logger.debug(
        "Provider configuration: async_client=%s, mode=%s",
        async_client,
        mode,
        extra=provider_info,
    )
    api_key = None
    if "api_key" in kwargs:
        api_key = kwargs.pop("api_key")
        if api_key:
            logger.debug(
                "API key provided for %s provider (length: %d characters)",
                provider,
                len(api_key),
                extra=provider_info,
            )

    builder = _PROVIDER_BUILDERS.get(provider)
    if builder is None:
        from instructor.v2.core.errors import ConfigurationError

        logger.error(
            "Error initializing %s client: unsupported provider",
            provider,
            extra={**provider_info, "status": "error"},
        )
        raise ConfigurationError(
            f"Unsupported provider: {provider}. "
            f"Supported providers are: {supported_providers}"
        )

    return builder(
        provider=provider,
        model_name=model_name,
        async_client=async_client,
        mode=mode,
        api_key=api_key,
        kwargs=kwargs,
        provider_info=provider_info,
    )


def _build_openai(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import openai
        from openai import DEFAULT_MAX_RETRIES, NotGiven, Timeout, not_given
        from collections.abc import Mapping
        from typing import cast
    except ImportError as err:
        missing_root = (getattr(err, "name", "") or "").split(".")[0]
        if missing_root != "openai":
            raise

        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The openai package is required to use the OpenAI provider. "
            "Install it with `pip install openai`."
        ) from None

    try:
        # Extract base_url and other OpenAI client parameters from kwargs
        base_url = kwargs.pop("base_url", None)
        organization = cast(Optional[str], kwargs.pop("organization", None))

        timeout_raw = kwargs.pop("timeout", not_given)
        timeout: float | Timeout | None | NotGiven
        timeout = (
            not_given
            if timeout_raw is not_given
            else cast(Optional[Union[float, Timeout]], timeout_raw)
        )

        max_retries_raw = kwargs.pop("max_retries", None)
        max_retries = (
            DEFAULT_MAX_RETRIES
            if max_retries_raw is None
            else int(cast(int, max_retries_raw))
        )

        default_headers = cast(
            Optional[Mapping[str, str]], kwargs.pop("default_headers", None)
        )
        default_query = cast(
            Optional[Mapping[str, object]], kwargs.pop("default_query", None)
        )
        http_client = kwargs.pop("http_client", None)
        strict_response_validation = bool(
            kwargs.pop("_strict_response_validation", False)
        )

        if async_client:
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                default_query=default_query,
                http_client=http_client,
                _strict_response_validation=strict_response_validation,
            )
        else:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                default_query=default_query,
                http_client=http_client,
                _strict_response_validation=strict_response_validation,
            )

        import instructor

        result = instructor.from_openai(
            client,
            model=model_name,
            mode=mode if mode else Mode.TOOLS,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_azure_openai(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import os
        from openai import AzureOpenAI, AsyncAzureOpenAI
        from instructor.v2.providers.openai.client import from_openai

        # Get required Azure OpenAI configuration from environment
        api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = kwargs.pop(
            "azure_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        api_version = kwargs.pop("api_version", "2024-02-01")

        if not api_key:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "AZURE_OPENAI_API_KEY is not set. "
                "Set it with `export AZURE_OPENAI_API_KEY=<your-api-key>` or pass it as kwarg api_key=<your-api-key>"
            )

        if not azure_endpoint:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "AZURE_OPENAI_ENDPOINT is not set. "
                "Set it with `export AZURE_OPENAI_ENDPOINT=<your-endpoint>` or pass it as kwarg azure_endpoint=<your-endpoint>"
            )

        client = (
            AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
            if async_client
            else AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
        )
        result = from_openai(
            client,
            model=model_name,
            mode=mode if mode else Mode.TOOLS,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The openai package is required to use the Azure OpenAI provider. "
            "Install it with `pip install openai`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_openai_compatible(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
    env_var: str,
    default_base_url: str,
    factory_name: str,
) -> InstructorType:
    try:
        import os
        import openai
        from instructor.v2.providers.openai import client as openai_client

        api_key = api_key or os.environ.get(env_var)
        if not api_key:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                f"{env_var} is not set. "
                f"Set it with `export {env_var}=<your-api-key>` or pass it as kwarg api_key=<your-api-key>"
            )

        base_url = kwargs.pop("base_url", default_base_url)
        client = (
            openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
            if async_client
            else openai.OpenAI(api_key=api_key, base_url=base_url)
        )
        factory = getattr(openai_client, factory_name)
        result = factory(
            client,
            model=model_name,
            mode=mode if mode else Mode.TOOLS,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            f"The openai package is required to use the {provider} provider. "
            "Install it with `pip install openai`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_anyscale(**kwargs: Any) -> InstructorType:
    return _build_openai_compatible(
        **kwargs,
        env_var="ANYSCALE_API_KEY",
        default_base_url="https://api.endpoints.anyscale.com/v1",
        factory_name="from_anyscale",
    )


def _build_together(**kwargs: Any) -> InstructorType:
    return _build_openai_compatible(
        **kwargs,
        env_var="TOGETHER_API_KEY",
        default_base_url="https://api.together.xyz/v1",
        factory_name="from_together",
    )


def _build_databricks(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import os
        import openai
        from instructor import from_openai

        api_key = (
            api_key
            or os.environ.get("DATABRICKS_TOKEN")
            or os.environ.get("DATABRICKS_API_KEY")
        )
        if not api_key:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "DATABRICKS_TOKEN is not set. "
                "Set it with `export DATABRICKS_TOKEN=<your-token>` or `export DATABRICKS_API_KEY=<your-token>` "
                "or pass it as kwarg `api_key=<your-token>`."
            )

        base_url = kwargs.pop("base_url", None)
        if base_url is None:
            base_url = (
                os.environ.get("DATABRICKS_BASE_URL")
                or os.environ.get("DATABRICKS_HOST")
                or os.environ.get("DATABRICKS_WORKSPACE_URL")
            )

        if not base_url:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "DATABRICKS_HOST is not set. "
                "Set it with `export DATABRICKS_HOST=<your-workspace-url>` or `export DATABRICKS_WORKSPACE_URL=<your-workspace-url>` "
                "or pass `base_url=<your-workspace-url>`."
            )

        base_url = str(base_url).rstrip("/")
        if not base_url.endswith("/serving-endpoints"):
            base_url = f"{base_url}/serving-endpoints"

        openai_client_kwargs = {}
        for key in (
            "organization",
            "timeout",
            "max_retries",
            "default_headers",
            "http_client",
            "app_info",
        ):
            if key in kwargs:
                openai_client_kwargs[key] = kwargs.pop(key)

        client = (
            openai.AsyncOpenAI(
                api_key=api_key, base_url=base_url, **openai_client_kwargs
            )
            if async_client
            else openai.OpenAI(
                api_key=api_key, base_url=base_url, **openai_client_kwargs
            )
        )
        result = from_openai(
            client,
            model=model_name,
            mode=mode if mode else Mode.TOOLS,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The openai package is required to use the Databricks provider. "
            "Install it with `pip install openai`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_anthropic(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import anthropic
        from instructor.v2.providers.anthropic.client import from_anthropic

        if from_anthropic is None:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "Failed to import Anthropic provider. "
                "This may be due to a configuration error or missing dependencies."
            )

        client = (
            anthropic.AsyncAnthropic(
                api_key=api_key,
                default_headers={"User-Agent": f"instructor/{__version__}"},
            )
            if async_client
            else anthropic.Anthropic(
                api_key=api_key,
                default_headers={"User-Agent": f"instructor/{__version__}"},
            )
        )
        # Set default max_tokens if not provided (like v1)
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096
        # Use Mode.TOOLS instead of Mode.ANTHROPIC_TOOLS
        result = from_anthropic(
            client,
            model=model_name,
            mode=mode if mode else Mode.TOOLS,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The anthropic package is required to use the Anthropic provider. "
            "Install it with `pip install anthropic`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_google(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    # Import google-genai package - catch ImportError only for actual imports
    try:
        import google.genai as genai
    except ImportError as e:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The google-genai package is required to use the Google provider. "
            "Install it with `pip install google-genai`."
        ) from e

    try:
        import os

        # Remove vertexai from kwargs if present to avoid passing it twice
        vertexai_flag = kwargs.pop("vertexai", False)

        # Get API key from kwargs or environment
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        # Extract client-specific parameters
        client_kwargs = {}
        for key in [
            "debug_config",
            "http_options",
            "credentials",
            "project",
            "location",
        ]:
            if key in kwargs:
                client_kwargs[key] = kwargs.pop(key)

        client = genai.Client(
            vertexai=vertexai_flag,
            api_key=api_key,
            **client_kwargs,
        )
        # Default to TOOLS for v2
        # Extract model from kwargs if present, otherwise use model_name
        model_param = kwargs.pop("model", model_name)
        import instructor

        result = instructor.from_genai(
            client,
            mode=mode if mode else Mode.TOOLS,
            use_async=async_client,
            model=model_param,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_gemini(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import os

        genai = cast(Any, importlib.import_module("google.generativeai"))
        from instructor.v2.providers.gemini.client import from_gemini

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

        client = genai.GenerativeModel(model_name)
        result = from_gemini(
            client,
            mode=mode if mode else Mode.MD_JSON,
            use_async=async_client,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The google-generativeai package is required to use the Gemini provider. "
            "Install it with `pip install google-generativeai`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_mistral(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,  # noqa: ARG001
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        try:
            mistral_module = cast(Any, importlib.import_module("mistralai.client"))
            Mistral = mistral_module.Mistral
        except (AttributeError, ImportError):
            mistral_module = cast(Any, importlib.import_module("mistralai"))
            Mistral = mistral_module.Mistral
        from instructor.v2.providers.mistral.client import from_mistral
        import os

        api_key = api_key or os.environ.get("MISTRAL_API_KEY")

        if api_key:
            client = Mistral(api_key=api_key)
        else:
            raise ValueError(
                "MISTRAL_API_KEY is not set. "
                "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
            )

        if async_client:
            result = from_mistral(client, model=model_name, use_async=True, **kwargs)
        else:
            result = from_mistral(client, model=model_name, **kwargs)
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The mistralai package is required to use the Mistral provider. "
            "Install it with `pip install mistralai`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_cohere(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import cohere
        from instructor.v2.providers.cohere.client import from_cohere

        client = (
            cohere.AsyncClientV2(api_key=api_key)
            if async_client
            else cohere.ClientV2(api_key=api_key)
        )
        # Use Mode.TOOLS as default for Cohere
        result = from_cohere(
            client,
            mode=mode if mode else Mode.TOOLS,
            model=model_name,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The cohere package is required to use the Cohere provider. "
            "Install it with `pip install cohere`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_perplexity(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,  # noqa: ARG001
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import openai
        from instructor.v2.providers.perplexity.client import from_perplexity
        import os

        api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "PERPLEXITY_API_KEY is not set. "
                "Set it with `export PERPLEXITY_API_KEY=<your-api-key>` or pass it as a kwarg api_key=<your-api-key>"
            )

        client = (
            openai.AsyncOpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
            if async_client
            else openai.OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        )
        result = from_perplexity(client, model=model_name, **kwargs)
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The openai package is required to use the Perplexity provider. "
            "Install it with `pip install openai`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_groq(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,  # noqa: ARG001
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import groq
        from instructor.v2.providers.groq.client import from_groq

        client = (
            groq.AsyncGroq(api_key=api_key)
            if async_client
            else groq.Groq(api_key=api_key)
        )
        result = from_groq(client, model=model_name, **kwargs)
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The groq package is required to use the Groq provider. "
            "Install it with `pip install groq`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_writer(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,  # noqa: ARG001
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        from writerai import AsyncWriter, Writer
        from instructor.v2.providers.writer.client import from_writer

        client = (
            AsyncWriter(api_key=api_key) if async_client else Writer(api_key=api_key)
        )
        result = from_writer(client, model=model_name, **kwargs)
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The writerai package is required to use the Writer provider. "
            "Install it with `pip install writer-sdk`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_bedrock(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,  # noqa: ARG001
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import os
        import boto3
        from instructor.v2.providers.bedrock.client import from_bedrock

        # Get AWS configuration from environment or kwargs
        if "region" in kwargs:
            region = kwargs.pop("region")
        else:
            logger.debug(
                "AWS_DEFAULT_REGION is not set. Using default region us-east-1"
            )
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        # Extract AWS-specific parameters
        # Dictionary to collect AWS credentials and session parameters for boto3 client
        aws_kwargs = {}
        for key in [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
        ]:
            if key in kwargs:
                aws_kwargs[key] = kwargs.pop(key)
            elif key.upper() in os.environ:
                logger.debug(f"Using {key.upper()} from environment variable")
                aws_kwargs[key] = os.environ[key.upper()]

        # Add region to client configuration
        aws_kwargs["region_name"] = region

        # Create bedrock-runtime client
        client = boto3.client("bedrock-runtime", **aws_kwargs)

        # Determine default mode based on model
        if mode is None:
            # Anthropic models (Claude) support tools, others use JSON
            if model_name and (
                "anthropic" in model_name.lower() or "claude" in model_name.lower()
            ):
                default_mode = Mode.TOOLS
            else:
                default_mode = Mode.MD_JSON
        else:
            default_mode = mode

        result = from_bedrock(
            client,
            mode=default_mode,
            async_client=async_client,
            model=model_name,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The boto3 package is required to use the AWS Bedrock provider. "
            "Install it with `pip install boto3`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_cerebras(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,  # noqa: ARG001
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        from cerebras.cloud.sdk import AsyncCerebras, Cerebras
        from instructor.v2.providers.cerebras.client import from_cerebras

        client = (
            AsyncCerebras(api_key=api_key)
            if async_client
            else Cerebras(api_key=api_key)
        )
        result = from_cerebras(client, model=model_name, **kwargs)
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The cerebras package is required to use the Cerebras provider. "
            "Install it with `pip install cerebras`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_fireworks(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,  # noqa: ARG001
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        from fireworks.client import AsyncFireworks, Fireworks
        from instructor.v2.providers.fireworks.client import from_fireworks

        client = (
            AsyncFireworks(api_key=api_key)
            if async_client
            else Fireworks(api_key=api_key)
        )
        result = from_fireworks(client, model=model_name, **kwargs)
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The fireworks-ai package is required to use the Fireworks provider. "
            "Install it with `pip install fireworks-ai`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_vertexai(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,  # noqa: ARG001
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    warnings.warn(
        "The 'vertexai' provider is deprecated. Use 'google' provider with vertexai=True instead. "
        "Example: instructor.from_provider('google/gemini-pro', vertexai=True)",
        DeprecationWarning,
        stacklevel=2,
    )
    # Import Vertex AI SDK
    try:
        import vertexai
        import vertexai.generative_models as gm
    except ImportError as e:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The vertexai package is required to use the VertexAI provider. "
            "Install it with `pip install google-cloud-aiplatform`."
        ) from e

    try:
        import os

        # Get project and location from kwargs or environment
        project = kwargs.pop("project", os.environ.get("GOOGLE_CLOUD_PROJECT"))
        location = kwargs.pop(
            "location", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        )

        if not project:
            raise ValueError(
                "Project ID is required for Vertex AI. "
                "Set it with `export GOOGLE_CLOUD_PROJECT=<your-project-id>` "
                "or pass it as kwarg project=<your-project-id>"
            )

        credentials = kwargs.pop("credentials", None)
        vertexai.init(project=project, location=location, credentials=credentials)

        client = gm.GenerativeModel(model_name)
        import instructor

        result = instructor.from_vertexai(
            client,
            use_async=async_client,
            mode=mode if mode else Mode.TOOLS,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_generative_ai(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    warnings.warn(
        "The 'generative-ai' provider is deprecated. Use 'google' provider instead. "
        "Example: instructor.from_provider('google/gemini-pro')",
        DeprecationWarning,
        stacklevel=2,
    )
    # Import google-genai package - catch ImportError only for actual imports
    try:
        from google import genai
        from instructor.v2.providers.genai.client import from_genai
    except ImportError as e:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The google-genai package is required to use the Google GenAI provider. "
            "Install it with `pip install google-genai`."
        ) from e

    try:
        import os

        # Get API key from kwargs or environment
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        client = genai.Client(vertexai=False, api_key=api_key)
        if async_client:
            result = from_genai(
                client,
                use_async=True,
                model=model_name,
                mode=mode if mode else Mode.TOOLS,
                **kwargs,
            )
        else:
            result = from_genai(
                client,
                model=model_name,
                mode=mode if mode else Mode.TOOLS,
                **kwargs,
            )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_ollama(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import openai
        from instructor.v2.providers.openai.client import from_openai

        # Get base_url from kwargs or use default
        base_url = kwargs.pop("base_url", "http://localhost:11434/v1")
        api_key = kwargs.pop("api_key", "ollama")  # required but unused

        client = (
            openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
            if async_client
            else openai.OpenAI(base_url=base_url, api_key=api_key)
        )

        # Models that support function calling (tools mode)
        tool_capable_models = {
            "llama3.1",
            "llama3.2",
            "llama4",
            "mistral-nemo",
            "firefunction-v2",
            "command-a",
            "command-r",
            "command-r-plus",
            "command-r7b",
            "qwen2.5",
            "qwen2.5-coder",
            "qwen3",
            "devstral",
        }

        # Check if model supports tools by looking at model name
        supports_tools = any(
            capable_model in model_name.lower() for capable_model in tool_capable_models
        )

        default_mode = Mode.TOOLS if supports_tools else Mode.JSON

        result = from_openai(
            client,
            model=model_name,
            mode=mode if mode else default_mode,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The openai package is required to use the Ollama provider. "
            "Install it with `pip install openai`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_deepseek(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import openai
        from instructor.v2.providers.openai.client import from_deepseek
        import os

        # Get API key from kwargs or environment
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

        if not api_key:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "DEEPSEEK_API_KEY is not set. "
                "Set it with `export DEEPSEEK_API_KEY=<your-api-key>` or pass it as kwarg api_key=<your-api-key>"
            )

        # DeepSeek uses OpenAI-compatible API
        base_url = kwargs.pop("base_url", "https://api.deepseek.com")

        client = (
            openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
            if async_client
            else openai.OpenAI(api_key=api_key, base_url=base_url)
        )

        result = from_deepseek(
            client,
            model=model_name,
            mode=mode if mode else Mode.TOOLS,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The openai package is required to use the DeepSeek provider. "
            "Install it with `pip install openai`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_xai(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        from xai_sdk.sync.client import Client as SyncClient
        from xai_sdk.aio.client import Client as AsyncClient
        from instructor.v2.providers.xai.client import from_xai

        if from_xai is None:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "Failed to import xAI provider. "
                "This may be due to a configuration error or missing dependencies."
            )

        client = (
            AsyncClient(api_key=api_key)
            if async_client
            else SyncClient(api_key=api_key)
        )
        # Use Mode.TOOLS instead of Mode.XAI_TOOLS (v2 uses generic modes)
        result = from_xai(
            client,
            mode=mode if mode else Mode.TOOLS,
            model=model_name,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The xAI provider needs the optional dependency `xai-sdk`. "
            'Install it with `uv pip install "instructor[xai]"` (or `pip install "instructor[xai]"`). '
            "Note: xai-sdk requires Python 3.10+."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_openrouter(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        import openai
        from instructor.v2.providers.openrouter.client import from_openrouter
        import os

        # Get API key from kwargs or environment
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "OPENROUTER_API_KEY is not set. "
                "Set it with `export OPENROUTER_API_KEY=<your-api-key>` or pass it as kwarg api_key=<your-api-key>"
            )

        # OpenRouter uses OpenAI-compatible API
        base_url = kwargs.pop("base_url", "https://openrouter.ai/api/v1")

        client = (
            openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
            if async_client
            else openai.OpenAI(api_key=api_key, base_url=base_url)
        )

        result = from_openrouter(
            client,
            model=model_name,
            mode=mode if mode else Mode.TOOLS,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The openai package is required to use the OpenRouter provider. "
            "Install it with `pip install openai`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


def _build_litellm(
    *,
    provider: str,
    model_name: str,  # noqa: ARG001
    async_client: bool,
    mode: Mode | None,
    api_key: str | None,  # noqa: ARG001
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> InstructorType:
    try:
        from litellm import completion, acompletion
        from instructor.v2.providers.litellm.client import from_litellm

        completion_func = acompletion if async_client else completion
        result = from_litellm(
            completion_func,
            mode=mode if mode else Mode.TOOLS,
            **kwargs,
        )
        logger.info(
            "Client initialized",
            extra={**provider_info, "status": "success"},
        )
        return result
    except ImportError:
        from instructor.v2.core.errors import ConfigurationError

        raise ConfigurationError(
            "The litellm package is required to use the LiteLLM provider. "
            "Install it with `pip install litellm`."
        ) from None
    except Exception as e:
        logger.error(
            "Error initializing %s client: %s",
            provider,
            e,
            exc_info=True,
            extra={**provider_info, "status": "error"},
        )
        raise


ProviderBuilder = Callable[..., InstructorType]

_PROVIDER_BUILDERS: dict[str, ProviderBuilder] = {
    "openai": _build_openai,
    "anyscale": _build_anyscale,
    "together": _build_together,
    "azure_openai": _build_azure_openai,
    "databricks": _build_databricks,
    "anthropic": _build_anthropic,
    "google": _build_google,
    "gemini": _build_gemini,
    "mistral": _build_mistral,
    "cohere": _build_cohere,
    "perplexity": _build_perplexity,
    "groq": _build_groq,
    "writer": _build_writer,
    "bedrock": _build_bedrock,
    "cerebras": _build_cerebras,
    "fireworks": _build_fireworks,
    "vertexai": _build_vertexai,
    "generative-ai": _build_generative_ai,
    "ollama": _build_ollama,
    "deepseek": _build_deepseek,
    "xai": _build_xai,
    "openrouter": _build_openrouter,
    "litellm": _build_litellm,
}
