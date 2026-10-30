"""v2 OpenAI client factory.

Creates Instructor instances using v2 hierarchical registry system.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal, overload

import openai

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.openai import handlers  # noqa: F401


def map_chat_completion_to_response(messages, client, *args, **kwargs) -> Any:
    return client.responses.create(*args, input=messages, **kwargs)


async def async_map_chat_completion_to_response(
    messages, client, *args, **kwargs
) -> Any:
    return await client.responses.create(*args, input=messages, **kwargs)


def _from_openai_compat(
    client: openai.OpenAI | openai.AsyncOpenAI,
    provider: Provider,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    from instructor.v2.core.registry import mode_registry, normalize_mode

    normalized_mode = (
        Mode.RESPONSES_TOOLS
        if mode == Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
        else normalize_mode(provider, mode)
    )
    if not mode_registry.is_registered(provider, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(provider)
        raise ModeError(
            mode=str(mode.value),
            provider=provider.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    valid_client_types = (
        openai.OpenAI,
        openai.AsyncOpenAI,
    )

    if not isinstance(client, valid_client_types):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}:
        create = (
            partial(map_chat_completion_to_response, client=client)
            if isinstance(client, openai.OpenAI)
            else partial(async_map_chat_completion_to_response, client=client)
        )
    else:
        create = client.chat.completions.create
    patched_create = patch_v2(
        func=create,
        provider=provider,
        mode=normalized_mode,
        default_model=model,
    )

    if isinstance(client, openai.OpenAI):
        return Instructor(
            client=client,
            create=patched_create,
            provider=provider,
            mode=normalized_mode,
            **kwargs,
        )
    return AsyncInstructor(
        client=client,
        create=patched_create,
        provider=provider,
        mode=normalized_mode,
        **kwargs,
    )


@overload
def from_openai(
    client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_openai(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from an OpenAI client using v2 registry.

    Args:
        client: An instance of OpenAI client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for OpenAI
        ClientError: If client is not a valid OpenAI client instance

    Examples:
        >>> import openai
        >>> from instructor import Mode
        >>> from instructor.v2.providers.openai import from_openai
        >>>
        >>> client = openai.OpenAI()
        >>> instructor_client = from_openai(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use JSON_SCHEMA mode for structured outputs
        >>> instructor_client = from_openai(client, mode=Mode.JSON_SCHEMA)
    """
    return _from_openai_compat(
        client=client,
        provider=Provider.OPENAI,
        mode=mode,
        model=model,
        **kwargs,
    )


@overload
def from_anyscale(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_anyscale(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_anyscale(
    model_or_client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_anyscale(
    model_or_client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_anyscale(
    model_or_client: str | openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance for Anyscale.

    Supports two usage patterns:

    1. String-based (recommended): Pass a model name string
       >>> from instructor.v2 import from_anyscale
       >>> client = from_anyscale("Mixtral-8x7B-Instruct-v0.1", mode=Mode.TOOLS)

    2. Client-based (backward compatible): Pass an OpenAI client instance
       >>> from openai import OpenAI
       >>> client = OpenAI(base_url="https://api.endpoints.anyscale.com/v1")
       >>> instructor_client = from_anyscale(client, mode=Mode.TOOLS)

    Args:
        model_or_client: Model name string (delegates to from_provider) or OpenAI client instance
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model name (only used with client-based usage)
        async_client: Whether to return async client (only used with string-based usage)
        **kwargs: Additional keyword arguments passed to from_provider or Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on usage pattern)

    Raises:
        ModeError: If mode is not registered for Anyscale
        ClientError: If client is not a valid OpenAI client instance (client-based usage)
    """
    # String-based: delegate to from_provider
    if isinstance(model_or_client, str):
        from instructor import from_provider

        return from_provider(
            f"anyscale/{model_or_client}",
            mode=mode,
            async_client=async_client,
            **kwargs,
        )

    # Client-based: existing behavior
    return _from_openai_compat(
        model_or_client,
        provider=Provider.ANYSCALE,
        mode=mode,
        model=model,
        **kwargs,
    )


@overload
def from_together(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_together(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_together(
    model_or_client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_together(
    model_or_client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_together(
    model_or_client: str | openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance for Together AI.

    Supports two usage patterns:

    1. String-based (recommended): Pass a model name string
       >>> from instructor.v2 import from_together
       >>> client = from_together("Mixtral-8x7B-Instruct-v0.1", mode=Mode.TOOLS)

    2. Client-based (backward compatible): Pass an OpenAI client instance
       >>> from openai import OpenAI
       >>> client = OpenAI(base_url="https://api.together.xyz/v1")
       >>> instructor_client = from_together(client, mode=Mode.TOOLS)

    Args:
        model_or_client: Model name string (delegates to from_provider) or OpenAI client instance
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model name (only used with client-based usage)
        async_client: Whether to return async client (only used with string-based usage)
        **kwargs: Additional keyword arguments passed to from_provider or Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on usage pattern)

    Raises:
        ModeError: If mode is not registered for Together AI
        ClientError: If client is not a valid OpenAI client instance (client-based usage)
    """
    # String-based: delegate to from_provider
    if isinstance(model_or_client, str):
        from instructor import from_provider

        return from_provider(
            f"together/{model_or_client}",
            mode=mode,
            async_client=async_client,
            **kwargs,
        )

    # Client-based: existing behavior
    return _from_openai_compat(
        model_or_client,
        provider=Provider.TOGETHER,
        mode=mode,
        model=model,
        **kwargs,
    )


@overload
def from_databricks(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_databricks(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_databricks(
    model_or_client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_databricks(
    model_or_client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_databricks(
    model_or_client: str | openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance for Databricks.

    Supports two usage patterns:

    1. String-based (recommended): Pass a model name string
       >>> from instructor.v2 import from_databricks
       >>> client = from_databricks("dbrx-instruct", mode=Mode.TOOLS)

    2. Client-based (backward compatible): Pass an OpenAI client instance
       >>> from openai import OpenAI
       >>> client = OpenAI(base_url="https://workspace.cloud.databricks.com/serving-endpoints")
       >>> instructor_client = from_databricks(client, mode=Mode.TOOLS)

    Args:
        model_or_client: Model name string (delegates to from_provider) or OpenAI client instance
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model name (only used with client-based usage)
        async_client: Whether to return async client (only used with string-based usage)
        **kwargs: Additional keyword arguments passed to from_provider or Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on usage pattern)

    Raises:
        ModeError: If mode is not registered for Databricks
        ClientError: If client is not a valid OpenAI client instance (client-based usage)
    """
    # String-based: delegate to from_provider
    if isinstance(model_or_client, str):
        from instructor import from_provider

        return from_provider(
            f"databricks/{model_or_client}",
            mode=mode,
            async_client=async_client,
            **kwargs,
        )

    # Client-based: existing behavior
    return _from_openai_compat(
        model_or_client,
        provider=Provider.DATABRICKS,
        mode=mode,
        model=model,
        **kwargs,
    )


@overload
def from_deepseek(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_deepseek(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_deepseek(
    model_or_client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_deepseek(
    model_or_client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_deepseek(
    model_or_client: str | openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance for DeepSeek.

    Supports two usage patterns:

    1. String-based (recommended): Pass a model name string
       >>> from instructor.v2 import from_deepseek
       >>> client = from_deepseek("deepseek-chat", mode=Mode.TOOLS)

    2. Client-based (backward compatible): Pass an OpenAI client instance
       >>> from openai import OpenAI
       >>> client = OpenAI(base_url="https://api.deepseek.com")
       >>> instructor_client = from_deepseek(client, mode=Mode.TOOLS)

    Args:
        model_or_client: Model name string (delegates to from_provider) or OpenAI client instance
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model name (only used with client-based usage)
        async_client: Whether to return async client (only used with string-based usage)
        **kwargs: Additional keyword arguments passed to from_provider or Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on usage pattern)

    Raises:
        ModeError: If mode is not registered for DeepSeek
        ClientError: If client is not a valid OpenAI client instance (client-based usage)
    """
    # String-based: delegate to from_provider
    if isinstance(model_or_client, str):
        from instructor import from_provider

        return from_provider(
            f"deepseek/{model_or_client}",
            mode=mode,
            async_client=async_client,
            **kwargs,
        )

    # Client-based: existing behavior
    return _from_openai_compat(
        model_or_client,
        provider=Provider.DEEPSEEK,
        mode=mode,
        model=model,
        **kwargs,
    )


@overload
def from_zhipuai(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_zhipuai(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_zhipuai(
    model_or_client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_zhipuai(
    model_or_client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_zhipuai(
    model_or_client: str | openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance for ZhipuAI / BigModel.

    Supports two usage patterns:

    1. String-based (recommended): Pass a model name string
       >>> from instructor.v2 import from_zhipuai
       >>> client = from_zhipuai("glm-5.3-flash", mode=Mode.TOOLS)

    2. Client-based (backward compatible): Pass an OpenAI client instance pointing to ZhipuAI
       >>> from openai import OpenAI
       >>> client = OpenAI(base_url="https://open.bigmodel.cn/api/paas/v4/")
       >>> instructor_client = from_zhipuai(client, mode=Mode.TOOLS)

    Args:
        model_or_client: Model name string (delegates to from_provider) or OpenAI client instance
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model name (only used with client-based usage)
        async_client: Whether to return async client (only used with string-based usage)
        **kwargs: Additional keyword arguments passed to from_provider or Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on usage pattern)

    Raises:
        ModeError: If mode is not registered for ZhipuAI
        ClientError: If client is not a valid OpenAI client instance (client-based usage)
    """
    # String-based: delegate to from_provider
    if isinstance(model_or_client, str):
        from instructor import from_provider

        return from_provider(
            f"zhipuai/{model_or_client}",
            mode=mode,
            async_client=async_client,
            **kwargs,
        )

    # Client-based: existing behavior
    return _from_openai_compat(
        model_or_client,
        provider=Provider.ZHIPUAI,
        mode=mode,
        model=model,
        **kwargs,
    )


@overload
def from_zai(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_zai(
    model_or_client: str,
    mode: Mode = Mode.TOOLS,
    model: None = None,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_zai(
    model_or_client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_zai(
    model_or_client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_zai(
    model_or_client: str | openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance for Z.ai.

    Supports two usage patterns:

    1. String-based (recommended): Pass a model name string
       >>> from instructor.v2 import from_zai
       >>> client = from_zai("glm-5.3-flash", mode=Mode.TOOLS)

    2. Client-based (backward compatible): Pass an OpenAI client instance pointing to Z.ai
       >>> from openai import OpenAI
       >>> client = OpenAI(base_url="https://api.z.ai/api/paas/v4/")
       >>> instructor_client = from_zai(client, mode=Mode.TOOLS)

    Args:
        model_or_client: Model name string (delegates to from_provider) or OpenAI client instance
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model name (only used with client-based usage)
        async_client: Whether to return async client (only used with string-based usage)
        **kwargs: Additional keyword arguments passed to from_provider or Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on usage pattern)

    Raises:
        ModeError: If mode is not registered for Z.ai
        ClientError: If client is not a valid OpenAI client instance (client-based usage)
    """
    # String-based: delegate to from_provider
    if isinstance(model_or_client, str):
        from instructor import from_provider

        return from_provider(
            f"zai/{model_or_client}",
            mode=mode,
            async_client=async_client,
            **kwargs,
        )

    # Client-based: existing behavior
    return _from_openai_compat(
        model_or_client,
        provider=Provider.ZAI,
        mode=mode,
        model=model,
        **kwargs,
    )
