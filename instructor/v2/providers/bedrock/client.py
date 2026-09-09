"""v2 Bedrock client factory."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.bedrock import handlers  # noqa: F401

logger = logging.getLogger("instructor.auto_client")

if TYPE_CHECKING:
    from botocore.client import BaseClient
else:
    try:
        from botocore.client import BaseClient
    except ImportError:
        BaseClient = None


@overload
def from_bedrock(
    client: BaseClient,
    mode: Mode = Mode.TOOLS,
    async_client: Literal[False] = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_bedrock(
    client: BaseClient,
    mode: Mode = Mode.TOOLS,
    async_client: Literal[True] = True,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_bedrock(
    client: BaseClient,
    mode: Mode = Mode.TOOLS,
    async_client: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Bedrock client using v2 registry.

    Bedrock uses the Converse API through a boto3 BaseClient. This factory supports
    TOOLS and MD_JSON modes, and can wrap calls in an async interface if needed.

    Args:
        client: boto3 Bedrock Runtime client
        mode: The mode to use (defaults to Mode.TOOLS)
        async_client: Whether to return an async Instructor wrapper
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on async_client)

    Raises:
        ModeError: If mode is not registered for Bedrock
        ClientError: If client is not a valid BaseClient or botocore not installed
    """
    from instructor.v2.core.registry import mode_registry, normalize_mode

    if BaseClient is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "botocore is not installed. Install it with: pip install boto3"
        )

    normalized_mode = normalize_mode(Provider.BEDROCK, mode)

    if not mode_registry.is_registered(Provider.BEDROCK, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.BEDROCK)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.BEDROCK.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    mode = normalized_mode

    if not isinstance(client, BaseClient):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            f"Client must be an instance of botocore.client.BaseClient. "
            f"Got: {type(client).__name__}"
        )

    create = client.converse

    if async_client:

        async def async_wrapper(**async_kwargs: Any):
            return create(**async_kwargs)

        patched_create = patch_v2(
            func=async_wrapper,
            provider=Provider.BEDROCK,
            mode=mode,
            default_model=model,
        )
        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.BEDROCK,
            mode=mode,
            **kwargs,
        )

    patched_create = patch_v2(
        func=create,
        provider=Provider.BEDROCK,
        mode=mode,
        default_model=model,
    )

    return Instructor(
        client=client,
        create=patched_create,
        provider=Provider.BEDROCK,
        mode=mode,
        **kwargs,
    )


def _build_from_model(
    *,
    provider: str,
    model_name: str,
    async_client: bool,
    mode: Mode | None,
    kwargs: dict[str, Any],
    provider_info: dict[str, str],
) -> Instructor | AsyncInstructor:
    """Construct and wrap a native Bedrock SDK client for ``from_provider``."""
    try:
        import boto3

        if "region" in kwargs:
            region = kwargs.pop("region")
        else:
            logger.debug(
                "AWS_DEFAULT_REGION is not set. Using default region us-east-1"
            )
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

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

        aws_kwargs["region_name"] = region
        client = boto3.client("bedrock-runtime", **aws_kwargs)

        if mode is None:
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


__all__ = ["from_bedrock"]
