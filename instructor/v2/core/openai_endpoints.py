"""Endpoint and capability declarations for OpenAI-compatible chat providers."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider


@dataclass(frozen=True)
class OpenAIEndpoint:
    base_url: str
    key_variables: tuple[str, ...]
    modes: tuple[Mode, ...]
    default_mode: Mode = Mode.TOOLS


OPENAI_ENDPOINTS = {
    Provider.ZHIPUAI: OpenAIEndpoint(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        key_variables=("ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"),
        modes=(
            Mode.TOOLS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.PARALLEL_TOOLS,
        ),
    ),
    Provider.ZAI: OpenAIEndpoint(
        base_url="https://api.z.ai/api/paas/v4/",
        key_variables=("ZAI_API_KEY",),
        modes=(
            Mode.TOOLS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.PARALLEL_TOOLS,
        ),
    ),
    Provider.HUBRIS: OpenAIEndpoint(
        base_url="https://api.hubris.pw/v1",
        key_variables=("HUBRIS_API_KEY",),
        modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
    ),
    Provider.ATLASCLOUD: OpenAIEndpoint(
        base_url="https://api.atlascloud.ai/v1",
        key_variables=("ATLASCLOUD_API_KEY",),
        modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
    ),
    Provider.REQUESTY: OpenAIEndpoint(
        base_url="https://router.requesty.ai/v1",
        key_variables=("REQUESTY_API_KEY",),
        modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
    ),
    Provider.SARVAM: OpenAIEndpoint(
        base_url="https://api.sarvam.ai/v2",
        key_variables=("SARVAM_API_KEY",),
        modes=(Mode.TOOLS, Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON),
        default_mode=Mode.MD_JSON,
    ),
    Provider.ORCAROUTER: OpenAIEndpoint(
        base_url="https://api.orcarouter.ai/v1",
        key_variables=("ORCAROUTER_API_KEY",),
        modes=(Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS),
    ),
    Provider.MINIMAX: OpenAIEndpoint(
        base_url="https://api.minimax.io/v1",
        key_variables=("MINIMAX_API_KEY",),
        modes=(Mode.TOOLS, Mode.MD_JSON),
    ),
    Provider.MANIFEST: OpenAIEndpoint(
        base_url="https://app.manifest.build/v1",
        key_variables=("MANIFEST_API_KEY",),
        modes=(Mode.TOOLS, Mode.JSON, Mode.MD_JSON),
    ),
}

ENDPOINT_HOSTS = {
    urlsplit(endpoint.base_url).hostname: provider
    for provider, endpoint in OPENAI_ENDPOINTS.items()
}


def endpoint_providers(mode: Mode) -> list[Provider]:
    return [
        provider
        for provider, endpoint in OPENAI_ENDPOINTS.items()
        if mode in endpoint.modes
    ]
