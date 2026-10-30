"""Tests for ZhipuAI and Z.ai provider client factories and detection."""

from __future__ import annotations

from unittest.mock import MagicMock

import openai
import pytest
from pydantic import BaseModel

from instructor import Mode, Provider, from_zai, from_zhipuai
from instructor.v2.auto_client import _build_zai, _build_zhipuai
from instructor.v2.core.errors import ClientError, ConfigurationError
from instructor.v2.core.providers import get_provider


class CityInfo(BaseModel):
    name: str
    country: str


def test_get_provider_detects_zhipuai_and_zai():
    assert get_provider("https://open.bigmodel.cn/api/paas/v4/") == Provider.ZHIPUAI
    assert get_provider("https://api.z.ai/api/paas/v4/") == Provider.ZAI
    assert get_provider("https://custom.zhipuai.cn/v1") == Provider.ZHIPUAI
    assert get_provider("https://z.ai/v1") == Provider.ZAI


def test_from_zhipuai_client_validation():
    with pytest.raises(ClientError, match="must be an instance"):
        from_zhipuai(12345)  # type: ignore

    client = openai.OpenAI(
        api_key="mock-key",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )
    instr = from_zhipuai(client, mode=Mode.TOOLS)
    assert instr.provider == Provider.ZHIPUAI
    assert instr.mode == Mode.TOOLS


def test_from_zai_client_validation():
    with pytest.raises(ClientError, match="must be an instance"):
        from_zai(12345)  # type: ignore

    client = openai.OpenAI(
        api_key="mock-key",
        base_url="https://api.z.ai/api/paas/v4/",
    )
    instr = from_zai(client, mode=Mode.TOOLS)
    assert instr.provider == Provider.ZAI
    assert instr.mode == Mode.TOOLS


def test_build_zhipuai_api_key_fallbacks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    provider_info = {"provider": "zhipuai", "model": "glm-5.3-flash"}

    # Missing all keys -> ConfigurationError
    with pytest.raises(ConfigurationError, match="ZHIPUAI_API_KEY is not set"):
        _build_zhipuai(
            provider="zhipuai",
            model_name="glm-5.3-flash",
            async_client=False,
            mode=None,
            api_key=None,
            kwargs={},
            provider_info=provider_info,
        )

    # Fallback to ZAI_API_KEY
    monkeypatch.setenv("ZAI_API_KEY", "zai-key")
    client_zai = _build_zhipuai(
        provider="zhipuai",
        model_name="glm-5.3-flash",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info=provider_info,
    )
    assert client_zai.client.api_key == "zai-key"

    # Fallback to BIGMODEL_API_KEY
    monkeypatch.setenv("BIGMODEL_API_KEY", "bigmodel-key")
    client_bm = _build_zhipuai(
        provider="zhipuai",
        model_name="glm-5.3-flash",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info=provider_info,
    )
    assert client_bm.client.api_key == "bigmodel-key"

    # Precedence of ZHIPUAI_API_KEY
    monkeypatch.setenv("ZHIPUAI_API_KEY", "zhipuai-key")
    client_zp = _build_zhipuai(
        provider="zhipuai",
        model_name="glm-5.3-flash",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info=provider_info,
    )
    assert client_zp.client.api_key == "zhipuai-key"


def test_build_zai_api_key_fallbacks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    provider_info = {"provider": "zai", "model": "glm-5.3-flash"}

    # Missing all keys -> ConfigurationError
    with pytest.raises(ConfigurationError, match="ZAI_API_KEY is not set"):
        _build_zai(
            provider="zai",
            model_name="glm-5.3-flash",
            async_client=False,
            mode=None,
            api_key=None,
            kwargs={},
            provider_info=provider_info,
        )

    # Fallback to ZHIPUAI_API_KEY
    monkeypatch.setenv("ZHIPUAI_API_KEY", "zhipu-key")
    client_zp = _build_zai(
        provider="zai",
        model_name="glm-5.3-flash",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info=provider_info,
    )
    assert client_zp.client.api_key == "zhipu-key"

    # Precedence of ZAI_API_KEY
    monkeypatch.setenv("ZAI_API_KEY", "zai-primary-key")
    client_zai = _build_zai(
        provider="zai",
        model_name="glm-5.3-flash",
        async_client=False,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info=provider_info,
    )
    assert client_zai.client.api_key == "zai-primary-key"


def test_zhipuai_mock_completion_parsing():
    raw_client = openai.OpenAI(
        api_key="mock-key",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_zhipu_1"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "CityInfo"
    mock_tool_call.function.arguments = '{"name": "Beijing", "country": "China"}'

    mock_choice = MagicMock()
    mock_choice.message.tool_calls = [mock_tool_call]
    mock_choice.message.content = None
    mock_choice.message.refusal = None

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    raw_client.chat.completions.create = MagicMock(return_value=mock_response)

    client = from_zhipuai(raw_client, mode=Mode.TOOLS)
    result = client.chat.completions.create(
        model="glm-5.3-flash",
        response_model=CityInfo,
        messages=[{"role": "user", "content": "What is the capital of China?"}],
    )

    assert isinstance(result, CityInfo)
    assert result.name == "Beijing"
    assert result.country == "China"
