from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
)

from instructor import Mode, Provider
from instructor.v2.core.errors import InstructorRetryException, ResponseParsingError
from instructor.v2.core.hooks import Hooks
from instructor.v2.core.retry import (
    _finalize_parsed_response,
    _initialize_usage,
    retry_async_v2,
    retry_sync_v2,
)
from instructor.v2.dsl.response_list import ListResponse


class Answer(BaseModel):
    value: int


def _validation_error(value: str = "bad") -> ValidationError:
    try:
        Answer.model_validate({"value": value})
    except ValidationError as exc:
        return exc
    raise AssertionError("Expected a validation error")


def test_finalize_parsed_response_wraps_plain_list_and_sets_raw_response() -> None:
    response = object()
    parsed = [Answer(value=1), Answer(value=2)]

    finalized = _finalize_parsed_response(parsed, response)

    assert isinstance(finalized, ListResponse)
    assert list(finalized) == parsed
    assert finalized._raw_response is response


def test_initialize_usage_returns_openai_usage_shape() -> None:
    usage = _initialize_usage(Provider.OPENAI)

    assert usage.completion_tokens == 0
    assert usage.prompt_tokens == 0
    assert usage.total_tokens == 0


def test_retry_sync_v2_returns_raw_result_when_no_response_model() -> None:
    def fake_func(*args: Any, **kwargs: Any) -> str:
        return f"{args[0]}:{kwargs['suffix']}"

    result = retry_sync_v2(
        func=fake_func,
        response_model=None,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=1,
        args=("hello",),
        kwargs={"suffix": "world"},
        strict=True,
        hooks=None,
    )

    assert result == "hello:world"


def test_retry_sync_v2_reasks_after_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    parser_calls: list[str] = []
    emitted: dict[str, list[Any]] = {
        "args": [],
        "responses": [],
        "errors": [],
        "completion_errors": [],
        "last_attempts": [],
    }

    def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        return {"payload": kwargs["messages"][-1]["content"]}

    def fake_parser(**kwargs: Any) -> Answer:
        parser_calls.append(kwargs["response"]["payload"])
        if len(parser_calls) == 1:
            raise _validation_error()
        return Answer(value=7)

    def fake_reask_handler(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response == {"payload": "first"}
        assert isinstance(exception, ValidationError)
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "second"}],
        }

    hooks = Hooks()
    hooks.on("completion:kwargs", lambda **kwargs: emitted["args"].append(kwargs))
    hooks.on(
        "completion:response",
        lambda response: emitted["responses"].append(response),
    )
    hooks.on(
        "parse:error",
        lambda error, **kwargs: emitted["errors"].append((error, kwargs)),
    )
    hooks.on(
        "completion:error",
        lambda error, **kwargs: emitted["completion_errors"].append((error, kwargs)),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **kwargs: emitted["last_attempts"].append((error, kwargs)),
    )

    def no_validate(_provider: Provider, _mode: Mode) -> None:
        return None

    def get_handlers(_provider: Provider, _mode: Mode) -> SimpleNamespace:
        return SimpleNamespace(
            response_parser=fake_parser,
            reask_handler=fake_reask_handler,
        )

    def update_usage(response: Any, total_usage: Any) -> Any:
        assert total_usage == {"tokens": 0}
        return response

    def initialize_usage(_provider: Provider) -> dict[str, int]:
        return {"tokens": 0}

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        no_validate,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        get_handlers,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.update_total_usage",
        update_usage,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        initialize_usage,
    )

    result = retry_sync_v2(
        func=fake_func,
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context={"tenant": "acme"},
        max_retries=Retrying(
            stop=stop_after_attempt(2),
            retry=retry_if_exception_type(ValidationError),
            reraise=True,
        ),
        args=(),
        kwargs={"messages": [{"role": "user", "content": "first"}]},
        strict=True,
        hooks=hooks,
    )

    assert result.value == 7
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["content"] == "second"
    assert parser_calls == ["first", "second"]
    assert len(emitted["args"]) == 2
    assert len(emitted["responses"]) == 2
    assert len(emitted["errors"]) == 1
    assert isinstance(emitted["errors"][0][0], ValidationError)
    assert emitted["errors"][0][1]["attempt_number"] == 1
    assert emitted["completion_errors"] == []
    assert emitted["last_attempts"] == []


def test_retry_sync_v2_emits_last_attempt_metadata_on_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: dict[str, list[Any]] = {"completion_errors": [], "last_attempts": []}

    def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"payload": kwargs["messages"][-1]["content"]}

    def always_fail_parser(**_kwargs: Any) -> Answer:
        raise _validation_error()

    def reask_kwargs(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response["payload"] in {"first", "retry"}
        assert isinstance(exception, ValidationError)
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "retry"}],
        }

    hooks = Hooks()
    hooks.on(
        "completion:error",
        lambda error, **kwargs: emitted["completion_errors"].append((error, kwargs)),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **kwargs: emitted["last_attempts"].append((error, kwargs)),
    )

    def no_validate(_provider: Provider, _mode: Mode) -> None:
        return None

    def get_handlers(_provider: Provider, _mode: Mode) -> SimpleNamespace:
        return SimpleNamespace(
            response_parser=always_fail_parser,
            reask_handler=reask_kwargs,
        )

    def update_usage(response: Any, total_usage: Any) -> Any:
        assert total_usage == {"tokens": 0}
        return response

    def initialize_usage(_provider: Provider) -> dict[str, int]:
        return {"tokens": 0}

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        no_validate,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        get_handlers,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.update_total_usage",
        update_usage,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        initialize_usage,
    )

    with pytest.raises(InstructorRetryException):
        retry_sync_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=1,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=hooks,
        )

    assert emitted["completion_errors"] == []
    assert len(emitted["last_attempts"]) == 1
    assert emitted["last_attempts"][0][1] == {
        "attempt_number": 2,
        "max_attempts": 2,
        "is_last_attempt": True,
    }


def test_retry_sync_v2_marks_api_error_as_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    emitted: dict[str, list[Any]] = {"completion_errors": [], "last_attempts": []}

    def fake_func(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    hooks = Hooks()
    hooks.on(
        "completion:error",
        lambda error, **kwargs: emitted["completion_errors"].append((error, kwargs)),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **kwargs: emitted["last_attempts"].append((error, kwargs)),
    )

    def no_validate(_provider: Provider, _mode: Mode) -> None:
        return None

    def get_handlers(_provider: Provider, _mode: Mode) -> SimpleNamespace:
        return SimpleNamespace(
            response_parser=lambda **_kwargs: Answer(value=1),
            reask_handler=lambda **kwargs: kwargs,
        )

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        no_validate,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        get_handlers,
    )

    with pytest.raises(InstructorRetryException) as exc_info:
        retry_sync_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=3,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=hooks,
        )

    assert calls == 1
    assert exc_info.value.n_attempts == 1
    assert len(emitted["completion_errors"]) == 1
    assert emitted["completion_errors"][0][1] == {
        "attempt_number": 1,
        "max_attempts": 4,
        "is_last_attempt": True,
    }
    assert len(emitted["last_attempts"]) == 1
    assert emitted["last_attempts"][0][1] == {
        "attempt_number": 1,
        "max_attempts": 4,
        "is_last_attempt": True,
    }


def test_retry_sync_v2_integer_max_retries_counts_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_calls = 0

    def fake_func(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"payload": "ok"}

    def fake_parser(**_kwargs: Any) -> Answer:
        nonlocal parser_calls
        parser_calls += 1
        if parser_calls == 1:
            raise _validation_error()
        return Answer(value=42)

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=fake_parser,
            reask_handler=lambda **call: call["kwargs"],
        ),
    )

    result = retry_sync_v2(
        func=fake_func,
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=1,
        args=(),
        kwargs={},
        strict=True,
        hooks=None,
    )

    assert result == Answer(value=42)
    assert parser_calls == 2


@pytest.mark.parametrize(
    "parse_error",
    [
        json.JSONDecodeError("bad json", "{", 1),
        pytest.param(
            ResponseParsingError("missing tool call"),
            id="response-parsing-error",
        ),
    ],
)
def test_retry_sync_v2_retries_parse_errors(
    monkeypatch: pytest.MonkeyPatch,
    parse_error: Exception,
) -> None:
    parser_calls = 0

    def fake_parser(**_kwargs: Any) -> Answer:
        nonlocal parser_calls
        parser_calls += 1
        if parser_calls == 1:
            raise parse_error
        return Answer(value=42)

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=fake_parser,
            reask_handler=lambda **call: call["kwargs"],
        ),
    )

    result = retry_sync_v2(
        func=lambda *_args, **_kwargs: {"payload": "ok"},
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.JSON,
        context=None,
        max_retries=1,
        args=(),
        kwargs={},
        strict=True,
        hooks=None,
    )

    assert result == Answer(value=42)
    assert parser_calls == 2


def test_retry_sync_v2_reports_terminal_api_error_after_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_func(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("terminal API failure")
        return {"payload": "invalid"}

    def always_fail_parser(**_kwargs: Any) -> Answer:
        raise _validation_error()

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=always_fail_parser,
            reask_handler=lambda **call: call["kwargs"],
        ),
    )

    with pytest.raises(
        InstructorRetryException, match="terminal API failure"
    ) as exc_info:
        retry_sync_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=1,
            args=(),
            kwargs={},
            strict=True,
            hooks=None,
        )

    assert exc_info.value.n_attempts == 2


def test_retry_sync_v2_raises_instructor_retry_exception_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"payload": kwargs["messages"][-1]["content"]}

    def always_fail_parser(**_kwargs: Any) -> Answer:
        raise _validation_error()

    def reask_kwargs(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response["payload"] in {"first", "retry"}
        assert isinstance(exception, ValidationError)
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "retry"}],
        }

    def no_validate(_provider: Provider, _mode: Mode) -> None:
        return None

    def get_handlers(_provider: Provider, _mode: Mode) -> SimpleNamespace:
        return SimpleNamespace(
            response_parser=always_fail_parser,
            reask_handler=reask_kwargs,
        )

    def update_usage(response: Any, total_usage: Any) -> Any:
        assert total_usage == {"tokens": 0}
        return response

    def initialize_usage(_provider: Provider) -> dict[str, int]:
        return {"tokens": 0}

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        no_validate,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        get_handlers,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.update_total_usage",
        update_usage,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        initialize_usage,
    )

    with pytest.raises(InstructorRetryException) as exc_info:
        retry_sync_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=Retrying(
                stop=stop_after_attempt(2),
                retry=retry_if_exception_type(ValidationError),
                reraise=True,
            ),
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=None,
        )

    error = exc_info.value
    assert error.n_attempts == 2
    assert error.last_completion == {"payload": "retry"}
    assert error.create_kwargs is not None
    assert error.create_kwargs["messages"][-1]["content"] == "retry"
    assert len(error.failed_attempts or []) == 2


@pytest.mark.asyncio
async def test_retry_async_v2_reasks_after_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_calls: list[str] = []

    async def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"payload": kwargs["messages"][-1]["content"]}

    def fake_parser(**kwargs: Any) -> Answer:
        parser_calls.append(kwargs["response"]["payload"])
        if len(parser_calls) == 1:
            raise _validation_error()
        return Answer(value=9)

    def reask_kwargs(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response["payload"] in {"first", "retry"}
        assert isinstance(exception, ValidationError)
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "retry"}],
        }

    def no_validate(_provider: Provider, _mode: Mode) -> None:
        return None

    def get_handlers(_provider: Provider, _mode: Mode) -> SimpleNamespace:
        return SimpleNamespace(
            response_parser=fake_parser,
            reask_handler=reask_kwargs,
        )

    def update_usage(response: Any, total_usage: Any) -> Any:
        assert total_usage == {"tokens": 0}
        return response

    def initialize_usage(_provider: Provider) -> dict[str, int]:
        return {"tokens": 0}

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        no_validate,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        get_handlers,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.update_total_usage",
        update_usage,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        initialize_usage,
    )

    result = await retry_async_v2(
        func=fake_func,
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=AsyncRetrying(
            stop=stop_after_attempt(2),
            retry=retry_if_exception_type(ValidationError),
            reraise=True,
        ),
        args=(),
        kwargs={"messages": [{"role": "user", "content": "first"}]},
        strict=True,
        hooks=None,
    )

    assert result == Answer(value=9)
    assert parser_calls == ["first", "retry"]


@pytest.mark.asyncio
async def test_retry_async_v2_marks_api_error_as_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    emitted: dict[str, list[Any]] = {"completion_errors": [], "last_attempts": []}

    async def fake_func(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    hooks = Hooks()
    hooks.on(
        "completion:error",
        lambda error, **kwargs: emitted["completion_errors"].append((error, kwargs)),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **kwargs: emitted["last_attempts"].append((error, kwargs)),
    )

    def no_validate(_provider: Provider, _mode: Mode) -> None:
        return None

    def get_handlers(_provider: Provider, _mode: Mode) -> SimpleNamespace:
        return SimpleNamespace(
            response_parser=lambda **_kwargs: Answer(value=1),
            reask_handler=lambda **kwargs: kwargs,
        )

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        no_validate,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        get_handlers,
    )

    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=3,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=hooks,
        )

    assert calls == 1
    assert exc_info.value.n_attempts == 1
    assert len(emitted["completion_errors"]) == 1
    assert emitted["completion_errors"][0][1] == {
        "attempt_number": 1,
        "max_attempts": 4,
        "is_last_attempt": True,
    }
    assert len(emitted["last_attempts"]) == 1
    assert emitted["last_attempts"][0][1] == {
        "attempt_number": 1,
        "max_attempts": 4,
        "is_last_attempt": True,
    }


@pytest.mark.asyncio
async def test_retry_async_v2_integer_max_retries_counts_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_calls = 0

    async def fake_func(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"payload": "ok"}

    def fake_parser(**_kwargs: Any) -> Answer:
        nonlocal parser_calls
        parser_calls += 1
        if parser_calls == 1:
            raise _validation_error()
        return Answer(value=42)

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=fake_parser,
            reask_handler=lambda **call: call["kwargs"],
        ),
    )

    result = await retry_async_v2(
        func=fake_func,
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=1,
        args=(),
        kwargs={},
        strict=True,
        hooks=None,
    )

    assert result == Answer(value=42)
    assert parser_calls == 2


@pytest.mark.asyncio
async def test_retry_async_v2_retries_json_decode_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_calls = 0

    async def fake_func(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"payload": "ok"}

    def fake_parser(**_kwargs: Any) -> Answer:
        nonlocal parser_calls
        parser_calls += 1
        if parser_calls == 1:
            raise json.JSONDecodeError("bad json", "{", 1)
        return Answer(value=42)

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=fake_parser,
            reask_handler=lambda **call: call["kwargs"],
        ),
    )

    result = await retry_async_v2(
        func=fake_func,
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.JSON,
        context=None,
        max_retries=1,
        args=(),
        kwargs={},
        strict=True,
        hooks=None,
    )

    assert result == Answer(value=42)
    assert parser_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream_kwargs", "expected_stream"),
    [({}, False), ({"stream": True}, True)],
    ids=["default-stream", "explicit-stream"],
)
async def test_retry_async_v2_preserves_parser_response_and_usage_contract(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stream_kwargs: dict[str, bool],
    expected_stream: bool,
) -> None:
    response = SimpleNamespace(payload="answer")
    context = {"tenant": "async"}
    usage = {"tokens": 0}
    seen_responses: list[Any] = []
    seen_updates: list[tuple[Any, Any]] = []

    def parser(**kwargs: Any) -> Answer:
        assert kwargs == {
            "response": response,
            "response_model": Answer,
            "validation_context": context,
            "strict": False,
            "stream": expected_stream,
            "is_async": True,
        }
        return Answer(value=11)

    def initialize_usage(provider: Provider) -> dict[str, int]:
        assert provider is Provider.OPENAI
        return usage

    def update_usage(response: Any, total_usage: Any) -> Any:
        seen_updates.append((response, total_usage))
        return response

    hooks = Hooks()
    hooks.on("completion:response", lambda value: seen_responses.append(value))
    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=parser,
            reask_handler=lambda **call: call["kwargs"],
        ),
    )
    monkeypatch.setattr("instructor.v2.core.retry._initialize_usage", initialize_usage)
    monkeypatch.setattr("instructor.v2.core.retry.update_total_usage", update_usage)
    caplog.set_level(logging.DEBUG, logger="instructor.v2.retry")

    async def create(**_kwargs: Any) -> SimpleNamespace:
        return response

    result = await retry_async_v2(
        func=create,
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.JSON,
        context=context,
        max_retries=0,
        args=(),
        kwargs={"messages": [], **stream_kwargs},
        strict=False,
        hooks=hooks,
    )

    assert result == Answer(value=11)
    assert getattr(result, "_raw_response", None) is response
    assert getattr(result, "_total_usage", None) is None
    assert seen_responses == [response]
    assert seen_updates == [(response, usage)]
    assert "Successfully parsed response on attempt 1" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_retries", "timeout"),
    [(0, None), (3, 0)],
    ids=["zero-retries", "expired-timeout"],
)
async def test_retry_async_v2_stops_after_one_attempt_when_retry_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    max_retries: int,
    timeout: int | None,
) -> None:
    calls = 0
    response = {"payload": "invalid"}

    async def create(**_kwargs: Any) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return response

    def retrying(*, stop: Any, retry: Any, reraise: bool) -> AsyncRetrying:
        return AsyncRetrying(stop=stop, retry=retry, reraise=reraise)

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=lambda **_kwargs: Answer.model_validate({"value": "bad"}),
            reask_handler=lambda **call: call["kwargs"],
        ),
    )
    monkeypatch.setattr("instructor.v2.core.retry.AsyncRetrying", retrying)
    caplog.set_level(logging.DEBUG, logger="instructor.v2.retry")

    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=max_retries,
            args=(),
            kwargs={"messages": [], "timeout": timeout},
            strict=True,
            hooks=None,
        )

    assert calls == 1
    assert exc_info.value.n_attempts == 1
    assert exc_info.value.last_completion is response
    assert "Validation error on attempt 1" in caplog.text
    assert "Max retries exceeded. Total attempts: 1" in caplog.text


@pytest.mark.asyncio
async def test_retry_async_v2_reports_each_validation_api_error_and_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    errors = [_validation_error(), _validation_error()]
    completion_errors: list[tuple[Exception, dict[str, Any]]] = []
    last_attempts: list[tuple[Exception, dict[str, Any]]] = []
    calls = 0

    async def create(**_kwargs: Any) -> dict[str, str]:
        nonlocal calls
        error = errors[calls]
        calls += 1
        raise error

    hooks = Hooks()
    hooks.on(
        "completion:error",
        lambda error, **metadata: completion_errors.append((error, metadata)),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **metadata: last_attempts.append((error, metadata)),
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=lambda **_kwargs: Answer(value=1),
            reask_handler=lambda **call: call["kwargs"],
        ),
    )
    caplog.set_level(logging.ERROR, logger="instructor.v2.retry")

    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=1,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "answer"}]},
            strict=True,
            hooks=hooks,
        )

    assert calls == 2
    assert [error for error, _ in completion_errors] == errors
    assert [metadata for _, metadata in completion_errors] == [
        {"attempt_number": 1, "max_attempts": 2, "is_last_attempt": False},
        {"attempt_number": 2, "max_attempts": 2, "is_last_attempt": True},
    ]
    assert last_attempts == [
        (
            errors[-1],
            {"attempt_number": 2, "max_attempts": 2, "is_last_attempt": True},
        )
    ]
    assert exc_info.value.n_attempts == 2
    assert exc_info.value.failed_attempts == []
    assert (
        f"API call failed on attempt 1 ({type(errors[0]).__name__})" in caplog.messages
    )
    assert (
        f"API call failed on attempt 2 ({type(errors[1]).__name__})" in caplog.messages
    )
    assert (
        f"Max retries exceeded. Total attempts: 2 ({type(errors[-1]).__name__})"
        in caplog.messages
    )
    assert str(errors[0]) not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback", [False, True], ids=["reraise", "return-from-policy"]
)
async def test_retry_async_v2_keeps_every_failed_completion_on_exhaustion(
    callback: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses = [
        {"payload": "first"},
        {"payload": "second"},
        {"payload": "third"},
    ]
    errors = [_validation_error(f"bad-{n}") for n in range(3)]
    parse_errors: list[tuple[Exception, dict[str, Any]]] = []
    completion_responses: list[Any] = []
    updates: list[tuple[Any, Any]] = []
    usage = {"tokens": 7}
    calls = 0
    parser_calls = 0

    async def create(**_kwargs: Any) -> dict[str, str]:
        nonlocal calls
        response = responses[calls]
        calls += 1
        return response

    def parser(**_kwargs: Any) -> Answer:
        nonlocal parser_calls
        error = errors[parser_calls]
        parser_calls += 1
        raise error

    def reask(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response is responses[parser_calls - 1]
        assert exception is errors[parser_calls - 1]
        return {
            **kwargs,
            "messages": [
                *kwargs["messages"],
                {"role": "user", "content": f"retry-{parser_calls}"},
            ],
        }

    hooks = Hooks()
    hooks.on(
        "parse:error",
        lambda error, **metadata: parse_errors.append((error, metadata)),
    )
    hooks.on(
        "completion:response",
        lambda response: completion_responses.append(response),
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=parser,
            reask_handler=reask,
        ),
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        lambda provider: usage if provider is Provider.OPENAI else None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.update_total_usage",
        lambda response, total_usage: updates.append((response, total_usage)),
    )
    caplog.set_level(logging.DEBUG, logger="instructor.v2.retry")

    policy = (
        AsyncRetrying(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(ValidationError),
            retry_error_callback=lambda _state: None,
        )
        if callback
        else 2
    )
    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=policy,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=hooks,
        )

    error = exc_info.value
    assert calls == parser_calls == 3
    assert completion_responses == responses
    assert updates == [(response, usage) for response in responses]
    assert [caught for caught, _ in parse_errors] == errors
    assert [metadata for _, metadata in parse_errors] == [
        {
            "attempt_number": attempt,
            "max_attempts": None if callback else 3,
            "is_last_attempt": attempt == 3 and not callback,
        }
        for attempt in (1, 2, 3)
    ]
    assert error.args[0] == str(errors[-1])
    assert error.n_attempts == 3
    assert error.last_completion is responses[-1]
    assert error.total_usage is usage
    assert error.messages == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "retry-1"},
        {"role": "user", "content": "retry-2"},
        {"role": "user", "content": "retry-3"},
    ]
    assert error.create_kwargs is not None
    assert error.create_kwargs["messages"] == error.messages
    assert error.failed_attempts is not None
    assert len(error.failed_attempts) == 3
    assert [attempt.attempt_number for attempt in error.failed_attempts] == [1, 2, 3]
    assert [attempt.exception for attempt in error.failed_attempts] == errors
    assert [attempt.completion for attempt in error.failed_attempts] == responses
    assert (
        f"Validation error on attempt 1 ({type(errors[0]).__name__})" in caplog.messages
    )
    assert (
        f"Validation error on attempt 2 ({type(errors[1]).__name__})" in caplog.messages
    )
    assert (
        f"Validation error on attempt 3 ({type(errors[2]).__name__})" in caplog.messages
    )
    if not callback:
        assert (
            f"Max retries exceeded. Total attempts: 3 ({type(errors[-1]).__name__})"
            in caplog.messages
        )


class NoAttemptsAsyncRetrying(AsyncRetrying):
    def __aiter__(self) -> NoAttemptsAsyncRetrying:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_retry_async_v2_reports_an_empty_retry_policy_without_losing_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    usage = {"tokens": 0}
    messages = [{"role": "user", "content": "unused"}]

    async def create(**_kwargs: Any) -> dict[str, str]:
        raise AssertionError("an empty retry policy must not call the provider")

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=lambda **_kwargs: Answer(value=1),
            reask_handler=lambda **call: call["kwargs"],
        ),
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        lambda provider: usage if provider is Provider.OPENAI else None,
    )
    caplog.set_level(logging.ERROR, logger="instructor.v2.retry")

    with pytest.raises(InstructorRetryException, match=r"^Unknown error$") as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=NoAttemptsAsyncRetrying(),
            args=(),
            kwargs={"messages": messages},
            strict=True,
            hooks=None,
        )

    error = exc_info.value
    assert error.n_attempts == 0
    assert error.last_completion is None
    assert error.total_usage is usage
    assert error.messages == messages
    assert error.create_kwargs == {"messages": messages}
    assert error.failed_attempts == []
    assert "Unexpected code path in retry_async_v2" in [
        record.getMessage() for record in caplog.records
    ]
