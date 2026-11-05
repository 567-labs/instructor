import importlib
import json
import logging
import warnings
from collections.abc import Callable, Iterator
from types import FunctionType
from typing import Any, Literal, cast, get_args

import pytest
from openai import OpenAI
from pydantic import BaseModel

from instructor.distil import (
    FinetuneFormat,
    Instructions,
    format_function,
    get_signature_from_fn,
    is_return_type_base_model_or_instance,
)


class UserRecord(BaseModel):
    name: str
    age: int


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> BaseModel:
        self.calls.append(kwargs)
        return kwargs["response_model"](name="dispatched", age=41)


class RecordingClient:
    def __init__(self) -> None:
        self.completions = RecordingCompletions()
        self.chat = self


@pytest.fixture(autouse=True)
def restore_distil_state() -> Iterator[None]:
    names = (
        "coverage.distil.messages",
        "coverage.distil.raw",
        "coverage.distil.dispatch",
        "coverage.distil.invalid",
        "coverage.distil.kwargs",
    )
    loggers = [logging.getLogger(name) for name in names]
    state = [(logger.handlers[:], logger.level, logger.propagate) for logger in loggers]
    format_function.cache_clear()

    yield

    format_function.cache_clear()
    for logger, (handlers, level, propagate) in zip(loggers, state):
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def lookup_user(user_id: int, region: str = "us") -> UserRecord:
    """Look up a user in a region."""
    return UserRecord(name=f"{region}-{user_id}", age=user_id)


def lookup_without_doc(user_id: int) -> UserRecord:
    return UserRecord(name=str(user_id), age=user_id)


def lookup_user_with_forward_ref(user_id: int, region: str = "us") -> "UserRecord":
    return lookup_user(user_id, region)


def fetch_user(user_id: int, active: bool = True) -> UserRecord:
    raise AssertionError(f"dispatch called the function body: {user_id}, {active}")


def fetch_user_with_forward_ref(user_id: int, active: bool = True) -> "UserRecord":
    raise AssertionError(f"dispatch called the function body: {user_id}, {active}")


def test_distil_function_formatters_keep_signature_source_and_help_text() -> None:
    documented = get_signature_from_fn(lookup_user)
    assert documented.startswith(
        "def lookup_user(user_id: int, region: str = 'us') -> "
    )
    assert '"""\nLook up a user in a region.\n"""' in documented

    assert get_signature_from_fn(lookup_without_doc).endswith("\n")

    formatted = format_function(lookup_user)
    assert "def lookup_user(" in formatted
    assert "Look up a user in a region." in formatted
    assert 'return UserRecord(name=f"{region}-{user_id}", age=user_id)' in formatted
    assert "Look up a user" not in format_function(lookup_without_doc)


@pytest.mark.parametrize("lookup", [lookup_user, lookup_user_with_forward_ref])
def test_distil_return_type_checks_accept_models_and_reject_invalid_hints(
    lookup: Callable[..., UserRecord],
) -> None:
    assert is_return_type_base_model_or_instance(lookup)

    def returns_number() -> int:
        return 1

    def returns_number_with_forward_ref() -> "int":
        return 1

    def no_hint():
        return UserRecord(name="unknown", age=0)

    assert not is_return_type_base_model_or_instance(returns_number)
    assert not is_return_type_base_model_or_instance(returns_number_with_forward_ref)
    with pytest.raises(AssertionError, match="Must have a return type hint"):
        is_return_type_base_model_or_instance(no_hint)


@pytest.mark.parametrize("lookup", [lookup_user, lookup_user_with_forward_ref])
def test_distil_decorator_tracks_message_and_raw_training_examples(
    lookup: Callable[..., UserRecord],
) -> None:
    assert isinstance(lookup, FunctionType)
    message_handler = RecordingHandler()
    message_instructions = Instructions(
        name="coverage.distil.messages",
        id="messages-id",
        log_handlers=[message_handler],
        openai_client=cast(OpenAI, RecordingClient()),
    )
    message_instructions.logger.setLevel(logging.INFO)
    message_instructions.logger.propagate = False

    decorated = message_instructions.distil(lookup)
    result = decorated(7, region="eu")

    assert result == UserRecord(name="eu-7", age=7)
    assert message_instructions.id == "messages-id"
    assert message_instructions.unique_id
    logged = json.loads(message_handler.messages[-1])
    assert (
        logged["messages"][1]["content"]
        == f'Return `{lookup.__name__}(7, region="eu")`'
    )
    assert logged["messages"][2]["function_call"] == {
        "name": "UserRecord",
        "arguments": '{\n  "name": "eu-7",\n  "age": 7\n}',
    }
    assert logged["functions"][0]["name"] == "UserRecord"

    raw_handler = RecordingHandler()
    raw_instructions = Instructions(
        name="coverage.distil.raw",
        log_handlers=[raw_handler],
        finetune_format=FinetuneFormat.RAW,
        include_code_body=True,
        openai_client=cast(OpenAI, RecordingClient()),
    )
    raw_instructions.logger.setLevel(logging.INFO)
    raw_instructions.logger.propagate = False

    raw_decorated = raw_instructions.distil(
        name="find_user", fine_tune_format=FinetuneFormat.RAW
    )(lookup)
    raw_result = raw_decorated(3)

    assert raw_result == UserRecord(name="us-3", age=3)
    raw = json.loads(raw_handler.messages[-1])
    assert raw["fn_name"] == "find_user"
    assert f"def {lookup.__name__}(" in raw["fn_repr"]
    assert raw["args"] == [3]
    assert raw["kwargs"] == {}
    assert raw["resp"] == {"name": "us-3", "age": 3}
    assert raw["schema"]["properties"]["name"]["type"] == "string"


@pytest.mark.parametrize("fetch", [fetch_user, fetch_user_with_forward_ref])
def test_distil_dispatch_sends_model_schema_and_never_calls_original(
    fetch: Callable[..., UserRecord],
) -> None:
    assert isinstance(fetch, FunctionType)
    client = RecordingClient()
    instructions = Instructions(
        name="coverage.distil.dispatch",
        openai_client=cast(OpenAI, client),
        include_code_body=True,
    )

    decorated = instructions.distil(name="fetch", mode="dispatch", model="gpt-4o-mini")(
        fetch
    )
    response = decorated(12, active=False)

    assert response == UserRecord(name="dispatched", age=41)
    assert len(client.completions.calls) == 1
    call = client.completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["response_model"] is UserRecord
    assert f"def {fetch.__name__}(" in call["messages"][0]["content"]
    assert call["messages"][1]["content"] == "Return `fetch(12, active=false)`"


def test_distil_rejects_invalid_modes_and_return_models() -> None:
    instructions = Instructions(
        name="coverage.distil.invalid",
        openai_client=cast(OpenAI, RecordingClient()),
    )

    with pytest.raises(AssertionError, match="Must be in"):
        instructions.distil(mode=cast(Literal["distil", "dispatch"], "unknown"))

    def returns_number() -> int:
        return 1

    with pytest.raises(AssertionError, match="must subclass"):
        instructions.distil(returns_number)


def test_distil_openai_kwargs_handle_empty_calls_and_keyword_only_calls() -> None:
    instructions = Instructions(
        name="coverage.distil.kwargs",
        openai_client=cast(OpenAI, RecordingClient()),
    )

    empty = instructions.openai_kwargs("lookup", lookup_without_doc, (), {}, UserRecord)
    keywords = instructions.openai_kwargs(
        "lookup", lookup_user, (), {"region": "apac"}, UserRecord
    )

    assert empty["messages"][1]["content"] == "Return `lookup()`"
    assert keywords["messages"][1]["content"] == 'Return `lookup(region="apac")`'
    system_content = empty["messages"][0]["content"]
    assert isinstance(system_content, str)
    assert "def lookup_without_doc(" in system_content


def test_legacy_type_and_dsl_modules_keep_their_v2_exports() -> None:
    model_names = importlib.import_module("instructor._types._alias").ModelNames
    assert "gpt-4o" in get_args(model_names)
    assert "text-embedding-3-large" in get_args(model_names)

    exports = [
        ("citation", "CitationMixin"),
        ("json_tracker", "JsonCompleteness"),
        ("maybe", "Maybe"),
        ("parallel", "ParallelModel"),
        ("simple_type", "is_simple_type"),
    ]
    for module_name, symbol in exports:
        legacy = importlib.import_module(f"instructor.dsl.{module_name}")
        current = importlib.import_module(f"instructor.v2.dsl.{module_name}")
        assert getattr(legacy, symbol) is getattr(current, symbol)

    legacy_calls = importlib.import_module("instructor.function_calls")
    current_calls = importlib.import_module("instructor.processing.function_calls")
    assert legacy_calls.ResponseSchema is current_calls.ResponseSchema


def test_legacy_exceptions_and_hooks_warn_and_keep_their_exports() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exceptions = importlib.import_module("instructor.exceptions")
        exceptions = importlib.reload(exceptions)

    assert any("instructor.exceptions" in str(item.message) for item in caught)
    core_exceptions = importlib.import_module("instructor.core.exceptions")
    assert (
        exceptions.InstructorRetryException is core_exceptions.InstructorRetryException
    )
    assert "InstructorRetryException" in exceptions.__all__

    hooks = importlib.import_module("instructor.hooks")
    with pytest.warns(DeprecationWarning, match="instructor.core.hooks.Hooks"):
        hook_type = hooks.__getattr__("Hooks")
    assert hook_type is importlib.import_module("instructor.core.hooks").Hooks

    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(AttributeError, match="has no attribute 'missing_hook'"),
    ):
        hooks.__getattr__("missing_hook")


def test_legacy_provider_and_utility_lazy_exports_have_clear_missing_errors() -> None:
    providers = importlib.import_module("instructor.providers")
    providers.__dict__.pop("from_openai", None)
    exported = providers.__getattr__("from_openai")
    current = importlib.import_module("instructor.v2.providers.openai.client")
    assert exported is current.from_openai
    assert providers.from_openai is exported

    with pytest.raises(AttributeError, match="missing_provider"):
        providers.__getattr__("missing_provider")

    compat = importlib.import_module("instructor.providers._compat")
    resolved = compat.resolve_provider_attr(
        "openai", ("handlers", "client"), "from_openai"
    )
    assert resolved is current.from_openai
    assert compat.make_getattr("openai", ("client",))("from_openai") is resolved
    with pytest.raises(AttributeError, match="missing_factory"):
        compat.resolve_provider_attr("openai", ("handlers",), "missing_factory")

    utils = importlib.import_module("instructor.utils")
    gemini_utils = importlib.import_module("instructor.v2.providers.gemini.utils")
    anthropic_handlers = importlib.import_module(
        "instructor.v2.providers.anthropic.handlers"
    )
    assert (
        utils.__getattr__("map_to_gemini_function_schema")
        is gemini_utils.map_to_gemini_function_schema
    )
    assert (
        utils.__getattr__("combine_system_messages")
        is anthropic_handlers.combine_system_messages
    )
    with pytest.raises(AttributeError, match="missing_utility"):
        utils.__getattr__("missing_utility")
