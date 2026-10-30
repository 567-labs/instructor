from __future__ import annotations

import json
import sys
from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from pathlib import Path
from typing import Any, Union, cast

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

from instructor.v2.core.errors import ResponseParsingError
from instructor.v2.core.mode import Mode
from instructor.v2.dsl import parallel as parallel_module
from instructor.v2.dsl.iterable import IterableBase, IterableModel
from instructor.v2.dsl.parallel import (
    AnthropicParallelModel,
    ParallelBase,
    ParallelModel,
    VertexAIParallelModel,
    get_types_array,
    handle_anthropic_parallel_model,
    handle_parallel_model,
    is_union_type,
)
from tests.coverage._openai import chat_completion, tool_call
from tests.coverage._streams import async_items


class EmailJob(BaseModel):
    address: str
    attempts: int = 1

    @field_validator("address")
    @classmethod
    def check_domain(cls, value: str, info: ValidationInfo) -> str:
        domain = (info.context or {}).get("domain")
        if domain and not value.endswith(f"@{domain}"):
            raise ValueError(f"address must use {domain}")
        return value


class SmsJob(BaseModel):
    number: str


class EmailIterable(IterableBase):
    task_type = EmailJob


class UnionIterable(IterableBase):
    task_type = Union[EmailJob, SmsJob]


def extract_deltas(completion: Iterable[Any]) -> Generator[str, None, None]:
    for event in completion:
        yield event["delta"]


async def extract_deltas_async(
    completion: AsyncGenerator[dict[str, str], None],
) -> AsyncGenerator[str, None]:
    async for event in completion:
        yield event["delta"]


def tool_response(*calls: tuple[str, str]) -> ChatCompletion:
    return chat_completion(
        tool_calls=[
            tool_call(name, arguments, f"call-{index}")
            for index, (name, arguments) in enumerate(calls)
        ],
        finish_reason="tool_calls",
    )


def test_iterable_sync_stream_handles_split_objects_and_custom_list_parser() -> None:
    split_stream = [
        {"delta": '{"tasks": ['},
        {"delta": '{"address": "first@example.test", "attempts": 2},'},
        {"delta": '{"address": "second@example.test"}]}'},
    ]
    parsed = list(
        EmailIterable.from_streaming_response(
            split_stream,
            stream_extractor=extract_deltas,
            context={"domain": "example.test"},
        )
    )
    assert parsed == [
        EmailJob(address="first@example.test", attempts=2),
        EmailJob(address="second@example.test"),
    ]

    list_stream = [
        {"delta": ""},
        {"delta": '{"tasks": []}'},
        {"delta": '{"tasks": [{"address": "third@example.test"}]}'},
    ]
    assert list(
        EmailIterable.from_streaming_response(
            list_stream,
            stream_extractor=extract_deltas,
            task_parser=EmailIterable.tasks_from_task_list_chunks,
        )
    ) == [EmailJob(address="third@example.test")]


@pytest.mark.asyncio
async def test_iterable_async_stream_handles_split_objects_and_custom_list_parser() -> (
    None
):
    split_stream = async_items(
        [
            {"delta": '{"tasks":'},
            {"delta": " ["},
            {"delta": '{"address": "first@example.test", "attempts": 2},'},
            {"delta": '{"address": "second@example.test"}]}'},
        ]
    )
    parsed = [
        item
        async for item in EmailIterable.from_streaming_response_async(
            split_stream,
            stream_extractor=extract_deltas_async,
            context={"domain": "example.test"},
        )
    ]
    assert parsed == [
        EmailJob(address="first@example.test", attempts=2),
        EmailJob(address="second@example.test"),
    ]

    list_stream = async_items(
        [
            {"delta": ""},
            {"delta": '{"tasks": []}'},
            {"delta": '{"tasks": [{"address": "third@example.test"}]}'},
        ]
    )
    assert [
        item
        async for item in EmailIterable.from_streaming_response_async(
            list_stream,
            stream_extractor=extract_deltas_async,
            task_parser=EmailIterable.tasks_from_task_list_chunks_async,
        )
    ] == [EmailJob(address="third@example.test")]


def test_iterable_sync_rejects_missing_extractor_and_malformed_task_lists() -> None:
    missing_extractor = cast(
        Callable[[Iterable[Any]], Generator[str, None, None]], None
    )
    with pytest.raises(ValueError, match="stream_extractor is required"):
        list(
            EmailIterable.from_streaming_response(
                [], stream_extractor=missing_extractor
            )
        )
    with pytest.raises(ValueError, match="stream_extractor is required"):
        list(EmailIterable.extract_json([], stream_extractor=missing_extractor))

    with pytest.raises(json.JSONDecodeError):
        list(EmailIterable.tasks_from_task_list_chunks(["not-json"]))
    with pytest.raises(KeyError, match="tasks"):
        list(EmailIterable.tasks_from_task_list_chunks(['{"items": []}']))
    with pytest.raises(ValidationError):
        list(
            EmailIterable.tasks_from_task_list_chunks(
                ['{"tasks": [{"address": "bad@other.test"}]}'],
                context={"domain": "example.test"},
            )
        )


@pytest.mark.asyncio
async def test_iterable_async_rejects_missing_extractor_and_malformed_task_lists() -> (
    None
):
    missing_extractor = cast(
        Callable[[AsyncGenerator[Any, None]], AsyncGenerator[str, None]], None
    )
    with pytest.raises(ValueError, match="stream_extractor is required"):
        [
            item
            async for item in EmailIterable.from_streaming_response_async(
                async_items([]), stream_extractor=missing_extractor
            )
        ]
    with pytest.raises(ValueError, match="stream_extractor is required"):
        [
            item
            async for item in EmailIterable.extract_json_async(
                async_items([]), stream_extractor=missing_extractor
            )
        ]

    with pytest.raises(json.JSONDecodeError):
        [
            item
            async for item in EmailIterable.tasks_from_task_list_chunks_async(
                async_items(["not-json"])
            )
        ]
    with pytest.raises(KeyError, match="tasks"):
        [
            item
            async for item in EmailIterable.tasks_from_task_list_chunks_async(
                async_items(['{"items": []}'])
            )
        ]
    with pytest.raises(ValidationError):
        [
            item
            async for item in EmailIterable.tasks_from_task_list_chunks_async(
                async_items(['{"tasks": [{"address": "bad@other.test"}]}']),
                context={"domain": "example.test"},
            )
        ]


@pytest.mark.asyncio
async def test_extract_json_helpers_preserve_real_chunk_order() -> None:
    completion = [{"delta": "first"}, {"delta": "second"}]
    assert list(EmailIterable.extract_json(completion, extract_deltas)) == [
        "first",
        "second",
    ]
    assert [
        chunk
        async for chunk in EmailIterable.extract_json_async(
            async_items(completion), extract_deltas_async
        )
    ] == ["first", "second"]


def test_iterable_union_parser_tries_each_model_and_reports_invalid_payload() -> None:
    assert UnionIterable.extract_cls_task_type('{"address": "a@example.test"}') == (
        EmailJob(address="a@example.test")
    )
    assert UnionIterable.extract_cls_task_type('{"number": "+15550001111"}') == (
        SmsJob(number="+15550001111")
    )
    with pytest.raises(ValueError, match="Failed to extract task type"):
        UnionIterable.extract_cls_task_type('{"unknown": true}')


def test_iterable_object_scanner_ignores_escaped_quotes_and_braces_in_strings() -> None:
    task_json, remaining = IterableBase.get_object(
        r'{"address": "first\"{tag}@example.test"}, {"number": "+15550001111"}',
        0,
    )

    assert task_json is not None
    assert json.loads(task_json) == {"address": 'first"{tag}@example.test'}
    assert json.loads(remaining) == {"number": "+15550001111"}


def test_iterable_model_supports_custom_names_union_names_and_forward_refs() -> None:
    named = IterableModel(EmailJob, name="QueuedEmail", description="Queued email jobs")
    union = IterableModel(cast(type[BaseModel], Union[EmailJob, SmsJob]))
    forward_ref = IterableModel(cast(type[BaseModel], "EmailJob"))

    assert named.__name__ == "IterableQueuedEmail"
    assert named.__doc__ == "Queued email jobs"
    assert union.__name__ == "IterableEmailJobOrSmsJob"
    assert forward_ref.__name__ == "IterableEmailJob"
    assert forward_ref.model_fields["tasks"].annotation == list["EmailJob"]


def test_parallel_base_validates_real_tool_calls_context_and_strictness() -> None:
    model = ParallelBase(EmailJob, SmsJob)
    response = tool_response(
        ("EmailJob", '{"address": "a@example.test", "attempts": 2}'),
        ("SmsJob", '{"number": "+15550001111"}'),
    )
    assert list(
        model.from_response(
            response,
            mode=Mode.TOOLS,
            validation_context={"domain": "example.test"},
            strict=True,
        )
    ) == [
        EmailJob(address="a@example.test", attempts=2),
        SmsJob(number="+15550001111"),
    ]

    with pytest.raises(ValidationError):
        list(
            model.from_response(
                tool_response(("EmailJob", '{"address": "a@other.test"}')),
                mode=Mode.TOOLS,
                validation_context={"domain": "example.test"},
            )
        )
    with pytest.raises(ValidationError):
        list(
            model.from_response(
                tool_response(
                    ("EmailJob", '{"address": "a@example.test", "attempts": "2"}')
                ),
                mode=Mode.TOOLS,
                strict=True,
            )
        )
    with pytest.raises(ResponseParsingError, match="MissingJob"):
        list(model.from_response(tool_response(("MissingJob", "{}")), mode=Mode.TOOLS))


def test_parallel_base_requires_at_least_one_model() -> None:
    with pytest.raises(AssertionError, match="At least one model is required"):
        ParallelBase()


def test_parallel_type_helpers_support_single_and_union_models() -> None:
    assert get_types_array(Iterable[EmailJob]) == (EmailJob,)
    assert get_types_array(Iterable[Union[EmailJob, SmsJob]]) == (EmailJob, SmsJob)
    assert is_union_type(Iterable[Union[EmailJob, SmsJob]]) is True
    assert is_union_type(Iterable[EmailJob]) is False

    with pytest.raises(TypeError, match="Model should be with Iterable"):
        get_types_array(list[EmailJob])


def test_parallel_type_helpers_keep_python_39_union_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = Path(parallel_module.__file__)
    namespace: dict[str, Any] = {
        "__name__": "parallel_python_39",
        "__file__": str(source_path),
    }
    monkeypatch.setattr(sys, "version_info", (3, 9, 18))
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)

    legacy_is_union = namespace["is_union_type"]
    assert legacy_is_union(Iterable[Union[EmailJob, SmsJob]]) is True
    assert legacy_is_union(Iterable[EmailJob]) is False


def test_parallel_schema_and_provider_factories_register_all_models() -> None:
    typehint = Iterable[Union[EmailJob, SmsJob]]
    openai_tools = handle_parallel_model(typehint)
    anthropic_tools = handle_anthropic_parallel_model(typehint)

    assert [tool["function"]["name"] for tool in openai_tools] == [
        "EmailJob",
        "SmsJob",
    ]
    assert [tool["name"] for tool in anthropic_tools] == ["EmailJob", "SmsJob"]
    assert set(ParallelModel(typehint).registry) == {"EmailJob", "SmsJob"}
    assert set(VertexAIParallelModel(typehint).registry) == {"EmailJob", "SmsJob"}
    assert set(AnthropicParallelModel(typehint).registry) == {"EmailJob", "SmsJob"}
