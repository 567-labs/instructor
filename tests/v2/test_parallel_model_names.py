"""Parallel tool names must identify exactly one response model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Union

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, create_model

from instructor.v2.core.mode import Mode
from instructor.v2.dsl.parallel import ParallelModel, handle_parallel_model
from instructor.v2.providers.anthropic.parallel import (
    AnthropicParallelModel,
    handle_parallel_model as handle_anthropic_parallel_model,
)
from instructor.v2.providers.mistral.handlers import MistralToolsHandler
from instructor.v2.providers.vertexai.parallel import VertexAIParallelModel


class Payment(BaseModel):
    model_config = ConfigDict(title="Record")
    amount: int = 0


class Contact(BaseModel):
    model_config = ConfigDict(title="Record")
    name: str = ""


@pytest.mark.parametrize("factory", [ParallelModel, AnthropicParallelModel])
def test_parallel_models_reject_duplicate_schema_titles(factory: Any) -> None:
    with pytest.raises(ValueError, match="unique.*Record"):
        factory(Iterable[Union[Payment, Contact]])


@pytest.mark.parametrize(
    "factory", [handle_parallel_model, handle_anthropic_parallel_model]
)
def test_tool_schemas_reject_duplicate_names(factory: Any) -> None:
    with pytest.raises(ValueError, match="unique.*Record"):
        factory(Iterable[Union[Payment, Contact]])


def test_mistral_preparation_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique.*Record"):
        MistralToolsHandler().prepare_request(
            Iterable[Union[Payment, Contact]],  # type: ignore[arg-type]
            {"messages": [{"role": "user", "content": "Extract records"}]},
        )


def test_vertex_rejects_duplicate_class_names() -> None:
    payment = create_model(
        "Record", __config__=ConfigDict(title="Payment"), amount=(int, 0)
    )
    contact = create_model(
        "Record", __config__=ConfigDict(title="Contact"), name=(str, "")
    )
    with pytest.raises(ValueError, match="unique.*Record"):
        VertexAIParallelModel(Iterable[Union[payment, contact]])


@pytest.mark.parametrize("factory", [ParallelModel, AnthropicParallelModel])
def test_distinct_titles_keep_models_with_the_same_class_name(factory: Any) -> None:
    payment = create_model(
        "Record", __config__=ConfigDict(title="Payment"), amount=(int, 0)
    )
    contact = create_model(
        "Record", __config__=ConfigDict(title="Contact"), name=(str, "")
    )
    parallel = factory(Iterable[Union[payment, contact]])
    assert parallel.registry == {"Payment": payment, "Contact": contact}


def test_vertex_uses_class_names_instead_of_schema_titles() -> None:
    parallel = VertexAIParallelModel(Iterable[Union[Payment, Contact]])
    assert parallel.registry == {"Payment": Payment, "Contact": Contact}


def test_distinct_names_preserve_tool_call_routing() -> None:
    class PaymentRecord(Payment):
        model_config = ConfigDict(title="Payment")

    class ContactRecord(Contact):
        model_config = ConfigDict(title="Contact")

    parallel = ParallelModel(Iterable[Union[PaymentRecord, ContactRecord]])
    response = ChatCompletion.model_validate(
        {
            "id": "offline",
            "object": "chat.completion",
            "created": 0,
            "model": "offline",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "Payment",
                                    "arguments": '{"amount": 42}',
                                },
                            },
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "Contact",
                                    "arguments": '{"name": "Ada"}',
                                },
                            },
                        ],
                    },
                }
            ],
        }
    )
    assert list(parallel.from_response(response, Mode.PARALLEL_TOOLS)) == [
        PaymentRecord(amount=42),
        ContactRecord(name="Ada"),
    ]
