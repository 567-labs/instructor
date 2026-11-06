from __future__ import annotations

import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

import instructor

from instructor.v2.validation.async_validators import (
    async_field_validator,
    async_model_validator,
    run_async_validators,
    model_declares_async_validators,
)


class Ordered(BaseModel):
    value: str

    @async_field_validator("value")
    async def prefix(cls, value: str) -> str:
        return "prefix:" + value

    @async_field_validator("value")
    async def suffix(cls, value: str) -> str:
        return value + ":suffix"


def test_sync_client_rejects_nested_async_validators_before_request() -> None:
    class Parent(BaseModel):
        child: Ordered

    with OpenAI(api_key="unused", base_url="http://localhost:1") as sdk:
        client = instructor.from_openai(sdk)
        with pytest.raises(ValueError, match="async validators are not supported"):
            client.create(
                model="unused",
                response_model=Parent,
                messages=[{"role": "user", "content": "unused"}],
            )


def test_validation_lazy_helpers_match_direct_exports() -> None:
    from instructor.v2 import validation
    from instructor.v2.core.validators import Validator
    from instructor.v2.validation.llm_validators import (
        llm_validator,
        openai_moderation,
    )

    assert validation.llm_validator is llm_validator
    assert validation.openai_moderation is openai_moderation
    assert validation.__getattr__("Validator") is Validator


@pytest.mark.asyncio
async def test_async_field_validators_chain_results() -> None:
    result = await run_async_validators(Ordered(value="x"), context=None)
    assert result.value == "prefix:x:suffix"


@pytest.mark.asyncio
async def test_undecorated_override_suppresses_inherited_validator() -> None:
    class Child(Ordered):
        async def prefix(cls, value: str) -> str:
            return value

    result = await run_async_validators(Child(value="x"), context=None)
    assert result.value == "x:suffix"


def test_nested_container_validators_are_detected() -> None:
    class Parent(BaseModel):
        children: dict[str, list[Ordered]]

    assert model_declares_async_validators(Parent)
    assert model_declares_async_validators(list[Parent])


def test_recursive_model_without_validators_terminates() -> None:
    class Node(BaseModel):
        children: list[Node] = []

    Node.model_rebuild()
    assert not model_declares_async_validators(Node)


@pytest.mark.asyncio
async def test_nested_container_transformations_preserve_original() -> None:
    class Parent(BaseModel):
        children: dict[str, tuple[Ordered, list[Ordered]]]

    original = Parent(children={"group": (Ordered(value="a"), [Ordered(value="b")])})
    result = await run_async_validators(original, context=None)
    assert result.children["group"][0].value == "prefix:a:suffix"
    assert result.children["group"][1][0].value == "prefix:b:suffix"
    assert original.children["group"][0].value == "a"
    assert original.children["group"][1][0].value == "b"


@pytest.mark.asyncio
async def test_model_validator_errors_are_reported() -> None:
    from instructor.v2.core.errors import AsyncValidationError

    class Rejected(BaseModel):
        value: str

        @async_model_validator()
        async def reject(self) -> Rejected:
            raise ValueError(f"Rejected {self.value}")

    with pytest.raises(AsyncValidationError, match="Rejected x") as error:
        await run_async_validators(Rejected(value="x"), context=None)
    assert len(error.value.errors) == 1


@pytest.mark.asyncio
async def test_inherited_validator_for_subclass_field_is_ignored_on_base() -> None:
    class Base(BaseModel):
        @async_field_validator("value")
        async def normalize(cls, value: str) -> str:
            return value.strip()

    class Child(Base):
        value: str

    assert await run_async_validators(Base(), context=None) == Base()
    assert (await run_async_validators(Child(value=" x "), context=None)).value == "x"


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["stream", "list", "partial", "parallel"])
async def test_unsupported_async_workflows_fail_before_request(workflow: str) -> None:
    async with AsyncOpenAI(api_key="unused", base_url="http://localhost:1") as sdk:
        mode = (
            instructor.Mode.PARALLEL_TOOLS
            if workflow == "parallel"
            else instructor.Mode.TOOLS
        )
        client = instructor.from_openai(sdk, mode=mode)
        response_model = (
            list[Ordered]
            if workflow == "list"
            else instructor.Partial[Ordered]
            if workflow == "partial"
            else Ordered
        )
        with pytest.raises(ValueError, match="async validators are not supported"):
            await client.create(
                model="unused",
                response_model=response_model,
                messages=[{"role": "user", "content": "unused"}],
                stream=workflow == "stream",
            )
