from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from pydantic import VERSION, BaseModel, Field, ValidationError, field_validator

from instructor import Partial

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"]),
]
requires_data_factory = pytest.mark.skipif(
    tuple(map(int, VERSION.split(".")[:2])) < (2, 10),
    reason="Pydantic added data-aware default factories in 2.10",
)


async def stream(
    schema: type[BaseModel], chunks: list[str], asynchronous: bool
) -> AsyncGenerator[Any, None]:
    # Partial adds these streaming methods dynamically.
    api = cast(Any, Partial[schema])
    if asynchronous:

        async def source() -> AsyncGenerator[str, None]:
            for chunk in chunks:
                yield chunk

        async for obj in api.model_from_chunks_async(source()):
            yield obj
    else:
        for obj in api.model_from_chunks(iter(chunks)):
            yield obj


@requires_data_factory
@pytest.mark.parametrize("fragmented", [False, True])
async def test_dependent_default_waits_for_validation(
    asynchronous: bool, fragmented: bool
) -> None:
    factory_inputs: list[int] = []

    def derive(data: dict[str, Any]) -> int:
        factory_inputs.append(data["quantity"])
        assert isinstance(data["quantity"], int)
        return data["quantity"] * 2

    class Order(BaseModel):
        quantity: int
        total: int = Field(default_factory=derive)

        @field_validator("quantity")
        @classmethod
        def add_one(cls, value: int) -> int:
            return value + 1

    chunks = ["{", '"quantity":"2', '1"', "}"]
    if not fragmented:
        chunks = ["".join(chunks)]
    outputs = []
    async for obj in stream(Order, chunks, asynchronous):
        outputs.append(obj)
        if len(outputs) < len(chunks):
            assert obj.total is None
            assert factory_inputs == []
    assert outputs[-1].quantity == 22
    assert outputs[-1].total == 44
    assert factory_inputs and all(value == 22 for value in factory_inputs)
    assert outputs[-1] == Order.model_validate_json("".join(chunks))


@requires_data_factory
async def test_explicit_value_does_not_invoke_factory(asynchronous: bool) -> None:
    def derive(data: dict[str, Any]) -> str:
        raise AssertionError(f"Explicit username must bypass the factory: {data}")

    class User(BaseModel):
        email: str
        username: str = Field(default_factory=derive)

    chunks = ['{"email":"alice@example.com","username":"custom"', "}"]
    outputs = [obj async for obj in stream(User, chunks, asynchronous)]
    assert all(obj.username == "custom" for obj in outputs)


@requires_data_factory
@pytest.mark.parametrize("in_list", [False, True], ids=["nested", "list-item"])
async def test_complete_child_evaluates_default_before_parent(
    asynchronous: bool, in_list: bool
) -> None:
    class User(BaseModel):
        email: str
        username: str = Field(default_factory=lambda data: data["email"].split("@")[0])

    if in_list:

        class Envelope(BaseModel):
            users: list[User]
            note: str

        chunks = ['{"users":[{"email":"ali', 'ce@example.com"}],"note":"d', 'one"}']
    else:

        class Envelope(BaseModel):
            user: User
            note: str

        chunks = ['{"user":{"email":"ali', 'ce@example.com"},"note":"d', 'one"}']

    outputs = [obj async for obj in stream(Envelope, chunks, asynchronous)]
    first_child = outputs[0].users[0] if in_list else outputs[0].user
    complete_child = outputs[1].users[0] if in_list else outputs[1].user
    assert first_child.email == "ali"
    assert first_child.username is None
    assert complete_child.username == "alice"
    assert outputs[1].note == "d"
    assert outputs[-1] == Envelope.model_validate_json("".join(chunks))


@requires_data_factory
async def test_incomplete_stream_keeps_dependent_default_unset(
    asynchronous: bool,
) -> None:
    class User(BaseModel):
        email: str
        username: str = Field(default_factory=lambda data: data["email"].split("@")[0])

    outputs = [obj async for obj in stream(User, ['{"email":"ali'], asynchronous)]
    assert outputs[0].email == "ali"
    assert outputs[0].username is None


@requires_data_factory
@pytest.mark.parametrize("chunks", [["{}"], ["", " ", "{", "}"]])
async def test_empty_object_validates_defaulted_dependencies(
    asynchronous: bool, chunks: list[str]
) -> None:
    class Defaults(BaseModel):
        quantity: int = 7
        total: int = Field(default_factory=lambda data: data["quantity"] * 2)

    outputs = [obj async for obj in stream(Defaults, chunks, asynchronous)]
    assert all(obj.total is None for obj in outputs[:-1])
    assert outputs[-1] == Defaults.model_validate_json("{}")
    assert outputs[-1].total == 14


async def test_independent_defaults_remain_available(asynchronous: bool) -> None:
    class Envelope(BaseModel):
        note: str
        items: list[str] = Field(default_factory=list)
        metadata: dict[str, str] = Field(default_factory=dict)
        label: str = Field(default_factory=lambda: "ready")
        optional_argument: str = Field(default_factory=lambda value="default": value)
        tags: list[str] = ["initial"]

    outputs = [
        obj async for obj in stream(Envelope, ["{", '"note":"d', 'one"}'], asynchronous)
    ]
    for obj in outputs:
        assert obj.items == []
        assert obj.metadata == {}
        assert obj.label == "ready"
        assert obj.optional_argument == "default"
        assert obj.tags == ["initial"]
    outputs[0].items.append("changed")
    outputs[0].tags.append("changed")
    assert outputs[1].items == []
    assert outputs[1].tags == ["initial"]


async def test_empty_object_still_requires_mandatory_fields(asynchronous: bool) -> None:
    class Required(BaseModel):
        name: str

    outputs = []
    with pytest.raises(ValidationError, match="name"):
        async for obj in stream(Required, [" ", "{", "}"], asynchronous):
            outputs.append(obj)
            assert obj.name is None
    assert len(outputs) == 2


async def test_independent_factory_errors_propagate(asynchronous: bool) -> None:
    def fail() -> str:
        raise ValueError("default unavailable")

    class Envelope(BaseModel):
        note: str
        label: str = Field(default_factory=fail)

    with pytest.raises(ValueError, match="default unavailable"):
        async for _ in stream(Envelope, ["{"], asynchronous):
            pass


@requires_data_factory
async def test_aliased_mapping_children_wait_for_validated_data(
    asynchronous: bool,
) -> None:
    calls = []

    def derived(data: dict[str, Any]) -> str:
        calls.append(data["code"])
        return data["code"].upper()

    class Child(BaseModel):
        code: str = Field(alias="wireCode")
        label: str = Field(default_factory=derived)
        metadata: dict = Field(default_factory=dict)

    class Envelope(BaseModel):
        entries: dict[str, list[Child]] = Field(alias="groups")

    chunks = ['{"groups":{"a":[{"wireCode":"ab"', '}],"b":[{"wireCode":"cd"', "}]}}"]
    values = [obj async for obj in stream(Envelope, chunks, asynchronous)]
    first = values[0].entries["a"][0]
    assert isinstance(first, Child)
    assert first.label is None and first.metadata == {}
    assert values[1].entries["a"][0].label == "AB"
    assert values[1].entries["b"][0].label is None
    assert [values[-1].entries[k][0].label for k in ("a", "b")] == ["AB", "CD"]
    assert set(calls) == {"ab", "cd"}


@pytest.mark.parametrize(
    "kind", ["value", "factory", pytest.param("dependent", marks=requires_data_factory)]
)
async def test_missing_nested_model_preserves_default(
    asynchronous: bool, kind: str
) -> None:
    class Child(BaseModel):
        value: int

    if kind == "value":
        default = Field(default=Child(value=7))
    elif kind == "factory":
        default = Field(default_factory=lambda: Child(value=7))
    else:
        default = Field(default_factory=lambda data: Child(value=data["number"]))

    class Parent(BaseModel):
        number: int
        child: Child = default
        note: str

    chunks = ['{"number":7,"note":"par', 'tial"}']
    outputs = [obj async for obj in stream(Parent, chunks, asynchronous)]
    if kind == "dependent":
        assert outputs[0].child is None
    else:
        assert outputs[0].child == Child(value=7)
    assert outputs[-1] == Parent.model_validate_json("".join(chunks))


async def test_nested_default_is_copied_between_snapshots(asynchronous: bool) -> None:
    class Child(BaseModel):
        values: list[int]

    class Parent(BaseModel):
        child: Child = Child(values=[7])
        note: str

    outputs = [
        obj async for obj in stream(Parent, ['{"note":"a', "b", 'c"}'], asynchronous)
    ]
    assert outputs[0].child.values == [7]
    outputs[0].child.values.append(9)
    assert outputs[1].child.values == [7]
    assert Parent.model_fields["child"].default.values == [7]


async def test_streamed_nested_value_overrides_default(asynchronous: bool) -> None:
    class Child(BaseModel):
        value: int

    class Parent(BaseModel):
        child: Child = Child(value=7)
        required_child: Child
        note: str

    chunks = ['{"child":{"value":3},"note":"a', '","required_child":{"value":4}}']
    outputs = [obj async for obj in stream(Parent, chunks, asynchronous)]
    assert outputs[0].child == Child(value=3)
    assert outputs[0].required_child.value is None
    assert outputs[-1] == Parent.model_validate_json("".join(chunks))


async def test_explicit_null_overrides_nested_default(asynchronous: bool) -> None:
    class Child(BaseModel):
        value: int

    class Parent(BaseModel):
        child: Child | None = Child(value=7)
        note: str

    chunks = ['{"child":null,"note":"a', 'b"}']
    outputs = [obj async for obj in stream(Parent, chunks, asynchronous)]
    assert all(obj.child is None for obj in outputs)
    assert outputs[-1] == Parent.model_validate_json("".join(chunks))


async def test_recursive_optional_default_remains_none(asynchronous: bool) -> None:
    class Node(BaseModel):
        name: str
        child: Node | None = None

    chunks = ['{"name":"a', 'b"}']
    outputs = [obj async for obj in stream(Node, chunks, asynchronous)]
    assert all(obj.child is None for obj in outputs)
    assert outputs[-1] == Node.model_validate_json("".join(chunks))
