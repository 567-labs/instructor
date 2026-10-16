from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from typing import (
    Any,
    ClassVar,
    cast,
    get_origin,
    get_args,
    Union,
)
import json
import sys

from pydantic import BaseModel, Field, create_model

if sys.version_info >= (3, 10):
    from types import UnionType

    _UNION_ORIGINS: tuple[Any, ...] = (Union, UnionType)
else:  # pragma: no cover - Python 3.9 has no runtime ``X | Y`` syntax
    _UNION_ORIGINS = (Union,)


# Resumable object scanner state: (next index, brace depth, in string, escape next).
_ScanState = tuple[int, int, bool, bool]
_SCAN_START: _ScanState = (0, 0, False, False)


class IterableBase:
    task_type: ClassVar[type[BaseModel] | None] = None

    @classmethod
    def from_streaming_response(
        cls,
        completion: Iterable[Any],
        stream_extractor: Callable[[Iterable[Any]], Generator[str, None, None]],
        task_parser: Callable[..., Generator[BaseModel, None, None]] | None = None,
        **kwargs: Any,
    ) -> Generator[BaseModel, None, None]:
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        json_chunks = stream_extractor(completion)
        parser = task_parser or cls.tasks_from_chunks
        yield from parser(json_chunks, **kwargs)

    @classmethod
    async def from_streaming_response_async(
        cls,
        completion: AsyncGenerator[Any, None],
        stream_extractor: Callable[
            [AsyncGenerator[Any, None]], AsyncGenerator[str, None]
        ],
        task_parser: Callable[..., AsyncGenerator[BaseModel, None]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseModel, None]:
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        json_chunks = stream_extractor(completion)
        parser = task_parser or cls.tasks_from_chunks_async
        async for item in parser(json_chunks, **kwargs):
            yield item

    @classmethod
    async def tasks_from_task_list_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[BaseModel, None]:
        """Process streaming chunks that contain a full tasks list."""

        async for chunk in json_chunks:
            if not chunk:
                continue
            json_response = json.loads(chunk)
            if not json_response["tasks"]:
                continue

            for item in json_response["tasks"]:
                obj = cls.extract_cls_task_type(json.dumps(item), **kwargs)
                yield obj

    @classmethod
    def tasks_from_task_list_chunks(
        cls, json_chunks: Iterable[str], **kwargs: Any
    ) -> Generator[BaseModel, None, None]:
        """Process streaming chunks that contain a full tasks list."""
        for chunk in json_chunks:
            if not chunk:
                continue
            json_response = json.loads(chunk)
            if not json_response["tasks"]:
                continue

            for item in json_response["tasks"]:
                obj = cls.extract_cls_task_type(json.dumps(item), **kwargs)
                yield obj

    @classmethod
    def tasks_from_chunks(
        cls, json_chunks: Iterable[str], **kwargs: Any
    ) -> Generator[BaseModel, None, None]:
        started = False
        potential_object = ""
        scan_state = _SCAN_START
        for chunk in json_chunks:
            potential_object += chunk
            if not started:
                if "[" in chunk:
                    started = True
                    potential_object = chunk[chunk.find("[") + 1 :]
                    scan_state = _SCAN_START

            while True:
                task_json, potential_object, scan_state = cls._scan_object(
                    potential_object, scan_state
                )
                if task_json:
                    assert cls.task_type is not None
                    obj = cls.extract_cls_task_type(task_json, **kwargs)
                    yield obj
                else:
                    break

    @classmethod
    async def tasks_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[BaseModel, None]:
        started = False
        potential_object = ""
        scan_state = _SCAN_START
        async for chunk in json_chunks:
            potential_object += chunk
            if not started:
                if "[" in chunk:
                    started = True
                    potential_object = chunk[chunk.find("[") + 1 :]
                    scan_state = _SCAN_START

            while True:
                task_json, potential_object, scan_state = cls._scan_object(
                    potential_object, scan_state
                )
                if task_json:
                    assert cls.task_type is not None
                    obj = cls.extract_cls_task_type(task_json, **kwargs)
                    yield obj
                else:
                    break

    @classmethod
    def extract_cls_task_type(
        cls,
        task_json: str,
        **kwargs: Any,
    ):
        assert cls.task_type is not None
        # PEP 604 unions use types.UnionType rather than typing.Union as their origin.
        if get_origin(cls.task_type) in _UNION_ORIGINS:
            union_members = get_args(cls.task_type)
            for member in union_members:
                try:
                    return member.model_validate_json(task_json, **kwargs)
                except Exception:
                    pass
        else:
            return cls.task_type.model_validate_json(task_json, **kwargs)
        raise ValueError(
            f"Failed to extract task type with {task_json} for {cls.task_type}"
        )

    @staticmethod
    def extract_json(
        completion: Iterable[Any],
        stream_extractor: Callable[[Iterable[Any]], Generator[str, None, None]],
    ) -> Generator[str, None, None]:
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        yield from stream_extractor(completion)

    @staticmethod
    async def extract_json_async(
        completion: AsyncGenerator[Any, None],
        stream_extractor: Callable[
            [AsyncGenerator[Any, None]], AsyncGenerator[str, None]
        ],
    ) -> AsyncGenerator[str, None]:
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        async for chunk in stream_extractor(completion):
            yield chunk

    @staticmethod
    def get_object(s: str, stack: int) -> tuple[str | None, str]:
        task_json, remaining, _ = IterableBase._scan_object(s, (0, stack, False, False))
        return task_json, remaining

    @staticmethod
    def _scan_object(s: str, state: _ScanState) -> tuple[str | None, str, _ScanState]:
        """Find the first complete object in ``s``, resuming from ``state``.

        ``state`` is the state returned by a previous call on a prefix of ``s``,
        so a buffer that grows one chunk at a time is scanned once instead of
        from the start on every chunk.
        """
        start, stack, in_string, escape_next = state
        for i, c in enumerate(s[start:], start):
            if escape_next:
                escape_next = False
            elif c == "\\" and in_string:
                escape_next = True
                continue
            elif c == '"':
                in_string = not in_string

            if in_string:
                continue

            if c == "{":
                stack += 1
            elif c == "}":
                stack -= 1
                if stack == 0:
                    return s[s.find("{") : i + 1], s[i + 2 :], _SCAN_START
        return None, s, (len(s), stack, in_string, escape_next)


def IterableModel(
    subtask_class: type[BaseModel],
    name: str | None = None,
    description: str | None = None,
) -> type[BaseModel]:
    # Import at runtime to avoid circular import
    from instructor.v2.core.function_calls import ResponseSchema

    """
    Dynamically create a IterableModel ResponseSchema that can be used to segment multiple
    tasks given a base class. This creates class that can be used to create a toolkit
    for a specific task, names and descriptions are automatically generated. However
    they can be overridden.

    ## Usage

    ```python
    from pydantic import BaseModel, Field
    from instructor import IterableModel

    class User(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")
        role: str = Field(description="The role of the person")

    MultiUser = IterableModel(User)
    ```

    ## Result

    ```python
    class MultiUser(ResponseSchema, MultiTaskBase):
        tasks: List[User] = Field(
            default_factory=list,
            repr=False,
            description="Correctly segmented list of `User` tasks",
        )

        @classmethod
        def from_streaming_response(cls, completion) -> Generator[User]:
            '''
            Parse the streaming response and yield a `User` object
            for each task in the response.
            '''
            json_chunks = cls.extract_json(completion, stream_extractor)
            yield from cls.tasks_from_chunks(json_chunks)
    ```

    Parameters:
        subtask_class (Type[ResponseSchema]): The base class to use for the MultiTask
        name (Optional[str]): The name of the MultiTask class, if None then the name
            of the subtask class is used as `Multi{subtask_class.__name__}`
        description (Optional[str]): The description of the MultiTask class, if None
            then the description is set to `Correct segmentation of `{subtask_class.__name__}` tasks`

    Returns:
        schema (ResponseSchema): A new class that can be used to segment multiple tasks
    """
    if name is not None:
        task_name = name
    else:
        # Handle `Union[A, B]` / `A | B` task types.
        # Both `typing.Union[A, B]` and PEP 604 `A | B` expose ``__name__ == "Union"``
        # (and `types.UnionType` normalizes to ``typing.Union`` on modern Python), so we
        # must detect a union *origin* first and build the name from its members --
        # otherwise every union collapses to the unhelpful ``IterableUnion``.
        if get_origin(subtask_class) in _UNION_ORIGINS:
            members = get_args(subtask_class)
            task_name = "Or".join(getattr(m, "__name__", str(m)) for m in members)
        else:
            task_name = getattr(subtask_class, "__name__", None) or str(subtask_class)

    name = f"Iterable{task_name}"

    list_tasks = (
        list[subtask_class],  # type: ignore
        Field(
            default_factory=list,
            repr=False,
            description=f"Correctly segmented list of `{task_name}` tasks",
        ),
    )

    base_models = cast(tuple[type[BaseModel], ...], (ResponseSchema, IterableBase))
    new_cls = create_model(
        name,
        tasks=list_tasks,
        __base__=base_models,
    )
    new_cls = cast(type[IterableBase], new_cls)

    # set the class constructor BaseModel
    new_cls.task_type = subtask_class

    new_cls.__doc__ = (
        f"Correct segmentation of `{task_name}` tasks"
        if description is None
        else description
    )
    assert issubclass(new_cls, ResponseSchema), (
        "The new class should be a subclass of ResponseSchema"
    )
    return new_cls
