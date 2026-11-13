from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from typing import (
    Any,
    ClassVar,
    cast,
    get_origin,
    get_args,
)
import json
import re
import inspect

from pydantic import BaseModel, Field, create_model

from instructor.v2.core.utils import UNION_ORIGINS


class _ToolCallStart(str):
    """A text fragment that starts a separate tool call in the string extractor API."""


class _TaskObjects:
    """Scan object arguments and the items of a tasks array across chunk boundaries."""

    def __init__(self, *, unwrap_tasks: bool = True, depth: int = 0) -> None:
        self.unwrap_tasks = unwrap_tasks
        self.depth = depth
        self.quoted = False
        self.escaped = False
        self.in_array = False
        self.may_unwrap = False
        self.wrapper_array_depth = 0
        self.wrapper_tail_depth = 0
        self.buffer: list[str] = []

    def feed(self, chunk: str) -> Generator[tuple[str, int], None, None]:
        for index, char in enumerate(chunk):
            if self.depth and not self.wrapper_tail_depth:
                self.buffer.append(char)
            if self.escaped:
                self.escaped = False
            elif self.quoted and char == "\\":
                self.escaped = True
            elif char == '"':
                self.quoted = not self.quoted
            elif not self.quoted:
                # An unwrapped tasks array still belongs to its outer object.
                # Ignore its remaining fields until that wrapper closes, rather
                # than interpreting metadata objects as additional tasks.
                if self.wrapper_tail_depth:
                    if char == "{":
                        self.wrapper_tail_depth += 1
                    elif char == "}":
                        self.wrapper_tail_depth -= 1
                    continue
                if self.wrapper_array_depth:
                    if char == "[":
                        self.wrapper_array_depth += 1
                    elif char == "]":
                        self.wrapper_array_depth -= 1
                        if not self.wrapper_array_depth:
                            self.wrapper_tail_depth = 1
                if char == "{":
                    if self.depth == 0:
                        self.buffer = [char]
                        self.may_unwrap = self.unwrap_tasks and not self.in_array
                    self.depth += 1
                elif char == "}" and self.depth:
                    self.depth -= 1
                    if not self.depth:
                        yield "".join(self.buffer), index + 1
                        self.buffer.clear()
                elif char == "[":
                    if not self.depth:
                        self.in_array = True
                    elif self.may_unwrap:
                        self.may_unwrap = False
                        if self.depth == 1 and re.fullmatch(
                            r'\{\s*"tasks"\s*:\s*\[', "".join(self.buffer)
                        ):
                            self.buffer.clear()
                            self.depth = 0
                            self.in_array = True
                            self.wrapper_array_depth = 1
                elif char == "]" and not self.depth:
                    self.in_array = False

    def finish(self) -> None:
        if self.depth or self.wrapper_tail_depth > 1:
            from instructor.v2.core.errors import IncompleteOutputException

            raise IncompleteOutputException(
                message="The stream ended inside an unfinished item; the output is incomplete."
            )


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
        try:
            json_chunks = stream_extractor(completion)
            parser = task_parser or cls.tasks_from_chunks
            yield from parser(json_chunks, **kwargs)
        finally:
            close = getattr(completion, "close", None)
            if callable(close):
                close()

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
        try:
            json_chunks = stream_extractor(completion)
            parser = task_parser or cls.tasks_from_chunks_async
            async for item in parser(json_chunks, **kwargs):
                yield item
        finally:
            close = getattr(completion, "aclose", None) or getattr(
                completion, "close", None
            )
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result

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
        scanner = _TaskObjects()
        for chunk in json_chunks:
            if isinstance(chunk, _ToolCallStart):
                scanner.finish()
                scanner = _TaskObjects()
            for task_json, _ in scanner.feed(chunk):
                yield cls.extract_cls_task_type(task_json, **kwargs)
        scanner.finish()

    @classmethod
    async def tasks_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[BaseModel, None]:
        scanner = _TaskObjects()
        async for chunk in json_chunks:
            if isinstance(chunk, _ToolCallStart):
                scanner.finish()
                scanner = _TaskObjects()
            for task_json, _ in scanner.feed(chunk):
                yield cls.extract_cls_task_type(task_json, **kwargs)
        scanner.finish()

    @classmethod
    def extract_cls_task_type(
        cls,
        task_json: str,
        **kwargs: Any,
    ):
        from instructor.v2.dsl.simple_type import unwrap_adapter

        assert cls.task_type is not None
        # PEP 604 unions use types.UnionType rather than typing.Union as their origin.
        if get_origin(cls.task_type) in UNION_ORIGINS:
            union_members = get_args(cls.task_type)
            for member in union_members:
                try:
                    return unwrap_adapter(
                        member.model_validate_json(task_json, **kwargs)
                    )
                except Exception:
                    pass
        else:
            return unwrap_adapter(
                cls.task_type.model_validate_json(task_json, **kwargs)
            )
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
        scanner = _TaskObjects(unwrap_tasks=False, depth=stack)
        for _, end in scanner.feed(s):
            return s[s.find("{") : end], s[end:].removeprefix(",")
        return None, s


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
        if get_origin(subtask_class) in UNION_ORIGINS:
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
