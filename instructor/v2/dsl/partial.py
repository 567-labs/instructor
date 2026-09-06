# --------------------------------------------------------------------------------
# The following code is adapted from a comment on GitHub in the pydantic/pydantic repository by silviumarcu.
# Source: https://github.com/pydantic/pydantic/issues/6381#issuecomment-1831607091
#
# This code is used in accordance with the repository's license, and this reference
# serves as an acknowledgment of the original author's contribution to this project.
# --------------------------------------------------------------------------------

from __future__ import annotations

import re
import types
import warnings
from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from contextvars import ContextVar
from copy import deepcopy
from functools import cache
from functools import reduce
from operator import or_
from typing import (  # noqa: UP035
    Any,
    Generic,
    List,  # needed for runtime check against typing.List annotations from user code
    NoReturn,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from jiter import from_json
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from instructor.v2.dsl.json_tracker import JsonCompleteness, is_json_complete

T_Model = TypeVar("T_Model", bound=BaseModel)

UNION_TYPE = getattr(types, "UnionType", None)
UNION_ORIGINS = (Union, UNION_TYPE) if UNION_TYPE is not None else (Union,)

# Track models currently being processed to prevent infinite recursion
# with self-referential models (e.g., TreeNode with children: List["TreeNode"]).
# Each top-level partial-model construction receives an isolated guard.
_processing_models: ContextVar[set[type] | None] = ContextVar(
    "processing_models", default=None
)


def _unwrap_optional_base_model(annotation: Any) -> type[BaseModel] | None:
    """Return the BaseModel subclass for a type annotation, unwrapping Optional[T].

    Handles:
    - Direct BaseModel subclass  →  returns it as-is
    - ``Optional[T]`` / ``Union[T, None]`` where T is a BaseModel  →  returns T
    - Arbitrary ``Union`` with multiple non-None args  →  returns None (ambiguous)
    - Any other annotation  →  returns None
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin in UNION_ORIGINS:
        non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none_args) == 1:
            arg = non_none_args[0]
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None


class MakeFieldsOptional:
    pass


class PartialLiteralMixin:
    """DEPRECATED: This mixin is no longer necessary.

    With completeness-based validation, Literal and Enum types are handled
    automatically during streaming:
    - Incomplete JSON: no validation runs, partial values are stored as-is
    - Complete JSON: full validation against original model

    You can safely remove this mixin from your models.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        warnings.warn(
            "PartialLiteralMixin is deprecated and no longer necessary. "
            "Completeness-based validation now handles Literal and Enum types "
            "automatically during streaming. You can safely remove this mixin.",
            DeprecationWarning,
            stacklevel=2,
        )


def remove_control_chars(s):
    return re.sub(r"[\x00-\x1F\x7F-\x9F]", "", s)


def process_potential_object(potential_object, partial_mode, partial_model, **kwargs):
    """Process a potential JSON object using completeness-based validation.

    - If JSON is complete (closed braces/brackets): validate against original model
    - If JSON is incomplete: build partial object using model_construct (no validation)

    Note: Pydantic v2.10+ has `experimental_allow_partial` but it doesn't support
    BaseModel constraints during partial validation (only TypedDict). If Pydantic
    adds BaseModel support in the future, this could potentially be simplified.
    See: https://docs.pydantic.dev/latest/concepts/partial_validation/
    """
    json_str = potential_object.strip() or "{}"
    parsed = from_json(json_str.encode(), partial_mode=partial_mode)

    tracker = JsonCompleteness()
    tracker.analyze(json_str)

    # Get original model for validation
    original_model = getattr(partial_model, "_original_model", None)

    # Check if root is complete AND has actual data (not just empty {})
    root_complete = tracker.is_root_complete()
    has_data = bool(parsed) if isinstance(parsed, dict) else True

    validation_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        in {"context", "strict", "extra", "from_attributes", "by_alias", "by_name"}
    }

    if root_complete and has_data and original_model is not None:
        # Root object is complete with data - validate against original model
        return original_model.model_validate(parsed, **validation_kwargs)
    # Object is incomplete or empty - build instance using model_construct (no validation)
    model_for_construct = (
        original_model if original_model is not None else partial_model
    )
    return _build_partial_object(parsed, model_for_construct, tracker, "", **kwargs)


def _build_partial_object(
    data: Any,
    model: type[BaseModel],
    tracker: JsonCompleteness,
    path: str,
    **kwargs: Any,
) -> Any:
    """Build a partial object using model_construct() to skip validation.

    For each field:
    - If the field's JSON is complete AND it's a nested BaseModel: validate it
    - Otherwise: store without validation
    """
    if data is None:
        return None

    if not isinstance(data, dict):
        return data

    result: dict[str, Any] = {}

    for field_name in data:
        field_value = data[field_name]
        field_path = f"{path}.{field_name}" if path else field_name

        if field_value is None:
            result[field_name] = None
            continue

        field_complete = tracker.is_path_complete(field_path)
        field_info = model.model_fields.get(field_name)
        field_type = field_info.annotation if field_info else None

        if field_complete and field_type is not None:
            _base_model = _unwrap_optional_base_model(field_type)
            if _base_model is not None:
                result[field_name] = _base_model.model_validate(field_value, **kwargs)
                continue

        if isinstance(field_value, dict):
            nested_model = (
                _unwrap_optional_base_model(field_type)
                if field_type is not None
                else None
            )

            if nested_model:
                result[field_name] = _build_partial_object(
                    field_value, nested_model, tracker, field_path, **kwargs
                )
            else:
                result[field_name] = field_value
        elif isinstance(field_value, list):
            result[field_name] = _build_partial_list(
                field_value, model, field_name, tracker, field_path, **kwargs
            )
        else:
            result[field_name] = field_value

    # Set missing fields to None or empty nested models
    for field_name, field_info in model.model_fields.items():
        if field_name not in result:
            field_type = field_info.annotation
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                result[field_name] = _build_partial_object(
                    {}, field_type, tracker, "", **kwargs
                )
            elif field_info.is_required():
                result[field_name] = None
            else:
                result[field_name] = field_info.get_default(call_default_factory=True)

    return model.model_construct(**result)


def _build_partial_list(
    items: list,
    original_model: type[BaseModel] | None,
    field_name: str,
    tracker: JsonCompleteness,
    path: str,
    **kwargs: Any,
) -> list:
    """Build a partial list, validating complete items."""
    result = []

    item_type = None
    if original_model:
        field_info = original_model.model_fields.get(field_name)
        if field_info:
            field_type = field_info.annotation
            if get_origin(field_type) in UNION_ORIGINS:
                non_none_args = [
                    arg for arg in get_args(field_type) if arg is not type(None)
                ]
                # Only unwrap a nullable container with one unambiguous type.
                if len(non_none_args) == 1:
                    field_type = non_none_args[0]
            if get_origin(field_type) in (list, List):  # noqa: UP006
                args = get_args(field_type)
                if args:
                    item_type = args[0]

    for i, item in enumerate(items):
        item_path = f"{path}[{i}]"
        item_complete = tracker.is_path_complete(item_path)

        _item_model = _unwrap_optional_base_model(item_type) if item_type else None
        if item_complete and _item_model is not None and isinstance(item, dict):
            result.append(_item_model.model_validate(item, **kwargs))
            continue

        if _item_model is not None and isinstance(item, dict):
            result.append(
                _build_partial_object(item, _item_model, tracker, item_path, **kwargs)
            )
            continue

        result.append(item)

    return result


def _process_generic_arg(
    arg: Any,
    make_fields_optional: bool = False,
) -> Any:
    if _processing_models.get() is None:
        token = _processing_models.set(set())
        try:
            return _process_generic_arg(arg, make_fields_optional=make_fields_optional)
        finally:
            _processing_models.reset(token)

    processing_models = _processing_models.get()
    assert processing_models is not None
    arg_origin = get_origin(arg)

    if arg_origin is not None:
        # Handle any nested generic type (Union, List, Dict, etc.)
        nested_args = get_args(arg)
        modified_nested_args = tuple(
            _process_generic_arg(
                t,
                make_fields_optional=make_fields_optional,
            )
            for t in nested_args
        )
        # Special handling for Union types (types.UnionType isn't subscriptable)
        if arg_origin in UNION_ORIGINS:
            return _subscript_type(Union, modified_nested_args)

        return arg_origin[modified_nested_args]
    if isinstance(arg, type) and issubclass(arg, BaseModel):
        # Prevent infinite recursion for self-referential models
        if arg in processing_models:
            return arg  # Already processing this model, return unwrapped
        processing_models.add(arg)
        try:
            return (
                _make_partial_type(arg, make_fields_optional=True)
                if make_fields_optional
                else Partial[arg]
            )
        finally:
            processing_models.discard(arg)
    else:
        return arg


def _subscript_type(origin: Any, args: Any) -> Any:
    """Construct a runtime typing object from dynamically discovered arguments."""
    return origin[args]


def _make_optional_type(annotation: Any) -> Any:
    return _subscript_type(Optional, annotation)


def _make_partial_type(
    annotation: type[BaseModel], *, make_fields_optional: bool = False
) -> type[BaseModel]:
    key = (annotation, MakeFieldsOptional) if make_fields_optional else annotation
    return Partial.__class_getitem__(key)


def _make_field_optional(
    field: FieldInfo,
) -> tuple[Any, FieldInfo]:
    tmp_field = deepcopy(field)

    annotation = field.annotation

    # Handle generics (like List, Dict, Union, Literal, etc.)
    if get_origin(annotation) is not None:
        # Get the generic base (like List, Dict) and its arguments (like User in List[User])
        generic_base = get_origin(annotation)
        generic_args = get_args(annotation)

        modified_args = tuple(
            _process_generic_arg(arg, make_fields_optional=True) for arg in generic_args
        )

        # Reconstruct the generic type with modified arguments
        if generic_base is UNION_TYPE:
            tmp_field.annotation = _make_optional_type(reduce(or_, modified_args))
        else:
            tmp_field.annotation = (
                _make_optional_type(_subscript_type(generic_base, modified_args))
                if generic_base
                else None
            )
        tmp_field.default = None
        tmp_field.default_factory = None
    # If the field is a BaseModel, then recursively convert it's
    # attributes to optionals.
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        tmp_field.annotation = _make_optional_type(
            _make_partial_type(annotation, make_fields_optional=True)
        )
        tmp_field.default = {}
        tmp_field.default_factory = None
    else:
        tmp_field.annotation = _make_optional_type(field.annotation)
        tmp_field.default = None
        tmp_field.default_factory = None

    return tmp_field.annotation, tmp_field


class PartialBase(Generic[T_Model]):
    @classmethod
    @cache
    def get_partial_model(cls) -> type[T_Model]:
        """Return a partial model for holding incomplete streaming data.

        With completeness-based validation, we use model_construct() to build
        partial objects without validation. This method creates a model with
        all fields optional and stores a reference to the original model
        for validation when JSON is complete.
        """
        assert issubclass(cls, BaseModel), (
            f"{cls.__name__} must be a subclass of BaseModel"
        )

        model_name = (
            cls.__name__
            if cls.__name__.startswith("Partial")
            else f"Partial{cls.__name__}"
        )

        # Create partial model with optional fields
        partial_model = create_model(  # ty: ignore[no-matching-overload]
            model_name,
            __base__=cls,
            __module__=cls.__module__,
            **{
                field_name: _make_field_optional(field_info)
                for field_name, field_info in cls.model_fields.items()
            },
        )

        # Store reference to original model for validation of complete objects
        original = getattr(cls, "_original_model", cls)
        partial_model._original_model = original  # type: ignore[attr-defined]

        return partial_model

    @classmethod
    def from_streaming_response(
        cls,
        completion: Iterable[Any],
        stream_extractor: Callable[[Iterable[Any]], Generator[str, None, None]],
        chunk_parser: Callable[..., Generator[T_Model, None, None]] | None = None,
        **kwargs: Any,
    ) -> Generator[T_Model, None, None]:
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        json_chunks = stream_extractor(completion)
        parser = chunk_parser or cls.model_from_chunks
        yield from parser(json_chunks, **kwargs)

    @classmethod
    async def from_streaming_response_async(
        cls,
        completion: AsyncGenerator[Any, None],
        stream_extractor: Callable[
            [AsyncGenerator[Any, None]], AsyncGenerator[str, None]
        ],
        chunk_parser: Callable[..., AsyncGenerator[T_Model, None]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T_Model, None]:
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        json_chunks = stream_extractor(completion)
        parser = chunk_parser or cls.model_from_chunks_async
        async for item in parser(json_chunks, **kwargs):
            yield item

    @classmethod
    def model_from_chunks(
        cls, json_chunks: Iterable[Any], **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        potential_object = ""
        partial_model = cls.get_partial_model()
        # Always use trailing-strings mode to preserve incomplete data during streaming
        # PartialLiteralMixin is deprecated - completeness-based validation handles Literals
        partial_mode = "trailing-strings"
        final_obj = None
        for chunk in json_chunks:
            if chunk is None:
                continue
            if not isinstance(chunk, str):
                try:
                    chunk = str(chunk)
                except Exception:
                    continue
            potential_object += remove_control_chars(chunk)
            obj = process_potential_object(
                potential_object, partial_mode, partial_model, **kwargs
            )
            final_obj = obj
            yield obj

        # Final validation: only validate if the JSON is structurally complete
        # If JSON is incomplete (stream ended mid-object), skip validation.
        # Validate the accumulated JSON itself rather than a
        # model_dump(exclude_none=True) round-trip, which would strip fields
        # the model legitimately returned as null and make required-but-nullable
        # fields fail re-validation as "missing".
        if final_obj is not None:
            original_model = getattr(cls, "_original_model", None)
            if original_model is not None:
                json_str = potential_object.strip() or "{}"
                if is_json_complete(json_str):
                    original_model.model_validate(
                        from_json(json_str.encode()), **kwargs
                    )

    @classmethod
    async def model_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        potential_object = ""
        partial_model = cls.get_partial_model()
        # Always use trailing-strings mode to preserve incomplete data during streaming
        # PartialLiteralMixin is deprecated - completeness-based validation handles Literals
        partial_mode = "trailing-strings"
        final_obj = None
        async for chunk in json_chunks:
            if chunk is None:
                continue
            if not isinstance(chunk, str):
                try:
                    chunk = str(chunk)
                except Exception:
                    continue
            potential_object += remove_control_chars(chunk)
            obj = process_potential_object(
                potential_object, partial_mode, partial_model, **kwargs
            )
            final_obj = obj
            yield obj

        # Final validation: only validate if the JSON is structurally complete
        # If JSON is incomplete (stream ended mid-object), skip validation.
        # Validate the accumulated JSON itself rather than a
        # model_dump(exclude_none=True) round-trip, which would strip fields
        # the model legitimately returned as null and make required-but-nullable
        # fields fail re-validation as "missing".
        if final_obj is not None:
            original_model = getattr(cls, "_original_model", None)
            if original_model is not None:
                json_str = potential_object.strip() or "{}"
                if is_json_complete(json_str):
                    original_model.model_validate(
                        from_json(json_str.encode()), **kwargs
                    )

    @staticmethod
    def extract_json(
        completion: Iterable[Any],
        stream_extractor: Callable[[Iterable[Any]], Generator[str, None, None]] | Any,
        on_event: Callable[..., Any] | None = None,
    ) -> Generator[str, None, None]:
        from instructor.v2.core.mode import Mode

        if stream_extractor in {
            Mode.RESPONSES_TOOLS,
            Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        }:
            from openai.types.responses import (
                ResponseFunctionCallArgumentsDeltaEvent,
                ResponseReasoningSummaryTextDeltaEvent,
                ResponseReasoningSummaryTextDoneEvent,
            )

            for chunk in completion:
                if isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
                    yield chunk.delta
                elif on_event and isinstance(
                    chunk,
                    (
                        ResponseReasoningSummaryTextDeltaEvent,
                        ResponseReasoningSummaryTextDoneEvent,
                    ),
                ):
                    on_event(chunk)
            return
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        yield from stream_extractor(completion)

    @staticmethod
    async def extract_json_async(
        completion: AsyncGenerator[Any, None],
        stream_extractor: Callable[
            [AsyncGenerator[Any, None]], AsyncGenerator[str, None]
        ]
        | Any,
        on_event: Callable[..., Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        import inspect

        from instructor.v2.core.mode import Mode

        if stream_extractor in {
            Mode.RESPONSES_TOOLS,
            Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        }:
            from openai.types.responses import (
                ResponseFunctionCallArgumentsDeltaEvent,
                ResponseReasoningSummaryTextDeltaEvent,
                ResponseReasoningSummaryTextDoneEvent,
            )

            async for chunk in completion:
                if isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
                    yield chunk.delta
                elif on_event and isinstance(
                    chunk,
                    (
                        ResponseReasoningSummaryTextDeltaEvent,
                        ResponseReasoningSummaryTextDoneEvent,
                    ),
                ):
                    maybe_awaitable = on_event(chunk)
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable
            return
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        async for chunk in stream_extractor(completion):
            yield chunk


class Partial(Generic[T_Model]):
    """Generate a new class which has PartialBase as a base class.

    Notes:
        This will enable partial validation of the model while streaming.

    Example:
        Partial[SomeModel]
    """

    def __new__(
        cls,
        *args: object,  # noqa
        **kwargs: object,  # noqa
    ) -> Partial[T_Model]:
        """Cannot instantiate.

        Raises:
            TypeError: Direct instantiation not allowed.
        """
        raise TypeError("Cannot instantiate abstract Partial class.")

    def __init_subclass__(
        cls,
        *args: object,
        **kwargs: object,
    ) -> NoReturn:
        """Cannot subclass.

        Raises:
           TypeError: Subclassing not allowed.
        """
        raise TypeError(f"Cannot subclass {cls.__module__}.Partial")

    def __class_getitem__(
        cls,
        wrapped_class: type[T_Model] | tuple[type[T_Model], type[MakeFieldsOptional]],
    ) -> type[T_Model]:
        """Convert model to one that inherits from PartialBase.

        We don't make the fields optional at this point, we just wrap them with `Partial` so the names of the nested models will be
        `Partial{ModelName}`. We want the output of `model_json_schema()` to
        reflect the name change, but everything else should be the same as the
        original model. During validation, we'll generate a true partial model
        to support partially defined fields.

        """
        if _processing_models.get() is None:
            token = _processing_models.set(set())
            try:
                return cls.__class_getitem__(wrapped_class)
            finally:
                _processing_models.reset(token)

        processing_models = _processing_models.get()
        assert processing_models is not None

        make_fields_optional = None
        if isinstance(wrapped_class, tuple):
            wrapped_class, make_fields_optional = wrapped_class

        def _wrap_models(field: FieldInfo) -> tuple[object, FieldInfo]:
            tmp_field = deepcopy(field)

            annotation = field.annotation

            # Handle generics (like List, Dict, etc.)
            if get_origin(annotation) is not None:
                # Get the generic base (like List, Dict) and its arguments (like User in List[User])
                generic_base = get_origin(annotation)
                generic_args = get_args(annotation)

                modified_args = tuple(_process_generic_arg(arg) for arg in generic_args)

                # Reconstruct the generic type with modified arguments
                if generic_base is UNION_TYPE:
                    tmp_field.annotation = reduce(or_, modified_args)
                else:
                    tmp_field.annotation = (
                        generic_base[modified_args] if generic_base else None
                    )
            # If the field is a BaseModel, then recursively convert it's
            # attributes to optionals.
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                # Prevent infinite recursion for self-referential models
                if annotation in processing_models:
                    tmp_field.annotation = (
                        annotation  # Already processing, keep unwrapped
                    )
                else:
                    processing_models.add(annotation)
                    try:
                        tmp_field.annotation = Partial[annotation]
                    finally:
                        processing_models.discard(annotation)
            return tmp_field.annotation, tmp_field

        model_name = (
            wrapped_class.__name__
            if wrapped_class.__name__.startswith("Partial")
            else f"Partial{wrapped_class.__name__}"
        )

        partial_model = create_model(  # ty: ignore[no-matching-overload]
            model_name,
            __base__=(wrapped_class, PartialBase),
            __module__=wrapped_class.__module__,
            **{
                field_name: (
                    _make_field_optional(field_info)
                    if make_fields_optional is not None
                    else _wrap_models(field_info)
                )
                for field_name, field_info in wrapped_class.model_fields.items()
            },
        )

        # Store reference to original model for final validation
        partial_model._original_model = wrapped_class  # type: ignore[attr-defined]

        return partial_model
