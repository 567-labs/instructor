# --------------------------------------------------------------------------------
# The following code is adapted from a comment on GitHub in the pydantic/pydantic repository by silviumarcu.
# Source: https://github.com/pydantic/pydantic/issues/6381#issuecomment-1831607091
#
# This code is used in accordance with the repository's license, and this reference
# serves as an acknowledgment of the original author's contribution to this project.
# --------------------------------------------------------------------------------

from __future__ import annotations

import json
import re
import sys
import types
from collections.abc import AsyncGenerator, Generator, Iterable
from copy import deepcopy
from functools import cache
from typing import (
    Any,
    Generic,
    Literal,
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

from instructor.mode import Mode
from instructor.utils import extract_json_from_stream, extract_json_from_stream_async

T_Model = TypeVar("T_Model", bound=BaseModel)

if sys.version_info >= (3, 10):
    # types.UnionType is only available in Python 3.10 and above
    UNION_ORIGINS = (Union, types.UnionType)
else:
    UNION_ORIGINS = (Union,)


class MakeFieldsOptional:
    pass


class PartialLiteralMixin:
    pass


def remove_control_chars(s):
    return re.sub(r"[\x00-\x1F\x7F-\x9F]", "", s)


def process_potential_object(potential_object, partial_mode, partial_model, **kwargs):
    obj = from_json(
        (potential_object.strip() or "{}").encode(), partial_mode=partial_mode
    )
    obj = partial_model.model_validate(obj, strict=None, **kwargs)
    return obj


def _process_generic_arg(
    arg: Any,
    make_fields_optional: bool = False,
) -> Any:
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
            return Union[modified_nested_args]  # type: ignore

        return arg_origin[modified_nested_args]
    else:
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return (
                Partial[arg, MakeFieldsOptional]  # type: ignore[valid-type]
                if make_fields_optional
                else Partial[arg]
            )
        else:
            return arg


def _make_field_optional(
    field: FieldInfo,
) -> tuple[Any, FieldInfo]:
    tmp_field = deepcopy(field)

    annotation = field.annotation

    # Check if the original field is already Optional
    is_optional = False
    original_annotation = annotation
    if get_origin(annotation) is Union:
        # Check if None is in the union args (meaning it's Optional)
        union_args = get_args(annotation)
        if type(None) in union_args:
            is_optional = True
            # Get the non-None type from the Union
            non_none_args = [arg for arg in union_args if arg is not type(None)]
            if len(non_none_args) == 1:
                annotation = non_none_args[0]

    # Handle generics (like List, Dict, etc.)
    if get_origin(annotation) is not None:
        generic_base = get_origin(annotation)
        generic_args = get_args(annotation)

        # Special handling for Literal types
        if generic_base is Literal:
            # For Literal types during streaming, we want to accept:
            # 1. The literal values themselves
            # 2. Any string (to handle partial values during streaming)
            # 3. None only if the original field was optional
            if is_optional or field.is_required() is False:
                tmp_field.annotation = Optional[Union[annotation, str]]  # type: ignore[assignment]
                tmp_field.default = None
            else:
                # Required literal field - don't allow None
                tmp_field.annotation = Union[annotation, str]  # type: ignore[assignment]
                tmp_field.default = ""  # Use empty string instead of None
        else:
            # Get the generic base (like List, Dict) and its arguments (like User in List[User])
            modified_args = tuple(
                _process_generic_arg(arg, make_fields_optional=True)
                for arg in generic_args
            )

            # Reconstruct the generic type with modified arguments
            tmp_field.annotation = (
                Optional[generic_base[modified_args]] if generic_base else None
            )
            tmp_field.default = None
    # If the field is a BaseModel, then recursively convert it's
    # attributes to optionals.
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        tmp_field.annotation = Optional[Partial[annotation, MakeFieldsOptional]]  # type: ignore[assignment, valid-type]
        tmp_field.default = {}
    else:
        tmp_field.annotation = (
            Optional[original_annotation]
            if original_annotation
            else Optional[field.annotation]
        )  # type: ignore[assignment]
        tmp_field.default = None

    return tmp_field.annotation, tmp_field  # type: ignore


class PartialBase(Generic[T_Model]):
    @classmethod
    @cache
    def get_partial_model(cls) -> type[T_Model]:
        """Return a partial model we can use to validate partial results."""
        assert issubclass(cls, BaseModel), (
            f"{cls.__name__} must be a subclass of BaseModel"
        )

        model_name = (
            cls.__name__
            if cls.__name__.startswith("Partial")
            else f"Partial{cls.__name__}"
        )

        return create_model(
            model_name,
            __base__=cls,
            __module__=cls.__module__,
            **{
                field_name: _make_field_optional(field_info)
                for field_name, field_info in cls.model_fields.items()
            },  # type: ignore[all]
        )

    @classmethod
    def from_streaming_response(
        cls, completion: Iterable[Any], mode: Mode, **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        json_chunks = cls.extract_json(completion, mode)

        if mode in {Mode.MD_JSON, Mode.GEMINI_TOOLS}:
            json_chunks = extract_json_from_stream(json_chunks)

        if mode == Mode.WRITER_TOOLS:
            yield from cls.writer_model_from_chunks(json_chunks, **kwargs)
        else:
            yield from cls.model_from_chunks(json_chunks, **kwargs)

    @classmethod
    async def from_streaming_response_async(
        cls, completion: AsyncGenerator[Any, None], mode: Mode, **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        json_chunks = cls.extract_json_async(completion, mode)

        if mode == Mode.MD_JSON:
            json_chunks = extract_json_from_stream_async(json_chunks)
        elif mode == Mode.WRITER_TOOLS:
            return cls.writer_model_from_chunks_async(json_chunks, **kwargs)

        return cls.model_from_chunks_async(json_chunks, **kwargs)

    @classmethod
    def writer_model_from_chunks(
        cls, json_chunks: Iterable[Any], **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        potential_object = ""
        partial_model = cls.get_partial_model()
        partial_mode = (
            "on" if issubclass(cls, PartialLiteralMixin) else "trailing-strings"
        )
        for chunk in json_chunks:
            if len(chunk) > len(potential_object):
                potential_object = chunk
            else:
                potential_object += chunk
            obj = from_json(
                (potential_object.strip() or "{}").encode(), partial_mode=partial_mode
            )
            obj = partial_model.model_validate(obj, strict=None, **kwargs)
            yield obj

    @classmethod
    async def writer_model_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        potential_object = ""
        partial_model = cls.get_partial_model()
        partial_mode = (
            "on" if issubclass(cls, PartialLiteralMixin) else "trailing-strings"
        )
        async for chunk in json_chunks:
            if len(chunk) > len(potential_object):
                potential_object = chunk
            else:
                potential_object += chunk
            obj = from_json(
                (potential_object.strip() or "{}").encode(), partial_mode=partial_mode
            )
            obj = partial_model.model_validate(obj, strict=None, **kwargs)
            yield obj

    @classmethod
    def model_from_chunks(
        cls, json_chunks: Iterable[Any], **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        potential_object = ""
        partial_model = cls.get_partial_model()
        partial_mode = (
            "on" if issubclass(cls, PartialLiteralMixin) else "trailing-strings"
        )
        chunk_buffer = []
        for chunk in json_chunks:
            chunk_buffer += chunk
            if len(chunk_buffer) < 2:
                continue
            potential_object += remove_control_chars("".join(chunk_buffer))
            chunk_buffer = []
            obj = process_potential_object(
                potential_object, partial_mode, partial_model, **kwargs
            )
            yield obj
        if chunk_buffer:
            potential_object += remove_control_chars(chunk_buffer[0])
            obj = process_potential_object(
                potential_object, partial_mode, partial_model, **kwargs
            )
            yield obj

    @classmethod
    async def model_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        potential_object = ""
        partial_model = cls.get_partial_model()
        partial_mode = (
            "on" if issubclass(cls, PartialLiteralMixin) else "trailing-strings"
        )
        async for chunk in json_chunks:
            potential_object += chunk
            obj = from_json(
                (potential_object.strip() or "{}").encode(), partial_mode=partial_mode
            )
            obj = partial_model.model_validate(obj, strict=None, **kwargs)
            yield obj

    @staticmethod
    def extract_json(
        completion: Iterable[Any], mode: Mode
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from various LLM provider streaming responses.

        Each provider has a different structure for streaming responses that needs
        specific handling to extract the relevant JSON data."""
        for chunk in completion:
            try:
                if mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
                    yield chunk.data.choices[0].delta.content
                if mode == Mode.MISTRAL_TOOLS:
                    if not chunk.data.choices[0].delta.tool_calls:
                        continue
                    yield chunk.data.choices[0].delta.tool_calls[0].function.arguments
                if mode == Mode.ANTHROPIC_JSON:
                    if json_chunk := chunk.delta.text:
                        yield json_chunk
                if mode == Mode.ANTHROPIC_TOOLS:
                    yield chunk.delta.partial_json
                if mode == Mode.VERTEXAI_JSON:
                    yield chunk.candidates[0].content.parts[0].text
                if mode == Mode.VERTEXAI_TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )

                if mode == Mode.GENAI_STRUCTURED_OUTPUTS:
                    yield chunk.text
                if mode == Mode.GENAI_TOOLS:
                    fc = chunk.candidates[0].content.parts[0].function_call.args
                    yield json.dumps(fc)
                if mode == Mode.GEMINI_JSON:
                    yield chunk.text
                if mode == Mode.GEMINI_TOOLS:
                    resp = chunk.candidates[0].content.parts[0].function_call
                    resp_dict = type(resp).to_dict(resp)  # type:ignore
                    if "args" in resp_dict:
                        yield json.dumps(resp_dict["args"])
                elif mode in {
                    Mode.RESPONSES_TOOLS,
                    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                }:
                    from openai.types.responses import (
                        ResponseFunctionCallArgumentsDeltaEvent,
                    )

                    if isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
                        yield chunk.delta

                elif chunk.choices:
                    if mode == Mode.FUNCTIONS:
                        Mode.warn_mode_functions_deprecation()
                        if json_chunk := chunk.choices[0].delta.function_call.arguments:
                            yield json_chunk
                    elif mode in {
                        Mode.JSON,
                        Mode.MD_JSON,
                        Mode.JSON_SCHEMA,
                        Mode.CEREBRAS_JSON,
                        Mode.FIREWORKS_JSON,
                        Mode.PERPLEXITY_JSON,
                    }:
                        if json_chunk := chunk.choices[0].delta.content:
                            yield json_chunk
                    elif mode in {
                        Mode.TOOLS,
                        Mode.TOOLS_STRICT,
                        Mode.FIREWORKS_TOOLS,
                        Mode.WRITER_TOOLS,
                    }:
                        if json_chunk := chunk.choices[0].delta.tool_calls:
                            if json_chunk[0].function.arguments:
                                yield json_chunk[0].function.arguments
                    else:
                        raise NotImplementedError(
                            f"Mode {mode} is not supported for MultiTask streaming"
                        )
            except AttributeError:
                pass

    @staticmethod
    async def extract_json_async(
        completion: AsyncGenerator[Any, None], mode: Mode
    ) -> AsyncGenerator[str, None]:
        async for chunk in completion:
            try:
                if mode == Mode.ANTHROPIC_JSON:
                    if json_chunk := chunk.delta.text:
                        yield json_chunk
                if mode == Mode.ANTHROPIC_TOOLS:
                    yield chunk.delta.partial_json
                if mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
                    yield chunk.data.choices[0].delta.content
                if mode == Mode.MISTRAL_TOOLS:
                    if not chunk.data.choices[0].delta.tool_calls:
                        continue
                    yield chunk.data.choices[0].delta.tool_calls[0].function.arguments
                if mode == Mode.VERTEXAI_JSON:
                    yield chunk.candidates[0].content.parts[0].text
                if mode == Mode.VERTEXAI_TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                if mode == Mode.GENAI_STRUCTURED_OUTPUTS:
                    yield chunk.text
                if mode == Mode.GENAI_TOOLS:
                    fc = chunk.candidates[0].content.parts[0].function_call.args
                    yield json.dumps(fc)
                if mode == Mode.GEMINI_JSON:
                    yield chunk.text
                if mode == Mode.GEMINI_TOOLS:
                    resp = chunk.candidates[0].content.parts[0].function_call
                    resp_dict = type(resp).to_dict(resp)  # type:ignore
                    if "args" in resp_dict:
                        yield json.dumps(resp_dict["args"])

                if mode in {
                    Mode.RESPONSES_TOOLS,
                    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                }:
                    from openai.types.responses import (
                        ResponseFunctionCallArgumentsDeltaEvent,
                    )

                    if isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
                        yield chunk.delta
                elif chunk.choices:
                    if mode == Mode.FUNCTIONS:
                        Mode.warn_mode_functions_deprecation()
                        if json_chunk := chunk.choices[0].delta.function_call.arguments:
                            yield json_chunk
                    elif mode in {
                        Mode.JSON,
                        Mode.MD_JSON,
                        Mode.JSON_SCHEMA,
                        Mode.CEREBRAS_JSON,
                        Mode.FIREWORKS_JSON,
                        Mode.PERPLEXITY_JSON,
                    }:
                        if json_chunk := chunk.choices[0].delta.content:
                            yield json_chunk
                    elif mode in {
                        Mode.TOOLS,
                        Mode.TOOLS_STRICT,
                        Mode.FIREWORKS_TOOLS,
                        Mode.WRITER_TOOLS,
                    }:
                        if json_chunk := chunk.choices[0].delta.tool_calls:
                            if json_chunk[0].function.arguments:
                                yield json_chunk[0].function.arguments
                    else:
                        raise NotImplementedError(
                            f"Mode {mode} is not supported for MultiTask streaming"
                        )
            except AttributeError:
                pass


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
                tmp_field.annotation = (
                    generic_base[modified_args] if generic_base else None
                )
            # If the field is a BaseModel, then recursively convert it's
            # attributes to optionals.
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                tmp_field.annotation = Partial[annotation]
            return tmp_field.annotation, tmp_field

        model_name = (
            wrapped_class.__name__
            if wrapped_class.__name__.startswith("Partial")
            else f"Partial{wrapped_class.__name__}"
        )

        return create_model(
            model_name,
            __base__=(wrapped_class, PartialBase),  # type: ignore
            __module__=wrapped_class.__module__,
            **{
                field_name: (
                    _make_field_optional(field_info)
                    if make_fields_optional is not None
                    else _wrap_models(field_info)
                )
                for field_name, field_info in wrapped_class.model_fields.items()
            },  # type: ignore
        )
