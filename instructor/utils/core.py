"""Compatibility exports for v2-owned utility helpers."""

from instructor.v2.core.json import (  # noqa: F401
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)
from instructor.v2.core.messages import (  # noqa: F401
    dump_message,
    extract_messages,
    get_message_content,
    merge_consecutive_messages,
)
from instructor.v2.core.response_model import (  # noqa: F401
    is_simple_type,
    is_typed_dict,
    prepare_response_model,
)
from instructor.v2.core.usage import update_total_usage  # noqa: F401
from instructor.v2.core.utils import (  # noqa: F401
    classproperty,
    disable_pydantic_error_url,
    is_async,
)

__all__ = [
    "classproperty",
    "disable_pydantic_error_url",
    "dump_message",
    "extract_json_from_codeblock",
    "extract_json_from_stream",
    "extract_json_from_stream_async",
    "extract_messages",
    "get_message_content",
    "is_async",
    "is_simple_type",
    "is_typed_dict",
    "merge_consecutive_messages",
    "prepare_response_model",
    "update_total_usage",
]
