"""Legacy BatchJob parsing rejects token-limited records."""

import json

from pydantic import BaseModel

from instructor.batch import BatchJob


class User(BaseModel):
    name: str


def _openai_line(custom_id: str, finish_reason: str) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "body": {
                    "choices": [
                        {
                            "finish_reason": finish_reason,
                            "message": {"content": '{"name": "Ada"}'},
                        }
                    ]
                }
            },
        }
    )


def _anthropic_line(custom_id: str, stop_reason: str) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "result": {
                "type": "succeeded",
                "message": {
                    "stop_reason": stop_reason,
                    "content": [{"type": "tool_use", "input": {"name": "Ada"}}],
                },
            },
        }
    )


def test_legacy_parser_rejects_truncated_records() -> None:
    content = "\n".join(
        [
            _openai_line("openai-ok", "stop"),
            _openai_line("openai-cut", "length"),
            _anthropic_line("anthropic-ok", "tool_use"),
            _anthropic_line("anthropic-cut", "max_tokens"),
        ]
    )

    parsed, errors = BatchJob.parse_from_string(content, User)

    assert parsed == [User(name="Ada"), User(name="Ada")]
    assert [error["custom_id"] for error in errors] == ["openai-cut", "anthropic-cut"]
