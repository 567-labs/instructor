from instructor.processing.response import handle_response_model
from instructor.v2.core.mode import reset_deprecated_mode_warnings
from pydantic import BaseModel, Field
from typing import Any
import instructor
import pytest

modes = [
    instructor.Mode.ANTHROPIC_JSON,
    instructor.Mode.JSON,
    instructor.Mode.MD_JSON,
    instructor.Mode.GEMINI_JSON,
    instructor.Mode.VERTEXAI_JSON,
]

deprecated_modes = {
    instructor.Mode.ANTHROPIC_JSON,
    instructor.Mode.GEMINI_JSON,
    instructor.Mode.VERTEXAI_JSON,
}


def get_tool_definition(
    response_model: type[BaseModel], mode: instructor.Mode, **kwargs: Any
) -> dict[str, Any]:
    if mode in deprecated_modes:
        reset_deprecated_mode_warnings()
        try:
            with pytest.warns(
                DeprecationWarning, match=rf"Mode\.{mode.name} is deprecated"
            ):
                _, tool_definition = handle_response_model(
                    response_model, mode=mode, **kwargs
                )
            return tool_definition
        finally:
            reset_deprecated_mode_warnings()

    _, tool_definition = handle_response_model(response_model, mode=mode, **kwargs)
    return tool_definition


def get_system_prompt(user_tool_definition, mode):
    if mode == instructor.Mode.ANTHROPIC_JSON:
        system = user_tool_definition["system"]
        # Handle both string and list[dict] formats
        if isinstance(system, list):
            return "".join(block.get("text", "") for block in system)
        return system
    elif mode == instructor.Mode.GEMINI_JSON:
        return "\n".join(user_tool_definition["contents"][0]["parts"])
    elif mode == instructor.Mode.VERTEXAI_JSON:
        return str(user_tool_definition["generation_config"])
    return user_tool_definition["messages"][0]["content"]


@pytest.mark.parametrize("mode", modes)
def test_json_preserves_description_of_non_english_characters_in_json_mode(
    mode,
) -> None:
    messages = [
        {
            "role": "user",
            "content": "Extract the user from the text : 张三 20岁",
        }
    ]

    class User(BaseModel):
        name: str = Field(description="用户的名字")
        age: int = Field(description="用户的年龄")

    user_tool_definition = get_tool_definition(User, mode=mode, messages=messages)

    system_prompt = get_system_prompt(user_tool_definition, mode)
    assert "用户的名字" in system_prompt
    assert "用户的年龄" in system_prompt

    user_tool_definition = get_tool_definition(
        User,
        mode=mode,
        system="你是一个AI助手",
        messages=messages,
    )
    system_prompt = get_system_prompt(user_tool_definition, mode)
    assert "用户的名字" in system_prompt
    assert "用户的年龄" in system_prompt
