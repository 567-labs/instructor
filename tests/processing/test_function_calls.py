from typing import Any, TypeVar, cast
import pytest
from anthropic.types import Message, TextBlock, Usage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message import FunctionCall as OpenAIFunctionCall
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import BaseModel, ValidationError

import instructor
from instructor import ResponseSchema, response_schema, OpenAISchema, openai_schema
from instructor.core.exceptions import IncompleteOutputException
from instructor.utils import disable_pydantic_error_url

T = TypeVar("T")


class _FunctionCallTestModel(ResponseSchema):  # type: ignore[misc]
    name: str = "TestModel"
    data: str


@pytest.fixture  # type: ignore[misc]
def test_model() -> type[_FunctionCallTestModel]:
    return _FunctionCallTestModel


@pytest.fixture  # type: ignore[misc]
def mock_completion(request: Any) -> ChatCompletion:
    finish_reason = "stop"
    data_content = '{\n"data": "complete data"\n}'

    if hasattr(request, "param"):
        params = cast(dict[str, Any], request.param)
        finish_reason = params.get("finish_reason", finish_reason)
        data_content = params.get("data_content", data_content)

    completion = ChatCompletion(
        id="test_id",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=data_content,
                    function_call=OpenAIFunctionCall(
                        name="TestModel",
                        arguments=data_content,
                    ),
                ),
                finish_reason=finish_reason,
                logprobs=None,
            )
        ],
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
    )

    return completion


@pytest.fixture  # type: ignore[misc]
def mock_anthropic_message(request: Any) -> Message:
    data_content = '{\n"data": "Claude says hi"\n}'
    if hasattr(request, "param"):
        params = cast(dict[str, Any], request.param)
        data_content = params.get("data_content", data_content)
    return Message(
        id="test_id",
        content=[TextBlock(type="text", text=data_content)],
        model="claude-3-5-haiku-20241022",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=100,
            output_tokens=100,
        ),
    )


def test_response_schema() -> None:
    @response_schema
    class Dataframe(BaseModel):  # type: ignore[misc]
        """
        Class representing a dataframe. This class is used to convert
        data into a frame that can be used by pandas.
        """

        data: str
        columns: str

        def to_pandas(self) -> None:
            pass

    assert hasattr(Dataframe, "openai_schema")
    assert hasattr(Dataframe, "from_response")
    assert hasattr(Dataframe, "to_pandas")
    assert Dataframe.openai_schema["name"] == "Dataframe"


def test_response_schema_raises_error() -> None:
    with pytest.raises(
        TypeError,
        match="response_model must be a subclass of pydantic.BaseModel",
    ):

        @response_schema  # ty: ignore[invalid-argument-type]
        class Dummy:
            pass


def test_openai_schema_alias() -> None:
    """Test that OpenAISchema alias still works for backward compatibility."""

    @openai_schema
    class Dataframe(BaseModel):  # type: ignore[misc]
        """
        Class representing a dataframe. This class is used to convert
        data into a frame that can be used by pandas.
        """

        data: str
        columns: str

    assert hasattr(Dataframe, "openai_schema")
    assert hasattr(Dataframe, "from_response")
    assert Dataframe.openai_schema["name"] == "Dataframe"


def test_openai_schema_alias_raises_error() -> None:
    """Test that openai_schema alias still works for backward compatibility."""
    with pytest.raises(
        TypeError,
        match="response_model must be a subclass of pydantic.BaseModel",
    ):

        @openai_schema  # ty: ignore[invalid-argument-type]
        class Dummy:
            pass


def test_no_docstring() -> None:
    class Dummy(ResponseSchema):  # type: ignore[misc]
        attr: str


def test_openai_schema_backward_compat() -> None:
    """Test that OpenAISchema alias still works for backward compatibility."""

    class Dummy(OpenAISchema):  # type: ignore[misc]
        attr: str

    assert (
        Dummy.openai_schema["description"]
        == "Correctly extracted `Dummy` with all the required parameters with correct types"
    )


@pytest.mark.parametrize(
    "mock_completion",
    [{"finish_reason": "length", "data_content": '{\n"data": "incomplete dat"\n}'}],
    indirect=True,
)  # type: ignore[misc]
def test_incomplete_output_exception(
    test_model: type[_FunctionCallTestModel], mock_completion: ChatCompletion
) -> None:
    with pytest.raises(IncompleteOutputException):
        test_model.from_response(mock_completion, mode=instructor.Mode.FUNCTIONS)


def test_complete_output_no_exception(
    test_model: type[_FunctionCallTestModel], mock_completion: ChatCompletion
) -> None:
    test_model_instance = test_model.from_response(
        mock_completion, mode=instructor.Mode.FUNCTIONS
    )
    assert test_model_instance.data == "complete data"


@pytest.mark.parametrize(
    "mock_completion",
    [{"finish_reason": "length", "data_content": '{\n"data": "incomplete dat"\n}'}],
    indirect=True,
)  # type: ignore[misc]
def test_incomplete_output_exception_raise(
    test_model: type[_FunctionCallTestModel], mock_completion: ChatCompletion
) -> None:
    with pytest.raises(IncompleteOutputException):
        test_model.from_response(mock_completion, mode=instructor.Mode.TOOLS)


def test_anthropic_no_exception(
    test_model: type[_FunctionCallTestModel], mock_anthropic_message: Message
) -> None:
    test_model_instance = test_model.from_response(
        cast(Any, mock_anthropic_message),
        mode=instructor.Mode.ANTHROPIC_JSON,
    )
    assert test_model_instance.data == "Claude says hi"


@pytest.mark.parametrize(
    "mock_anthropic_message",
    [{"data_content": '{\n"data": "Claude likes\ncontrol\ncharacters"\n}'}],
    indirect=True,
)  # type: ignore[misc]
def test_control_characters_not_allowed_in_anthropic_json_strict_mode(
    test_model: type[_FunctionCallTestModel], mock_anthropic_message: Message
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        test_model.from_response(
            cast(Any, mock_anthropic_message),
            mode=instructor.Mode.ANTHROPIC_JSON,
            strict=True,
        )

    # https://docs.pydantic.dev/latest/errors/validation_errors/#json_invalid
    exc = exc_info.value
    assert len(exc.errors()) == 1
    assert exc.errors()[0]["type"] == "json_invalid"
    assert "control character" in exc.errors()[0]["msg"]


@pytest.mark.parametrize(
    "mock_anthropic_message",
    [{"data_content": '{\n"data": "Claude likes\ncontrol\ncharacters"\n}'}],
    indirect=True,
)  # type: ignore[misc]
def test_control_characters_allowed_in_anthropic_json_non_strict_mode(
    test_model: type[_FunctionCallTestModel], mock_anthropic_message: Message
) -> None:
    test_model_instance = test_model.from_response(
        cast(Any, mock_anthropic_message),
        mode=instructor.Mode.ANTHROPIC_JSON,
        strict=False,
    )
    assert test_model_instance.data == "Claude likes\ncontrol\ncharacters"


def test_pylance_url_config() -> None:
    import sys

    if sys.version_info >= (3, 11):
        reason = (
            "This test seems to fail on 3.11 but passes on 3.10 and 3.9. I "
            "suspect it's due to the ordering of tests - "
            "https://github.com/pydantic/pydantic-core/blob/"
            "e3eff5cb8a6dae8914e3831b00c690d9dee4b740/python/pydantic_core/"
            "_pydantic_core.pyi#L820C9-L829C12"
        )
        raise pytest.skip.Exception(reason)

    class Model(BaseModel):
        list_of_ints: list[int]
        a_float: float

    disable_pydantic_error_url()
    data = dict(list_of_ints=["1", 2, "bad"], a_float="Not a float")

    with pytest.raises(ValidationError) as exc_info:
        Model.model_validate(data)

    assert "https://errors.pydantic.dev" not in str(exc_info.value)


def test_refusal_attribute(test_model: type[_FunctionCallTestModel]):
    completion = ChatCompletion(
        id="test_id",
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    content="test_content",
                    refusal="test_refusal",
                    role="assistant",
                    tool_calls=[],
                ),
                finish_reason="stop",
                logprobs=None,
            )
        ],
    )

    try:
        test_model.from_response(completion, mode=instructor.Mode.TOOLS)
    except Exception as e:
        assert "Unable to generate a response due to test_refusal" in str(e)


def test_no_refusal_attribute(test_model: type[_FunctionCallTestModel]):
    completion = ChatCompletion(
        id="test_id",
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    content="test_content",
                    refusal=None,
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="test_id",
                            function=Function(
                                name="TestModel",
                                arguments='{"data": "test_data", "name": "TestModel"}',
                            ),
                            type="function",
                        )
                    ],
                ),
                finish_reason="stop",
                logprobs=None,
            )
        ],
    )

    resp = test_model.from_response(completion, mode=instructor.Mode.TOOLS)
    assert resp.data == "test_data"
    assert resp.name == "TestModel"


def test_missing_refusal_attribute(test_model: type[_FunctionCallTestModel]):
    message_without_refusal_attribute = ChatCompletionMessage(
        content="test_content",
        refusal="test_refusal",
        role="assistant",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="test_id",
                function=Function(
                    name="TestModel",
                    arguments='{"data": "test_data", "name": "TestModel"}',
                ),
                type="function",
            )
        ],
    )

    del message_without_refusal_attribute.refusal
    assert not hasattr(message_without_refusal_attribute, "refusal")

    completion = ChatCompletion(
        id="test_id",
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=message_without_refusal_attribute,
                finish_reason="stop",
                logprobs=None,
            )
        ],
    )

    resp = test_model.from_response(completion, mode=instructor.Mode.TOOLS)
    assert resp.data == "test_data"
    assert resp.name == "TestModel"
