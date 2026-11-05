import pytest
from pydantic import BaseModel, ValidationError, field_validator

from instructor.v2.core import utils

pytestmark = pytest.mark.usefixtures("disabled_error_urls")


@pytest.fixture
def disabled_error_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_validation_error_original_str", None)
    monkeypatch.setattr(ValidationError, "__str__", ValidationError.__str__)
    utils.disable_pydantic_error_url()


@pytest.mark.parametrize(
    "value", ["https://errors.pydantic.dev/custom", "not-a-number"]
)
def test_error_url_filter_preserves_input_details(
    value: str,
) -> None:
    class Number(BaseModel):
        value: int

    with pytest.raises(ValidationError) as error:
        Number.model_validate({"value": value})

    message = str(error.value)
    assert "Input should be a valid integer" in message
    assert f"input_value='{value}'" in message
    assert "For further information visit" not in message


@pytest.mark.parametrize(
    "message",
    [
        "See https://errors.pydantic.dev/custom for this value",
        "Rejected by custom policy\n"
        "For further information visit https://errors.pydantic.dev/custom-policy",
        "Rejected by custom policy\n"
        "For further information visit https://errors.pydantic.dev/custom-policy\n"
        "Contact support.",
    ],
)
def test_error_url_filter_preserves_custom_messages(message: str) -> None:
    class Number(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def reject_value(cls, _value: int) -> int:
            raise ValueError(message)

    with pytest.raises(ValidationError) as error:
        Number(value=1)

    rendered = str(error.value)
    assert message in rendered
    assert "input_value=1" in rendered
    assert "\n    For further information visit" not in rendered
