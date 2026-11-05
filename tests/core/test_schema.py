from typing import TypeVar


from datetime import datetime, date, time
from instructor import openai_schema
from decimal import Decimal
from uuid import UUID
from typing import Annotated, Union, Optional, Literal, Any
from collections import OrderedDict
from pydantic import BaseModel, Field

T = TypeVar("T")


def test_annotation_schema():
    class User(BaseModel):
        details: dict[
            Annotated[str, Field(description="User name", min_length=1)],
            Annotated[int, Field(description="User ID", gt=3)],
        ] = Field(max_length=1)

    assert openai_schema(User).model_json_schema() == User.model_json_schema()


class User(BaseModel):
    name: str
    age: int


class AdminUser(BaseModel):
    organization: str
    name: str
    email: str


def test_new_union_types():
    class Users(BaseModel):
        users: list[AdminUser | User]

    assert openai_schema(Users).model_json_schema() == Users.model_json_schema()


def test_old_union_type():
    class UsersOldUnion(BaseModel):
        users: list[Union[AdminUser, User]]

    assert (
        openai_schema(UsersOldUnion).model_json_schema()
        == UsersOldUnion.model_json_schema()
    )


def test_tuple_with_multiple_args():
    class TupleModel(BaseModel):
        coordinates: tuple[int, int, int]
        name_and_age: tuple[str, int]

    assert (
        openai_schema(TupleModel).model_json_schema() == TupleModel.model_json_schema()
    )


def test_dict_with_multiple_value_types():
    from collections import OrderedDict

    class DictModel(BaseModel):
        regular_dict: dict[str, Union[int, str]]
        ordered_dict: OrderedDict[str, Union[float, bool]]

    assert openai_schema(DictModel).model_json_schema() == DictModel.model_json_schema()


def test_nested_complex_types():
    class ComplexModel(BaseModel):
        nested_tuple_dict: dict[str, tuple[int, str, bool]]
        list_of_dicts: list[dict[str, Union[int, str]]]

    assert (
        openai_schema(ComplexModel).model_json_schema()
        == ComplexModel.model_json_schema()
    )


def test_openai_schema_tuple_mapping():
    class TestModel(BaseModel):
        field: tuple[str, int, int]

    assert openai_schema(TestModel).model_json_schema() == TestModel.model_json_schema()


def test_openai_schema_dict_mapping():
    class TestModel(BaseModel):
        field: dict[str, str]

    assert openai_schema(TestModel).model_json_schema() == TestModel.model_json_schema()


def test_openai_schema_ordered_dict_mapping():
    class TestModel(BaseModel):
        field: OrderedDict[str, int]

    assert openai_schema(TestModel).model_json_schema() == TestModel.model_json_schema()


def test_openai_schema_supports_optional_none_310():
    class DummyWithOptionalNone(BaseModel):
        """
        Class with a single attribute that can be a string or None.
        Validates support of UnionType in schema generation.
        """

        attr: str | None

    assert (
        openai_schema(DummyWithOptionalNone).model_json_schema()
        == DummyWithOptionalNone.model_json_schema()
    )


def test_openai_schema_supports_optional_none() -> None:
    class DummyWithOptionalNone(BaseModel):
        """
        Class with a single attribute that can be a string or None.
        Validates support of UnionType in schema generation.
        """

        attr: Optional[str]  # In python 3.10+ this is written as `attr: str | None`
        attr2: Union[str, None]

    assert (
        openai_schema(DummyWithOptionalNone).model_json_schema()
        == DummyWithOptionalNone.model_json_schema()
    )


def test_default_values_and_validators():
    class UserWithDefaults(BaseModel):
        name: str = "John Doe"
        age: int = Field(default=30, ge=0)

    assert (
        openai_schema(UserWithDefaults).model_json_schema()
        == UserWithDefaults.model_json_schema()
    )


def test_inheritance():
    class BaseUser(BaseModel):
        name: str

    class ExtendedUser(BaseUser):
        age: int

    assert (
        openai_schema(ExtendedUser).model_json_schema()
        == ExtendedUser.model_json_schema()
    )


def test_alias_and_field_customization():
    class AliasModel(BaseModel):
        actual_name: str = Field(..., alias="name")
        age: int = Field(..., title="User Age", description="The age of the user")

    assert (
        openai_schema(AliasModel).model_json_schema() == AliasModel.model_json_schema()
    )


def test_standard_python_types():
    class StandardTypesModel(BaseModel):
        timestamp: datetime
        date_field: date
        time_field: time
        price: Decimal
        unique_id: UUID

    assert (
        openai_schema(StandardTypesModel).model_json_schema()
        == StandardTypesModel.model_json_schema()
    )


def test_any_type():
    class AnyTypeModel(BaseModel):
        any_field: Any

    assert (
        openai_schema(AnyTypeModel).model_json_schema()
        == AnyTypeModel.model_json_schema()
    )


def test_literal_type():
    class LiteralTypeModel(BaseModel):
        status: Literal["active", "inactive", "pending"]

    assert (
        openai_schema(LiteralTypeModel).model_json_schema()
        == LiteralTypeModel.model_json_schema()
    )


def test_str_any_dict():
    class ChatResponse(BaseModel):
        action_data: dict[str, Any] | None = Field(
            default=None,
            description="The required data for the action that will be performed.",
        )

        content: str = Field(description="A contextual response to the user's message.")

    assert (
        openai_schema(ChatResponse).model_json_schema()
        == ChatResponse.model_json_schema()
    )
