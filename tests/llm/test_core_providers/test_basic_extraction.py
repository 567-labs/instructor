"""
Basic extraction tests that run across all core providers.

Tests basic functionality like simple extraction, lists, and nested models.
"""

from pydantic import BaseModel, Field
import pytest
import instructor


class User(BaseModel):
    name: str
    age: int


class UserList(BaseModel):
    users: list[User]


class Address(BaseModel):
    street: str
    city: str
    country: str


class UserWithAddress(BaseModel):
    name: str
    age: int
    address: Address


@pytest.mark.asyncio
async def test_simple_extraction(provider_config):
    """Test simple single object extraction."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    user = await client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    )

    assert isinstance(user, User)
    assert user.name == "Jason"
    assert user.age == 25


@pytest.mark.asyncio
async def test_list_extraction(provider_config):
    """Test extracting multiple objects in a list."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    result = await client.create(
        response_model=list[User],
        messages=[
            {
                "role": "user",
                "content": "Extract: Alice is 30, Bob is 25, Charlie is 35",
            }
        ],
    )

    assert isinstance(result, list)
    assert len(result) == 3
    assert {user.name for user in result} == {"Alice", "Bob", "Charlie"}
    assert {user.age for user in result} == {30, 25, 35}


@pytest.mark.asyncio
async def test_nested_model_extraction(provider_config):
    """Test extracting nested models."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    user = await client.create(
        response_model=UserWithAddress,
        messages=[
            {
                "role": "user",
                "content": "Extract: John Doe, 28 years old, lives at 123 Main St, Springfield, USA",
            }
        ],
    )

    assert isinstance(user, UserWithAddress)
    assert user.name == "John Doe"
    assert user.age == 28
    assert isinstance(user.address, Address)
    assert user.address.street == "123 Main St"
    assert user.address.city == "Springfield"
    assert user.address.country == "USA"


@pytest.mark.asyncio
async def test_extraction_with_field_descriptions(provider_config):
    """Test extraction with Pydantic Field descriptions."""
    model, mode = provider_config

    class Product(BaseModel):
        name: str = Field(description="Name of the product")
        price: float = Field(description="Price in USD")
        in_stock: bool = Field(description="Whether the product is in stock")

    client = instructor.from_provider(model, mode=mode, async_client=True)

    product = await client.create(
        response_model=Product,
        messages=[
            {
                "role": "user",
                "content": "iPhone 15 Pro costs $999 and is currently available",
            }
        ],
    )

    assert isinstance(product, Product)
    assert "iphone" in product.name.lower() or "iPhone" in product.name
    assert product.price == 999.0
    assert product.in_stock is True
