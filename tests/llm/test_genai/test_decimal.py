import pytest
from decimal import Decimal
from pydantic import BaseModel, field_validator
import instructor
from .util import models, modes


class Receipt(BaseModel):
    item: str
    quantity: int
    price: Decimal
    total: Decimal

    @field_validator("price", "total", mode="before")
    @classmethod
    def parse_decimals(cls, v):
        if isinstance(v, (str, float, int)):
            return Decimal(str(v))
        return v


class Invoice(BaseModel):
    receipts: list[Receipt]
    grand_total: Decimal

    @field_validator("grand_total", mode="before")
    @classmethod
    def parse_grand_total(cls, v):
        if isinstance(v, (str, float, int)):
            return Decimal(str(v))
        return v


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_decimal_extraction(client, model, mode):
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "I bought 2 apples for $1.50 each and 3 bananas for $0.75 each. Calculate the total.",
            },
        ],
        response_model=Invoice,
    )
    assert isinstance(response, Invoice)
    assert len(response.receipts) == 2

    # Check apple receipt
    apple_receipt = next(
        (r for r in response.receipts if "apple" in r.item.lower()), None
    )
    assert apple_receipt is not None
    assert apple_receipt.quantity == 2
    assert isinstance(apple_receipt.price, Decimal)
    assert isinstance(apple_receipt.total, Decimal)

    # Check banana receipt
    banana_receipt = next(
        (r for r in response.receipts if "banana" in r.item.lower()), None
    )
    assert banana_receipt is not None
    assert banana_receipt.quantity == 3
    assert isinstance(banana_receipt.price, Decimal)
    assert isinstance(banana_receipt.total, Decimal)

    # Check grand total
    assert isinstance(response.grand_total, Decimal)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
async def test_decimal_extraction_async(aclient, model, mode):
    aclient = instructor.from_provider(f"google/{model}", mode=mode, async_client=True)
    response = await aclient.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "I bought 1 coffee for $4.25 and 1 muffin for $2.75. What's the total?",
            },
        ],
        response_model=Invoice,
    )
    assert isinstance(response, Invoice)
    assert len(response.receipts) == 2

    # Check that all decimal fields are proper Decimal instances
    for receipt in response.receipts:
        assert isinstance(receipt.price, Decimal)
        assert isinstance(receipt.total, Decimal)

    assert isinstance(response.grand_total, Decimal)


class SimpleProduct(BaseModel):
    name: str
    price: Decimal

    @field_validator("price", mode="before")
    @classmethod
    def parse_price(cls, v):
        if isinstance(v, (str, float, int)):
            return Decimal(str(v))
        return v


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_simple_decimal_extraction(client, model, mode):
    """Test simple decimal extraction to ensure schema conversion works"""
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "The laptop costs $999.99",
            },
        ],
        response_model=SimpleProduct,
    )
    assert isinstance(response, SimpleProduct)
    assert response.name.lower() == "laptop"
    assert isinstance(response.price, Decimal)
    assert response.price == Decimal("999.99")
