#!/usr/bin/env python3

from decimal import Decimal
from pydantic import BaseModel, field_validator
import instructor


class Receipt(BaseModel):
    item: str
    price: Decimal

    @field_validator("price", mode="before")
    @classmethod
    def parse_price(cls, v):
        if isinstance(v, str):
            return Decimal(v)
        return v


if __name__ == "__main__":
    client = instructor.from_provider("openai/gpt-4.1-mini")

    receipt = client.chat.completions.create(
        messages=[{"role": "user", "content": "Coffee costs $4.99"}],
        response_model=Receipt,
    )

    print(f"Item: {receipt.item}")
    print(f"Price: {receipt.price}")  # Decimal('4.99')
    print(f"Type: {type(receipt.price)}")  # <class 'decimal.Decimal'>

    # Test precision
    total = receipt.price * 2
    print(f"Total for 2 items: {total}")  # Decimal('9.98')
