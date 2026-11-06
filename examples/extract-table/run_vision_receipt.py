from pydantic import BaseModel, model_validator
from openai import OpenAI
import instructor


client = instructor.from_openai(
    client=OpenAI(),
    mode=instructor.Mode.TOOLS,
)


class Item(BaseModel):
    name: str
    price: float
    quantity: int


class Receipt(BaseModel):
    items: list[Item]
    total: float

    @model_validator(mode="after")
    def check_total(cls, values: "Receipt"):
        items = values.items
        total = values.total
        calculated_total = sum(item.price * item.quantity for item in items)
        if calculated_total != total:
            raise ValueError(
                f"Total {total} does not match the sum of item prices {calculated_total}"
            )
        return values


def extract(url: str) -> Receipt:
    return client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4000,
        response_model=Receipt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                    {
                        "type": "text",
                        "text": "Analyze the image and return the items in the receipt and the total amount.",
                    },
                ],
            }
        ],
    )


# URLs of images containing receipts. Exhibits the use of the model validator to check the total amount.
urls = [
    "https://templates.mediamodifier.com/645124ff36ed2f5227cbf871/supermarket-receipt-template.jpg",
    "https://ocr.space/Content/Images/receipt-ocr-original.jpg",
]

for url in urls:
    receipt = extract(url)
    print(receipt)
