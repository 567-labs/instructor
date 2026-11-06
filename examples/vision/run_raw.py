from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()


class SearchQuery(BaseModel):
    product_name: str
    query: str = Field(
        ...,
        description="A descriptive query to search for the product, include adjectives, and the product type. will be used to serve relevant products to the user.",
    )


class MultiSearchQuery(BaseModel):
    products: list[SearchQuery]


def extract_table(url: str):
    completion = client.chat.completions.create(
        model="gpt-4-vision-preview",
        max_tokens=1800,
        temperature=0,
        stop=["```"],
        messages=[
            {
                "role": "system",
                "content": f"""
                You are an expert system designed to extract products from images for a ecommerse application
                Please provide the product name and a descriptive query to search for the product.
                Accuratly identify every product in an image and provide a descriptive query to search for the product
                
                You just return a correctly formatted JSON object with the product name and query for each product in the image
                and follows the schema below:

                {MultiSearchQuery.model_json_schema()}
                """,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the products from the image, and describe them in a query in JSON format",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Here is the following search queries for the products in the image\n ```json",
            },
        ],
    )
    return MultiSearchQuery.model_validate_json(completion.choices[0].message.content)


if __name__ == "__main__":
    url = "https://mensfashionpostingcom.files.wordpress.com/2020/03/fbe79-img_5052.jpg?w=768"
    products = extract_table(url)
    print(products.model_dump_json(indent=2))
    """
    {
    "products": [
        {
            "product_name": "Olive Green Shirt",
            "query": "Olive green casual long sleeve button-down shirt"
        },
        {
            "product_name": "Black Jeans",
            "query": "Slim fit black jeans for men"
        },
        {
            "product_name": "Sunglasses",
            "query": "Classic brown aviator sunglasses"
        },
        {
            "product_name": "Leather Strap Watch",
            "query": "Minimalist men's watch with black leather strap"
        },
        {
            "product_name": "Beige Sneakers",
            "query": "Men's beige lace-up fashion sneakers with white soles"
        }
    ]}
    """
