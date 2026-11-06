import json
import logging
import os
import sys
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich import print as rprint

import instructor

load_dotenv(find_dotenv())

# Add logger
logging.basicConfig()
logger = logging.getLogger("app")
logger.setLevel("INFO")


# Define models
class Product(BaseModel):
    """
    Represents a product extracted from an image using AI.

    The product attributes are dynamically determined based on the content
    of the image and the AI's interpretation. This class serves as a structured
    representation of the identified product characteristics.
    """

    name: str = Field(
        description="A generic name for the product.", example="Headphones"
    )
    key_features: Optional[list[str]] = Field(
        description="A list of key features of the product that stand out.",
        example=["Wireless", "Noise Cancellation"],
        default=None,
    )

    description: Optional[str] = Field(
        description="A description of the product.",
        example="Wireless headphones with noise cancellation.",
        default=None,
    )

    def generate_prompt(self):
        prompt = f"Product: {self.name}\n"
        if self.description:
            prompt += f"Description: {self.description}\n"
        if self.key_features:
            prompt += f"Key Features: {', '.join(self.key_features)}\n"
        return prompt


class IdentifiedProduct(BaseModel):
    """
    Represents a list of products identified in the images.
    """

    products: Optional[list[Product]] = Field(
        description="A list of products identified by the AI.",
        example=[
            Product(
                name="Headphones",
                description="Wireless headphones with noise cancellation.",
                key_features=["Wireless", "Noise Cancellation"],
            )
        ],
        default=None,
    )

    error: bool = Field(default=False)
    message: Optional[str] = Field(default=None)

    def __bool__(self):
        return self.products is not None and len(self.products) > 0


class AdCopy(BaseModel):
    """
    Represents a generated ad copy.
    """

    headline: str = Field(
        description="A short, catchy, and memorable headline for the given product. The headline should invoke curiosity and interest in the product.",
        example="Wireless Headphones",
    )
    ad_copy: str = Field(
        description="A long-form advertisement copy for the given product. This will be used in campaigns to promote the product with a persuasive message and a call-to-action with the objective of driving sales.",
        example="""
        "Experience the ultimate sound quality with our wireless headphones, featuring high-definition audio, noise-cancellation, and a comfortable, ergonomic design for all-day listening."
        """,
    )
    name: str = Field(
        description="The name of the product being advertised.",
        example="Headphones",
    )


# Define clients
client_image = instructor.from_openai(
    OpenAI(api_key=os.getenv("OPENAI_API_KEY")), mode=instructor.Mode.MD_JSON
)
client_copy = instructor.from_openai(
    OpenAI(api_key=os.getenv("OPENAI_API_KEY")), mode=instructor.Mode.TOOLS
)


# Define functions
def read_images(image_urls: list[str]) -> IdentifiedProduct:
    """
    Given a list of image URLs, identify the products in the images.
    """

    logger.info(f"Identifying products in images... {len(image_urls)} images")

    return client_image.chat.completions.create(
        model="gpt-4-vision-preview",
        response_model=IdentifiedProduct,
        max_tokens=1024,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify products using the given images and generate key features for each product.",
                    },
                    *[
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in image_urls
                    ],
                ],
            }
        ],
    )


def generate_ad_copy(product: Product) -> AdCopy:
    """
    Given a product, generate an ad copy for the product.
    """

    logger.info(f"Generating ad copy for product: {product.name}")

    return client_copy.chat.completions.create(
        model="gpt-4-1106-preview",
        response_model=AdCopy,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "You are an expert marketing assistant for all products. Your task is to generate an advertisement copy for a product using the name, description, and key features.",
            },
            {"role": "user", "content": product.generate_prompt()},
        ],
    )


def run(images: list[str]) -> tuple[list[Product], list[AdCopy]]:
    """
    Given a list of images, identify the products in the images and generate ad copy for each product.
    """

    identified_products: IdentifiedProduct = read_images(images)
    ad_copies = []

    if identified_products.error:
        rprint(f"[red]Error: {identified_products.message}[/red]")
        return []

    if not identified_products:
        rprint("[yellow]No products identified.[/yellow]")
        return []

    for product in identified_products.products:
        ad_copy: AdCopy = generate_ad_copy(product)
        ad_copies.append(ad_copy)

    return identified_products.products, ad_copies


if __name__ == "__main__":
    # Run logger
    logger.info("Starting app...")

    if len(sys.argv) != 2:
        print("Usage: python app.py <path_to_image_list_file>")
        sys.exit(1)

    image_file = sys.argv[1]
    with open(image_file) as file:
        logger.info(f"Reading images from file: {image_file}")
        try:
            image_list = file.read().splitlines()
            logger.info(f"{len(image_list)} images read from file: {image_file}")
        except Exception as e:
            logger.error(f"Error reading images from file: {image_file}")
            logger.error(e)
            sys.exit(1)

    products, ad_copies = run(image_list)

    rprint(f"[green]{len(products)} products identified:[/green]")
    for product, ad_copy in zip(products, ad_copies):
        rprint(f"[green]{product}[/green]")
        rprint(f"[blue]Ad Copy: {ad_copy.ad_copy}[/blue]")

    logger.info("Writing results to file...")

    with open("results.json", "w") as f:
        json.dump(
            {
                "products": [prod.model_dump() for prod in products],
                "ad_copies": [ad.model_dump() for ad in ad_copies],
            },
            f,
            indent=4,
        )

""" 
Example output:
{
    "products": [
        {
            "name": "Ice Skates",
            "key_features": [
                "Lace-up closure",
                "Durable blade",
                "Ankle support"
            ],
            "description": "A pair of ice skates with lace-up closure for secure fit, durable blade for ice skating, and reinforced ankle support."
        },
        {
            "name": "Hiking Boots",
            "key_features": [
                "High-top design",
                "Rugged outsole",
                "Water-resistant"
            ],
            "description": "Sturdy hiking boots featuring a high-top design for ankle support, rugged outsole for grip on uneven terrain, and water-resistant construction."
        },
        {
            "name": "Winter Boots",
            "key_features": [
                "Insulated lining",
                "Waterproof lower",
                "Slip-resistant sole"
            ],
            "description": "Warm winter boots with insulated lining for cold weather, waterproof lower section to keep feet dry, and a slip-resistant sole for stability."
        }
    ],
    "ad_copies": [
        {
            "headline": "Glide with Confidence - Discover the Perfect Ice Skates!",
            "ad_copy": "Step onto the ice with poise and precision with our premium Ice Skates. Designed for both beginners and seasoned skaters, these skates offer a perfect blend of comfort and performance. The lace-up closure ensures a snug fit that keeps you stable as you carve through the ice. With a durable blade that withstands the test of time, you can focus on perfecting your moves rather than worrying about your equipment. The reinforced ankle support provides the necessary protection and aids in preventing injuries, allowing you to skate with peace of mind. Whether you're practicing your spins, jumps, or simply enjoying a leisurely glide across the rink, our Ice Skates are the ideal companion for your ice adventures. Lace up and get ready to experience the thrill of ice skating like never before!",
            "name": "Ice Skates"
        },
        {
            "headline": "Conquer Every Trail with Confidence!",
            "ad_copy": "Embark on your next adventure with our top-of-the-line Hiking Boots! Designed for the trail-blazing spirits, these boots boast a high-top design that provides unparalleled ankle support to keep you steady on any path. The rugged outsole ensures a firm grip on the most uneven terrains, while the water-resistant construction keeps your feet dry as you traverse through streams and muddy trails. Whether you're a seasoned hiker or just starting out, our Hiking Boots are the perfect companion for your outdoor escapades. Lace up and step into the wild with confidence - your journey awaits!",
            "name": "Hiking Boots"
        },
        {
            "headline": "Conquer the Cold with Comfort!",
            "ad_copy": "Step into the season with confidence in our Winter Boots, the ultimate ally against the chill. Designed for those who don't let the cold dictate their moves, these boots feature an insulated lining that wraps your feet in a warm embrace, ensuring that the biting cold is a worry of the past. But warmth isn't their only virtue. With a waterproof lower section, your feet will remain dry and cozy, come rain, snow, or slush. And let's not forget the slip-resistant sole that stands between you and the treacherous ice, offering stability and peace of mind with every step you take. Whether you're braving a blizzard or just nipping out for a coffee, our Winter Boots are your trusty companions, keeping you warm, dry, and upright. Don't let winter slow you down. Lace up and embrace the elements!",
            "name": "Winter Boots"
        }
    ]
}
"""
