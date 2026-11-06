import json
import logging
import sys
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich import print as rprint

import instructor

load_dotenv(find_dotenv())

IMAGE_FILE = "image-file.txt"  # file with all the images to be processed

# Add logger
logging.basicConfig()
logger = logging.getLogger("app")
logger.setLevel("INFO")


class Competitor(BaseModel):
    name: str
    features: Optional[list[str]]


# Define models
class Industry(BaseModel):
    """
    Represents competitors from a specific industry extracted from an image using AI.
    """

    name: str = Field(description="The name of the industry")
    competitor_list: list[Competitor] = Field(
        description="A list of competitors for this industry"
    )


class Competition(BaseModel):
    """
    Represents competitors extracted from an image using AI.

    This class serves as a structured representation of
    competitors and their qualities.
    """

    industry_list: list[Industry] = Field(
        description="A list of industries and their competitors"
    )


# Define clients
client_image = instructor.from_openai(OpenAI(), mode=instructor.Mode.MD_JSON)


# Define functions
def read_images(image_urls: list[str]) -> Competition:
    """
    Given a list of image URLs, identify the competitors in the images.
    """

    logger.info(f"Identifying competitors in images... {len(image_urls)} images")

    return client_image.chat.completions.create(
        model="gpt-4-vision-preview",
        response_model=Competition,
        max_tokens=2048,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify competitors and generate key features for each competitor.",
                    },
                    *[
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in image_urls
                    ],
                ],
            }
        ],
    )


def process_and_identify_competitors():
    """
    Main function to process the image list file and identify competitors.
    """

    logger.info("Starting app...")

    try:
        with open(IMAGE_FILE) as file:
            logger.info(f"Reading images from file: {IMAGE_FILE}")
            image_list = file.read().splitlines()
            logger.info(f"{len(image_list)} images read from file: {IMAGE_FILE}")
    except Exception as e:
        logger.error(f"Error reading images from file: {IMAGE_FILE}")
        logger.error(e)
        sys.exit(1)

    competitors = read_images(image_list)

    rprint(f"[green]{len(competitors.industry_list)} industries identified:[/green]")
    for industry in competitors.industry_list:
        rprint(f"[green]{industry.name}[/green]")
        rprint(f"[blue]Features: {industry.competitor_list}[/blue]")

    logger.info("Writing results to file...")

    with open("results.json", "w") as f:
        json.dump(
            {
                "competitors": competitors.model_dump(),
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    process_and_identify_competitors()

"""
Example output:
{
    "competitors": {
        "industry_list": [
            {
                "name": "Accommodation and Hospitality",
                "competitor_list": [
                    {
                        "name": "craigslist",
                        "features": [
                            "Transactions Offline",
                            "Inexpensive"
                        ]
                    },
                    {
                        "name": "couchsurfing",
                        "features": [
                            "Transactions Offline",
                            "Inexpensive"
                        ]
                    },
                    {
                        "name": "BedandBreakfast.com",
                        "features": [
                            "Transactions Offline",
                            "Inexpensive"
                        ]
                    },
                    {
                        "name": "airbnb",
                        "features": [
                            "Transactions Online",
                            "Inexpensive"
                        ]
                    },
                    {
                        "name": "HOSTELS.com",
                        "features": [
                            "Transactions Online",
                            "Inexpensive"
                        ]
                    },
                    {
                        "name": "VRBO",
                        "features": [
                            "Transactions Offline",
                            "Costly"
                        ]
                    },
                    {
                        "name": "Rentahome",
                        "features": [
                            "Transactions Online",
                            "Costly"
                        ]
                    },
                        {
                        "name": "Orbitz",
                        "features": [
                            "Transactions Online",
                            "Costly"
                        ]
                    },
                    {
                        "name": "Hotels.com",
                        "features": [
                            "Transactions Online",
                            "Costly"
                        ]
                    }
                ]
            },
            {
                "name": "E-commerce Wine Retailers",
                "competitor_list": [
                    {
                        "name": "winesimple",
                        "features": [
                            "Ecommerce Retailers",
                            "True Personalized Selections",
                            "Brand Name Wine",
                            "No Inventory Cost",
                            "Target Mass Market"
                        ]
                    },
                    {
                        "name": "nakedwines.com",
                        "features": [
                            "Ecommerce Retailers",
                            "Target Mass Market"
                        ]
                    },
                    {
                        "name": "Club W",
                        "features": [
                            "Ecommerce Retailers",
                            "Brand Name Wine",
                            "Target Mass Market"
                        ]
                    },
                    {
                        "name": "Tasting Room",
                        "features": [
                            "Ecommerce Retailers",
                            "True Personalized Selections",
                            "Brand Name Wine"
                        ]
                    },
                    {
                        "name": "hellovino",
                        "features": [
                            "Ecommerce Retailers",
                            "True Personalized Selections",
                            "No Inventory Cost",
                            "Target Mass Market"
                        ]
                    }
                ]
            }
        ]
    }
}
"""
