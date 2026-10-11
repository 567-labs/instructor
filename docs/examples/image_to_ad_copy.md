---
title: Automatically Generate Advertising Copy from Product Images Using GPT-4 Vision
description: Learn how to use GPT-4 Vision API to create engaging advertising copy from product images, ideal for e-commerce and marketing teams.
---

# Use Vision API to detect products and generate advertising copy

This post demonstrates how to use GPT-4 Vision API and the Chat API to automatically generate advertising copy from product images. This method can be useful for marketing and advertising teams, as well as for e-commerce platforms.

The full code is available on [GitHub](https://www.github.com/jxnl/instructor/tree/main/examples/vision/image_to_ad_copy.py).

## Building the models

### Product

For the `Product` model, we define a class that represents a product extracted from an image and store the name, key features, and description. The product attributes are dynamically determined based on the content of the image.

Note that it is easy to add [Validators](https://python.useinstructor.com/concepts/reask_validation/) and other Pydantic features to the model to ensure that the data is valid and consistent.

```python
from pydantic import BaseModel, Field
from typing import List, Optional


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
    key_features: Optional[List[str]] = Field(
        description="A list of key features of the product that stand out.",
        default=None,
    )

    description: Optional[str] = Field(
        description="A description of the product.",
        default=None,
    )

    # Can be customized and automatically generated
    def generate_prompt(self):
        prompt = f"Product: {self.name}\n"
        if self.description:
            prompt += f"Description: {self.description}\n"
        if self.key_features:
            prompt += f"Key Features: {', '.join(self.key_features)}\n"
        return prompt
```

### Identified Product

We also define a class that represents a list of products identified in the images. We also add an error flag and message to indicate if there was an error in the processing of the image.

```python
from pydantic import BaseModel, Field
from typing import Optional, List


# <%hide%>
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
    key_features: Optional[List[str]] = Field(
        description="A list of key features of the product that stand out.",
        default=None,
    )

    description: Optional[str] = Field(
        description="A description of the product.",
        default=None,
    )

    # Can be customized and automatically generated
    def generate_prompt(self):
        prompt = f"Product: {self.name}\n"
        if self.description:
            prompt += f"Description: {self.description}\n"
        if self.key_features:
            prompt += f"Key Features: {', '.join(self.key_features)}\n"
        return prompt


# <%hide%>
class IdentifiedProduct(BaseModel):
    """
    Represents a list of products identified in the images.
    """

    products: Optional[List[Product]] = Field(
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
```

### Advertising Copy

Finally, the `AdCopy` models stores the output in a structured format with a headline and the text.

```python
from pydantic import BaseModel, Field


class AdCopy(BaseModel):
    """
    Represents a generated ad copy.
    """

    headline: str = Field(
        description="A short, catchy, and memorable headline for the given product. The headline should invoke curiosity and interest in the product.",
    )
    ad_copy: str = Field(
        description="A long-form advertisement copy for the given product. This will be used in campaigns to promote the product with a persuasive message and a call-to-action with the objective of driving sales.",
    )
    name: str = Field(description="The name of the product being advertised.")
```

## Calling the API

### Product Detection

The `read_images` function uses OpenAI's vision model to process a list of image URLs and identify products in each of them. We utilize the `instructor` library to patch the OpenAI client for this purpose.

```python
# <%hide%>
from pydantic import BaseModel, Field
from typing import Optional, List


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
    key_features: Optional[List[str]] = Field(
        description="A list of key features of the product that stand out.",
        default=None,
    )

    description: Optional[str] = Field(
        description="A description of the product.",
        default=None,
    )

    # Can be customized and automatically generated
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

    products: Optional[List[Product]] = Field(
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


# <%hide%>
def read_images(image_urls: list[str]) -> IdentifiedProduct:
    """
    Given a list of image URLs, identify the products in the images.
    """

    logger.info(f"Identifying products in images... {len(image_urls)} images")

    return client_image.create(
        response_model=IdentifiedProduct,
        max_tokens=1024,  # can be changed
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
```

This gives us a list of products identified in all the images.

### Generate advertising copy

Then, we can use the `generate_ad_copy` function to generate advertising copy for each of the products identified in the images.

Two clients are defined for the two different models. This is because the `gpt-5.4-mini` model is not compatible with the `gpt-5.4-mini` model in terms of their response format.

```python
# <%hide%>
from pydantic import BaseModel, Field
from typing import List, Optional


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
    key_features: Optional[List[str]] = Field(
        description="A list of key features of the product that stand out.",
        default=None,
    )

    description: Optional[str] = Field(
        description="A description of the product.",
        default=None,
    )

    # Can be customized and automatically generated
    def generate_prompt(self):
        prompt = f"Product: {self.name}\n"
        if self.description:
            prompt += f"Description: {self.description}\n"
        if self.key_features:
            prompt += f"Key Features: {', '.join(self.key_features)}\n"
        return prompt


class AdCopy(BaseModel):
    """
    Represents a generated ad copy.
    """

    headline: str = Field(
        description="A short, catchy, and memorable headline for the given product. The headline should invoke curiosity and interest in the product.",
    )
    ad_copy: str = Field(
        description="A long-form advertisement copy for the given product. This will be used in campaigns to promote the product with a persuasive message and a call-to-action with the objective of driving sales.",
    )
    name: str = Field(description="The name of the product being advertised.")


# <%hide%>
def generate_ad_copy(product: Product) -> AdCopy:
    """
    Given a product, generate an ad copy for the product.
    """

    logger.info(f"Generating ad copy for product: {product.name}")

    return client_copy.create(
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
```

### Putting it all together

Finally, we can put it all together in a single function that takes a list of image URLs and generates advertising copy for the products identified in the images. Please refer to the [full code](https://www.github.com/jxnl/instructor/tree/main/examples/vision/image_to_ad_copy.py) for the complete implementation.

## Input file

The input file is currently a list of image URLs, but this trivial to change to any required format.

```plaintext
https://contents.mediadecathlon.com/p1279823/9a1c59ad97a4084a346c014740ae4d3ff860ea70b485ee65f34017ff5e9ae5f7/recreational-ice-skates-fit-50-black.jpg?format=auto
https://contents.mediadecathlon.com/p1279822/a730505231dbd6747c14ee93e8f89e824d3fa2a5b885ec26de8d7feb5626638a/recreational-ice-skates-fit-50-black.jpg?format=auto
https://contents.mediadecathlon.com/p2329893/1ed75517602a5e00245b89ab6a1c6be6d8968a5a227c932b10599f857f3ed4cd/mens-hiking-leather-boots-sh-100-x-warm.jpg?format=auto
https://contents.mediadecathlon.com/p2047870/8712c55568dd9928c83b19c6a4067bf161811a469433dc89244f0ff96a50e3e9/men-s-winter-hiking-boots-sh-100-x-warm-grey.jpg?format=auto
```

??? Note "Expand to see the output"

    ![Recreational ice skates product image for ad copy generation](https://contents.mediadecathlon.com/p1279823/9a1c59ad97a4084a346c014740ae4d3ff860ea70b485ee65f34017ff5e9ae5f7/recreational-ice-skates-fit-50-black.jpg?format=auto)
    ![Men's hiking leather boots product image for ad copy generation](https://contents.mediadecathlon.com/p2329893/1ed75517602a5e00245b89ab6a1c6be6d8968a5a227c932b10599f857f3ed4cd/mens-hiking-leather-boots-sh-100-x-warm.jpg?format=auto)

    ```json
    {
        "products":
        [
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
    ```
