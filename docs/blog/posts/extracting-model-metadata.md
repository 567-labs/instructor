---
title: "Extracting Metadata from Images using Structured Extraction"
date: 2024-12-11
description: Structured Extraction makes working with images easy, in this post we'll see how to use it to extract metadata from images
categories:
  - OpenAI
  - Multimodal
authors:
  - ivanleomk
---

Multimodal Language Models like gpt-4o excel at processing multimodal, enabling us to extract rich, structured metadata from images.

This is particularly valuable in areas like fashion where we can use these capabilities to understand user style preferences from images and even videos. In this post, we'll see how to use instructor to map images to a given product taxonomy so we can recommend similar products for users.

<!-- more -->

## Why Image Metadata is useful

Most online e-commerce stores have a taxonomy of products that they sell. This is a way of categorizing products so that users can easily find what they're looking for.

A small example of a taxonomy is shown below. You can think of this as a way of mapping a product to a set of attributes, with some common attributes that are shared across all products.

```yaml
tops:
  t-shirts:
    - crew_neck
    - v_neck
    - graphic_tees
  sweaters:
    - crewneck
    - cardigan
    - pullover
  jackets:
    - bomber_jackets
    - denim_jackets
    - leather_jackets

bottoms:
  pants:
    - chinos
    - dress_pants
    - cargo_pants
  shorts:
    - athletic_shorts
    - cargo_shorts

colors:
  - black
  - navy
  - white
  - beige
  - brown
```

By using this taxonomy, we can ensure that our model is able to extract metadata that is consistent with the products we sell. In this example, we'll analyze style photos from a fitness influencer to understand their fashion preferences and possibly see what products we can recommend from our own catalog to him.

We're using some photos from a fitness influencer called [Jpgeez](https://www.instagram.com/jpgeez/) which you can see below.

<div class="grid" markdown>
![](./img/style_1.png){: style="height:200px"}
![](./img/style_2.png){: style="height:200px"}
![](./img/style_3.png){: style="height:200px"}
![](./img/style_4.png){: style="height:200px"}
![](./img/style_5.png){: style="height:200px"}
![](./img/style_6.png){: style="height:200px"}
</div>

While we're mapping these visual elements over to a taxonomy, this is really applicable to any other use case where you want to extract metadata from images.

## Extracting metadata from images

### Instructor's `Image` class

With instructor, working with `multimodal` data is easy. We can use the `Image` class to load images from a URL or local file. We can see this below in action.

```python
import instructor

# Load images using instructor.Image.from_path
images = []
for image_file in image_files:
    image_path = os.path.join("./images", image_file)
    image = instructor.Image.from_path(image_path)
    images.append(image)
```

We provide a variety of different methods for loading images, including from a URL, local file, and even from a base64 encoded string which you [can read about here](../../concepts/multimodal.md)

### Defining a response model

Since our taxonomy is defined as a yaml file, we can't use literals to define the response model. Instead, we can read in the configuration from a yaml file and then use that in a `model_validator` step to make sure that the metadata we extract is consistent with the taxonomy.

First, we read in the taxonomy from a yaml file and create a set of categories, subcategories, and product types.

```python
import yaml

with open("taxonomy.yml") as file:
    taxonomy = yaml.safe_load(file)

colors = taxonomy["colors"]
categories = set(taxonomy.keys())
categories.remove("colors")

subcategories = set()
product_types = set()
for category in categories:
    for subcategory in taxonomy[category].keys():
        subcategories.add(subcategory)
        for product_type in taxonomy[category][subcategory]:
            product_types.add(product_type)
```

Then we can use these in our `response_model` to make sure that the metadata we extract is consistent with the taxonomy.

```python
class PersonalStyle(BaseModel):
    """
    Ideally you map this to a specific taxonomy
    """

    categories: list[str]
    subcategories: list[str]
    product_types: list[str]
    colors: list[str]

    @model_validator(mode="after")
    def validate_options(self, info: ValidationInfo):
        context = info.context
        colors = context["colors"]
        categories = context["categories"]
        subcategories = context["subcategories"]
        product_types = context["product_types"]

        # Validate colors
        for color in self.colors:
            if color not in colors:
                raise ValueError(
                    f"Color {color} is not in the taxonomy. Valid colors are {colors}"
                )
        for category in self.categories:
            if category not in categories:
                raise ValueError(
                    f"Category {category} is not in the taxonomy. Valid categories are {categories}"
                )

        for subcategory in self.subcategories:
            if subcategory not in subcategories:
                raise ValueError(
                    f"Subcategory {subcategory} is not in the taxonomy. Valid subcategories are {subcategories}"
                )

        for product_type in self.product_types:
            if product_type not in product_types:
                raise ValueError(
                    f"Product type {product_type} is not in the taxonomy. Valid product types are {product_types}"
                )

        return self
```

### Making the API call

Lastly, we can combine these all into a single api call to `gpt-4o` where we pass in all of the images and the response model into the `response_model` parameter.

With our inbuilt support for `jinja` formatting using the `context` keyword that exposes data we can also re-use in our validation, this becomes an incredibly easy step to execute.

```python
import instructor

client = instructor.from_provider("openai/gpt-5-nano")

resp = client.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """
You are a helpful assistant. You are given a list of images and you need to map the person style of the person in the image to a given taxonomy.

Here is the taxonomy that you should use

Colors:
{% for color in colors %}
* {{ color }}
{% endfor %}

Categories:
{% for category in categories %}
* {{ category }}
{% endfor %}

Subcategories:
{% for subcategory in subcategories %}
* {{ subcategory }}
{% endfor %}

Product types:
{% for product_type in product_types %}
* {{ product_type }}
{% endfor %}
""",
        },
        {
            "role": "user",
            "content": [
                "Here are the images of the person, describe the personal style of the person in the image from a first-person perspective( Eg. You are ... )",
                *images,
            ],
        },
    ],
    response_model=PersonalStyle,
    context={
        "colors": colors,
        "categories": list(categories),
        "subcategories": list(subcategories),
        "product_types": list(product_types),
    },
)
```

This then returns the following response.

```python
PersonalStyle(
    categories=['tops', 'bottoms'],
    subcategories=['sweaters', 'jackets', 'pants'],
    product_types=['cardigan', 'crewneck', 'denim_jackets', 'chinos'],
    colors=['brown', 'beige', 'black', 'white', 'navy'],
)
```

## Looking Ahead

The ability to extract structured metadata from images opens up exciting possibilities for personalization in e-commerce. The key is maintaining the bridge between unstructured visual inspiration and structured product data through well-defined taxonomies and robust validation.

`instructor` makes working with multimodal data easy, and we're excited to see what you build with it. Give us a try today with `pip install instructor` and see how easy it is to work with language models using structured extraction.
