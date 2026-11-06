import instructor
from openai import OpenAI
from pydantic import BaseModel
import base64

client = instructor.from_openai(OpenAI(), mode=instructor.Mode.MD_JSON)


class Circle(BaseModel):
    x: int
    y: int
    color: str


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def draw_circle(image_size, num_circles, path):
    from PIL import Image, ImageDraw
    import random

    image = Image.new("RGB", image_size, "white")

    draw = ImageDraw.Draw(image)
    for _ in range(num_circles):
        # Randomize the circle properties
        radius = 100  # random.randint(10, min(image_size)//5)  # Radius between 10 and 1/5th of the smallest dimension
        x = random.randint(radius, image_size[0] - radius)
        y = random.randint(radius, image_size[1] - radius)
        color = ["red", "black", "blue", "green"][random.randint(0, 3)]

        circle_position = (x - radius, y - radius, x + radius, y + radius)
        print(f"Generating circle at {x, y} with color {color}")
        draw.ellipse(circle_position, fill=color, outline="black")

    image.save(path)


img_path = "circle.jpg"
draw_circle((1024, 1024), 1, img_path)
base64_image = encode_image(img_path)

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    max_tokens=1800,
    response_model=Circle,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "find the circle"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)

print(
    f"Found circle with center at x: {response.x}, y: {response.y} and color: {response.color}"
)
