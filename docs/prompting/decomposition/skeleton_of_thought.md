---
title: "Generate in Parallel"
description: "Skelelton-of-Thought is a technique which prompts an LLM to generate a skeleton outline of the response, then completes each point in the skeleton in parallel."
---

How do we decrease the latency of an LLM pipeline?

Skelelton-of-Thought is a technique which prompts an LLM to generate a skeleton outline of the response, then completes each point in the skeleton in parallel. The parallelism can be achieved by parallel API calls or batched decoding.

Below is an example of an implementation using parallel API calls with `instructor`:

```python
import instructor
from pydantic import BaseModel
import asyncio
client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


class Point(BaseModel):
    index: int
    description: str


class Skeleton(BaseModel):
    points: list[Point]


class Response(BaseModel):
    response: str


async def get_skeleton(question):
    return await client.create(
        model="gpt-4o",
        response_model=Skeleton,
        messages=[
            {
                "role": "user",
                "content": f"""
                You’re an organizer responsible for only giving the skeleton (not the full content) for answering the question.
                Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question.
                Instead of writing a full sentence, each skeleton point should be very short with only 3∼5 words.
                Generally, the skeleton should have 3∼10 points.

                Now, please provide the skeleton for the following question.

                <question>
                {question}
                </question>

                Skeleton:
                """,
            }
        ],
    )


async def expand_point(question, skeleton, point_index):
    return await client.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "user",
                "content": f"""
                You’re responsible for continuing the writing of one and only one point in the overall answer to the following question.

                <question>
                {question}
                </question>

                The skeleton of the answer is:

                <skeleton>
                {skeleton}
                </skeleton>

                Continue and only continue the writing of point {point_index}.
                Write it **very shortly** in 1∼2 sentence and do not continue with other points!
                """,
            }
        ],
    )


async def main():
    query = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."

    # Step 1: Get the skeleton
    skeleton = await get_skeleton(query)

    for point in skeleton.points:
        print(point)
        #> index=1 description='Introduction to Hawaii trip'
        #> index=2 description='Arrival and first impressions'
        #> index=3 description='Traditional Hawaiian cuisine'
        #> index=4 description='Exploring local markets'
        #> index=5 description='Visit to historic sites'
        #> index=6 description='Experience a Hawaiian luau'
        #> index=7 description='Day at the beach'
        #> index=8 description='Hiking adventures'
        #> index=9 description='Scenic viewpoints'
        #> index=10 description='Closing remarks and tips'

    # Step 2: Expand on each point in parallel
    tasks = [expand_point(query, skeleton, point.index) for point in skeleton.points]
    responses = await asyncio.gather(*tasks)

    for response in responses:
        print(response.response)
        """
        Hawaii-a paradise of golden beaches, lush landscapes, and vibrant culture-beckoned us with the promise of adventure and unforgettable experiences. Our journey began the moment we landed on this magical archipelago, ready to explore its unique blend of natural beauty and rich traditions.
        """
        """
        The moment we landed in Hawaii, we were greeted with warm aloha spirit, lush tropical landscapes, and the gentle aroma of hibiscus flowers in the air.
        """
        """
        The traditional Hawaiian cuisine was an exotic delight; from savoring the rich flavors of poke bowls to indulging in the sweet taste of haupia, every bite was a unique cultural experience.
        """
        """
        Exploring local markets was a vibrant and delightful experience, where the air was filled with the scent of exotic fruits, freshly-made poke, and sounds of local musicians. We discovered unique handicrafts and interacted with friendly vendors eager to share their stories and traditions.
        """
        """
        A visit to Pearl Harbor is a poignant reminder of the past, offering a chance to pay respects and learn about the events that shaped history. Walking through the USS Arizona Memorial and exploring the interactive exhibits was both humbling and enlightening.
        """
        """
        Point 6: Experience a Hawaiian luau - Attending a traditional Hawaiian luau was unforgettable, filled with vibrant dances, soulful music, and a feast of mouthwatering dishes cooked in an imu (underground oven). It was a magical evening that immersed us in the heart of Hawaiian culture.
        """
        """
        A day at the beach in Hawaii was pure bliss. The crystal-clear waters and soft sands were the perfect backdrop for both relaxation and adventure, from sunbathing to snorkeling.
        """
        """
        Hiking adventures in Hawaii offer a unique chance to connect with nature, with trails leading to stunning waterfalls and lush rainforests. Don’t miss out on the Na Pali Coast's breathtaking hikes!
        """
        """
        One of the highlights of my trip was visiting the scenic viewpoints such as the Na Pali Coast and Haleakalā National Park, offering breathtaking panoramic views that are perfect for photography aficionados and nature lovers alike.
        """
        """
        As you plan your trip, don't forget to pack plenty of sunscreen and a camera to capture every magical moment. Hawaii offers a unique blend of relaxation and adventure that's sure to leave you with unforgettable memories.
        """


if __name__ == "__main__":
    asyncio.run(main())
```


### References

<sup id="ref-1">1</sup>: [Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation](https://arxiv.org/abs/2307.15337)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
