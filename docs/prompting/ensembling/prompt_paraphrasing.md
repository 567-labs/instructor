---
description: "Use Large Language Models to perform back translation in order to improve prompt performance"
---

Large Language Models are sensitive to the way that they are prompted. When prompted incorrectly, they might perform much worse despite having the information or capability to respond to the prompt. We can help find semantically similar prompts by performing back translation - where we translate our prompts to another language and back to encourage more diversity in the rephrased prompts.

Prompt paraphrasing <sup><a href="https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00324/96460/How-Can-We-Know-What-Language-Models-Know">1</a></sup>. provides some ways for us to improve on the phrasing of our prompts to do so.

We can implement this using `instructor` as seen below.

```python hl_lines="20-25"
import instructor
from pydantic import BaseModel
import random
client = instructor.from_provider("openai/gpt-5-nano", async_client=True)


class TranslatedPrompt(BaseModel):
    translation: str


async def translate_prompt(prompt: str, from_language: str, to_language: str):
    return await client.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""
                You are an expert translation assistant.
                You are going to be given a prompt and
                asked to translate it from {from_language}
                to {to_language}. Paraphrase and use
                synonyms where possible, especially for
                the examples.
                """,
            },
            {"role": "user", "content": f"Prompt: {prompt}"},
        ],
        response_model=TranslatedPrompt,
    )


async def generate_permutation(prompt: str, language: str) -> str:
    tranlated_prompt = await translate_prompt(prompt, "english", language)
    backtranslated_prompt = await translate_prompt(
        tranlated_prompt.translation, language, "english"
    )
    return backtranslated_prompt.translation


async def generate_prompts(
    prompt: str, languages: list[str], permutations: int
) -> list[str]:
    coros = [
        generate_permutation(prompt, random.choice(languages))
        for _ in range(permutations)
    ]
    return await asyncio.gather(*coros)


if __name__ == "__main__":
    import asyncio

    prompt = """
    You are an expert system that excels at Sentiment
    Analysis of User Reviews.

    Here are a few examples to refer to:

    1. That was a fantastic experience I had! I'm
    definitely recommending this to all my friends
    // Positive
    2. I think it was a passable evening. I don't think
    there was anything remarkable or off-putting for me.
    // Negative
    3. I'm horrified at the state of affairs in this new
    restaurant // Negative

    Sentence: This was a fantastic experience!
    """
    languages = ["french", "spanish", "chinese"]
    permutations = 2

    generated_prompts = asyncio.run(generate_prompts(prompt, languages, permutations))
    for prompt in generated_prompts:
        print(prompt)
        """
        You are an expert system specializing in user review sentiment analysis. Here are a few examples to guide you: 1. It was an exceptional experience! I will definitely recommend it to all my friends // Positive 2. I think it was a mediocre evening. There wasn't anything outstanding or particularly bad for me // Negative 3. I am horrified by the condition of things in this new restaurant // Negative Sentence: It was an amazing experience!
        """
        """
        You are an expert system that excels in User Review Sentiment Analysis.

        Here are some reference examples:

        1. I had an amazing experience! I will definitely recommend it to all my friends.
        // Positive
        2. I think it was an average evening. I don’t believe there was anything remarkable or unpleasant about it for me.
        // Negative
        3. I am horrified by the situation at this new restaurant.
        // Negative

        Sentence: This was a fantastic experience!
        """
        """
        You are an expert system skilled in conducting user
        review sentiment analysis.

        Here are some examples for reference:

        1. That was an awesome experience! I'll definitely
        recommend it to all my friends // Positive
        2. I think it was an okay evening. I don't find
        anything particularly outstanding or unpleasant.
        // Neutral
        3. I am very shocked by the condition of this new
        restaurant // Negative

        Sentence: This was a wonderful experience!
        """
```

### References

<sup id="ref-1">1</sup>: [How Can We Know What Language Models Know? ](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00324/96460/How-Can-We-Know-What-Language-Models-Know)
