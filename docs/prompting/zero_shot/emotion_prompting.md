---
title: "Emotion Prompting"
description: "Adding phrases with emotional significance to humans can help enhance the performance of a language model."
---

Do language models respond to emotional stimuli?

Adding phrases with emotional significance to humans can help enhance the performance of a language model. This includes phrases such as:

- This is very important to my career.
- Take pride in your work.
- Are you sure?

!!! info
    For more examples of emotional stimuli to use in prompts, look into [EmotionPrompt](https://arxiv.org/abs/2307.11760) -- a set of prompts inspired by well-established human psychological phenomena.

## Implementation
```python hl_lines="34"
import openai
import instructor
from pydantic import BaseModel
from typing import Iterable


class Album(BaseModel):
    name: str
    artist: str
    year: int


client = instructor.from_provider("openai/gpt-5-nano")


def emotion_prompting(query, stimuli):
    return client.create(
        model="gpt-4o",
        response_model=Iterable[Album],
        messages=[
            {
                "role": "user",
                "content": f"""
                {query}
                {stimuli}
                """,
            }
        ],
    )


if __name__ == "__main__":
    query = "Provide me with a list of 3 musical albums from the 2000s."
    stimuli = "This is very important to my career."  # (1)!

    albums = emotion_prompting(query, stimuli)

    for album in albums:
        print(album)
        #> name='Kid A' artist='Radiohead' year=2000
        #> name='The Marshall Mathers LP' artist='Eminem' year=2000
        #> name='The College Dropout' artist='Kanye West' year=2004
```

1.  The phrase `This is very important to my career` is used as emotional stimuli in the prompt.

## References

<sup id="ref-1">1</sup>: [Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)