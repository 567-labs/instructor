import instructor
import openai
from pydantic import BaseModel, Field

client = instructor.from_openai(openai.OpenAI())


class Judgment(BaseModel):
    thought: str = Field(
        description="The step-by-step reasoning process used to analyze the question and text"
    )
    justification: str = Field(
        description="Explanation for the similarity judgment, detailing key factors that led to the conclusion"
    )
    similarity: bool = Field(
        description="Boolean judgment indicating whether the question and text are similar or relevant (True) or not (False)"
    )


prompt = """
You are tasked with comparing a question and a piece of text to determine if they are relevant to each other or similar in some way. Your goal is to analyze the content, context, and potential connections between the two.


To determine if the question and text are relevant or similar, please follow these steps:

1. Carefully read and understand both the question and the text.
2. Identify the main topic, keywords, and concepts in the question.
3. Analyze the text for any mention of these topics, keywords, or concepts.
4. Consider any potential indirect connections or implications that might link the question and text.
5. Evaluate the overall context and purpose of both the question and the text.

As you go through this process, please use a chain of thought approach. Write out your reasoning for each step inside <thought> tags.

After your analysis, provide a boolean judgment on whether the question and text are similar or relevant to each other. Use "true" if they are similar or relevant, and "false" if they are not.

Before giving your final judgment, provide a justification for your decision. Explain the key factors that led to your conclusion.
"""


def judge_relevance(question: str, text: str) -> Judgment:
    return client.chat.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": """
             Here is the question:

             <question>
             {{question}}
             </question>

             Here is the text:
             <text>
             {{text}}
             </text>
                """,
            },
        ],
        response_model=Judgment,
        context={"question": question, "text": text},
    )


if __name__ == "__main__":
    test_pairs = [
        {
            "question": "What are the main causes of climate change?",
            "text": "Global warming is primarily caused by human activities, such as burning fossil fuels, deforestation, and industrial processes. These activities release greenhouse gases into the atmosphere, trapping heat and leading to a rise in global temperatures.",
            "is_similar": True,
        },
        {
            "question": "How does photosynthesis work?",
            "text": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar. It occurs in the chloroplasts of plant cells and is essential for life on Earth.",
            "is_similar": True,
        },
        {
            "question": "What are the benefits of regular exercise?",
            "text": "The Eiffel Tower, located in Paris, France, was completed in 1889. It stands 324 meters tall and was originally built as the entrance arch for the 1889 World's Fair.",
            "is_similar": False,
        },
        {
            "question": "How do vaccines work?",
            "text": "The process of baking bread involves mixing flour, water, yeast, and salt to form a dough. The dough is then kneaded, left to rise, shaped, and finally baked in an oven.",
            "is_similar": False,
        },
    ]

    score = 0
    for pair in test_pairs:
        result = judge_relevance(pair["question"], pair["text"])
        if result.similarity == pair["is_similar"]:
            score += 1

    print(f"Score: {score}/{len(test_pairs)}")
