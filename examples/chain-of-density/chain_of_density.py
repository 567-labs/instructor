from pydantic import BaseModel, Field, field_validator
import instructor
import nltk
from openai import OpenAI
import spacy

client = instructor.from_openai(OpenAI())
nlp = spacy.load("en_core_web_sm")


class InitialSummary(BaseModel):
    """
    This is an initial summary which should be long ( 4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose languages and fillers (Eg. This article discusses) to reach ~80 words.
    """

    summary: str = Field(
        ...,
        description="This is a summary of the article provided which is overly verbose and uses fillers. It should be roughly 80 words in length",
    )


class RewrittenSummary(BaseModel):
    """
    This is a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.

    Guidelines
    - Make every word count : Rewrite the previous summary to improve flow and make space for additional entities
    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
    - The new summary should be highly dense and concise yet self-contained, eg., easily understood without the Article.
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses"
    - Missing entities can appear anywhere in the new summary

    An Entity is a real-world object that's assigned a name - for example, a person, country a product or a book title.
    """

    summary: str = Field(
        ...,
        description="This is a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities. It should have the same length ( ~ 80 words ) as the previous summary and should be easily understood without the Article",
    )
    absent: list[str] = Field(
        ...,
        default_factory=list,
        description="this is a list of Entities found absent from the new summary that were present in the previous summary",
    )
    missing: list[str] = Field(
        default_factory=list,
        description="This is a list of 1-3 informative Entities from the Article that are missing from the new summary which should be included in the next generated summary.",
    )

    @field_validator("summary")
    def min_entity_density(cls, v: str):
        # We want to make sure we have a minimum density of 0.12 whenever we do a rewrite. This ensures that the summary quality is always going up
        tokens = nltk.word_tokenize(v)
        num_tokens = len(tokens)

        # Extract Entities
        doc = nlp(v)
        num_entities = len(doc.ents)

        density = num_entities / num_tokens
        if density < 0.08:
            raise ValueError(
                f"The summary of {v} has too few entities. Please regenerate a new summary with more new entities added to it. Remember that new entities can be added at any point of the summary."
            )

        return v

    @field_validator("summary")
    def min_length(cls, v: str):
        tokens = nltk.word_tokenize(v)
        num_tokens = len(tokens)
        if num_tokens < 60:
            raise ValueError(
                "The current summary is too short. Please make sure that you generate a new summary that is around 80 words long."
            )
        return v

    @field_validator("missing")
    def has_missing_entities(cls, missing_entities: list[str]):
        if len(missing_entities) == 0:
            raise ValueError(
                "You must identify 1-3 informative Entities from the Article which are missing from the previously generated summary to be used in a new summary"
            )
        return missing_entities

    @field_validator("absent")
    def has_no_absent_entities(cls, absent_entities: list[str]):
        absent_entity_string = ",".join(absent_entities)
        if len(absent_entities) > 0:
            print(f"Detected absent entities of {absent_entity_string}")
            raise ValueError(
                f"Do not omit the following Entities {absent_entity_string} from the new summary"
            )
        return absent_entities


def summarize_article(article: str, summary_steps: int = 3):
    summary_chain = []
    # We first generate an initial summary
    summary: InitialSummary = client.chat.completions.create(
        model="gpt-4-0613",
        response_model=InitialSummary,
        messages=[
            {
                "role": "system",
                "content": "Write a summary about the article that is long (4-5 sentences) yet highly non-specific. Use overly, verbose language and fillers(eg.,'this article discusses') to reach ~80 words. ",
            },
            {"role": "user", "content": f"Here is the Article: {article}"},
            {
                "role": "user",
                "content": "The generated summary should be about 80 words.",
            },
        ],
        max_retries=2,
    )
    summary_chain.append(summary.summary)
    for _i in range(summary_steps):
        new_summary: RewrittenSummary = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                Article: {article}
                You are going to generate an increasingly concise,entity-dense summary of the following article.

                Perform the following two tasks
                - Identify 1-3 informative entities from the following article which is missing from the previous summary
                - Write a new denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities

                Guidelines
                - Make every word count: re-write the previous summary to improve flow and make space for additional entities
                - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
                - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
                - Missing entities can appear anywhere in the new summary
                - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
                """,
                },
                {
                    "role": "user",
                    "content": f"Here is the previous summary: {summary_chain[-1]}",
                },
            ],
            max_retries=5,
            max_tokens=1000,
            response_model=RewrittenSummary,
        )
        summary_chain.append(new_summary.summary)

    return summary_chain
