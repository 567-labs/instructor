---
title: Typed decisions
description: Evaluate a shared context with typed choices, probabilities, and scores.
---

# Typed decisions

Decision models evaluate one context against several questions in a single request.
Use `Mode.DECISIONS` with Jev through OpenRouter or TypeSafe. Define the questions
as a Pydantic model and receive a validated instance.

## Content moderation

This example uses a reusable choice enum, an inline choice, a yes/no probability,
and a score mapped to 0–10. Outputs shown in comments are illustrative.

```python
from typing import Annotated, Literal, Union

import instructor
from instructor.decisions import Choice, Choices, Level, Noul, Score
from pydantic import BaseModel, Field


class Category(Choices):
    """Which policy category best describes the post?"""

    HARASSMENT = Choice(
        description="Abusive content directed at a person",
        examples=["You're worthless and nobody wants you here."],
    )
    HATE = Choice(
        description="Content attacking people for a protected characteristic",
        examples=["People of that religion are inferior."],
    )
    NONE = Choice(
        description="Neither category applies",
        examples=["I disagree with your argument."],
    )


class ModerationDecision(BaseModel):
    category: Category

    policy_violation: Annotated[
        float,
        Noul(
            instructions="Does the post violate the supplied content policy?",
            when_true="The post attacks or harasses a person or group.",
            when_false="The post expresses disagreement without an attack.",
            examples=[
                ("You're worthless and nobody wants you here.", True),
                ("I disagree with your argument.", False),
            ],
        ),
    ]

    severity: Annotated[
        float,
        Score(
            instructions="How severe is the content?",
            scale=(0, 10),
            levels=[
                Level("No abuse", examples=["I disagree with you."]),
                Level("Insult or harassment", examples=["You're worthless."]),
                Level("Explicit threat", examples=["I'm going to hurt you."]),
            ],
        ),
    ]

    format: Literal["text", "image_caption", "other"] = Field(
        description="What kind of content is being reviewed?"
    )

    target: Union[
        Annotated[
            Literal["individual"],
            Choice(description="Directed at an individual"),
        ],
        Annotated[
            Literal["group"],
            Choice(description="Directed at a group of people"),
        ],
        Annotated[
            Literal["none"],
            Choice(description="No person or group is targeted"),
        ],
    ] = Field(description="Who is the post directed at?")


# Set OPENROUTER_API_KEY in your environment.
client = instructor.from_provider(
    "openrouter/typesafe/jev-1.13",
    mode=instructor.Mode.DECISIONS,
)

decision, raw = client.create_with_completion(
    response_model=ModerationDecision,
    context={
        "post": {
            "text": "You're worthless and nobody wants you here.",
            "format": "text",
        },
        "policy": ["No harassment", "No attacks on protected groups"],
    },
)

# Illustrative results, not a live model response.
print(decision.category)          # Category.HARASSMENT
print(decision.policy_violation)  # 0.84, probability of a violation
print(decision.severity)          # 7.0, on the requested 0–10 scale
print(decision.format)            # "text"
print(decision.target)            # "individual"

# The application decides what to do with the result.
if decision.policy_violation >= 0.80:
    print("Send to human review")

# Inspect the provider's response.
print(raw["answers"]["category"]["probabilities"])
print(raw["answers"]["category"]["confidence"])
print(raw["answers"]["severity"]["score"])  # e.g. 1.4 on Jev's 0–2 scale
print(raw["answers"]["severity"]["probabilities"])
print(raw["answers"]["severity"]["legend"])
```

`context` is a dictionary sent in full as provider state and passed to Pydantic
validators. Do not include validator-only values that should not be sent. Use
`create()` for the validated model alone, or `create_with_completion()` to also
inspect the unchanged provider response and its probabilities and confidence.

For direct TypeSafe access, set `TYPESAFE_API_KEY` and use:

```python
client = instructor.from_provider(
    "typesafe/jev-latest", mode=instructor.Mode.DECISIONS
)
```

Both providers use the same decision model and `context`. To switch routes,
change the provider/model string and configure that provider's API key; the
question annotations and validated result do not change. Raw response metadata
remains provider-specific. For example, OpenRouter includes a monetary `cost`
in `usage`; do not infer a direct TypeSafe cost from token counts.

For async use, pass `async_client=True`, await `create` or
`create_with_completion`, and call `await client.close()` when done. Synchronous
clients also support `with`; asynchronous clients support `async with`.

An owned HTTP client defaults to a 60-second timeout. When supplying
`http_client`, its timeout is preserved unless you pass `timeout=...` to
`from_provider`. That override applies to decision requests without changing the
supplied HTTP client's configuration. Closing a decision client only closes an
HTTP client it created. Pass `timeout=None` explicitly to disable timeouts.

## Choices

Use a `Literal` for a short list, a string-valued Python `Enum` for a reusable
type, or `Choices` when options need descriptions and examples. `Choices`
returns enum members; names become lowercase values unless `Choice(value=...)`
sets an explicit value. Inline `Annotated[Literal[...], Choice(...)]` branches
return literal values. On Python 3.9, use `Union[...]` instead of `|`.

Every question needs instructions: use `Question(instructions=...)`,
`Field(description=...)`, or a `Choices`/enum docstring, in that order. The
field name is an answer key, not a question. For structured instructions:

```python
from typing import Annotated, Literal
from pydantic import BaseModel
from instructor.decisions import Question

class Route(BaseModel):
    team: Annotated[
        Literal["billing", "support"],
        Question(instructions={"question": "Which team?", "policy": ["Route refunds to billing"]}),
    ]
```

## Probabilities and scores

`Noul` annotates a `float` with a yes/no question. The result is the probability
of true, from 0 to 1. `when_true` and `when_false` clarify the boundary; examples
are `(input, bool)` pairs. For the native shape, use
`criteria={"true": ..., "false": ...}` instead of `when_true` and `when_false`.

`Score` annotates a `float` with 2–10 ordered levels. Jev returns a weighted value
on the levels' zero-based indexes. Three levels span 0–2; `scale=(0, 10)` maps
them to 0, 5, and 10. A result of 7 is an intermediate weighted value, not an
additional defined level. Bounds must be finite and increasing. Score examples
are `(input, level_index)` pairs; a `Level` can also carry examples.

The raw score, probabilities, and legend remain in provider units. Choice and
Score confidence describe the distribution, not the probability of correctness;
choose any moderation thresholds using labeled data.

## Structured criteria and examples

Instructions and criteria accept JSON-compatible strings, objects, or arrays.
Choice descriptions can also be `None`. Instructions can use Instructor's
Jinja templates with `context`; the context itself is sent unchanged.

Examples are an Instructor convenience: inputs are converted with `str()` and
combined with the relevant meaning as `{"meaning": ..., "examples": [...]}`.
Without examples, the meaning passes through unchanged.

Unsupported field types, invalid labels, missing instructions, and invalid scales
fail before the request. Missing, mismatched, or out-of-range answers fail rather
than being replaced with defaults. Decision mode currently does not support
streaming, retries, caching, or completion hooks; HTTP errors propagate to the
caller. See the [TypeSafe API reference](https://docs.typesafe.ai/api) and
[OpenRouter's Jev guide](https://openrouter.ai/blog/tutorials/how-to-use-jev/).

## Testing

The local suites in `tests/v2/decisions/` cover annotations and answer parsing,
client configuration, and common provider contracts over localhost HTTP. They
do not call provider services or require API keys:

```bash
uv run pytest tests/v2/decisions/
```

Live provider contracts are a separate, opt-in suite. They use synthetic ticket
data, make billed API requests, and skip providers whose keys are missing:

```bash
uv run pytest tests/llm/test_decisions/ --run-decisions-live
```

Set `TYPESAFE_API_KEY` and/or `OPENROUTER_API_KEY` for the selected providers.
`TYPESAFE_DECISIONS_MODEL` and `OPENROUTER_DECISIONS_MODEL` optionally override
the live model IDs. Without the explicit flag, live cases are skipped even
when credentials are present. Local contracts do not certify a live provider.
