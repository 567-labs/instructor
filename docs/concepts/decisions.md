---
title: Decisions with Instructor
description: A proposed interface for typed choices, probabilities, and scores.
---

# Decisions with Instructor

!!! warning "Design mockup — not implemented"

    This page proposes an Instructor API. The imports and examples do not work
    in the current release, and example results are illustrative.

Decision models answer questions about a shared state. In the proposed
Instructor decisions mode, a Pydantic model can combine:

- **Choice**: select one option, represented by a Literal, enum, or inline union.
- **Noul**: return the probability that a statement is true.
- **Score**: rate something against an ordered set of descriptive levels.

## Content moderation: a complete example

This model classifies a post, estimates whether it violates policy, and scores
its severity on a 0–10 scale. The `Category` enum owns the criteria for each
option; the inline `target` union puts criteria directly on the field.

```python
from typing import Annotated, Literal

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

    target: (
        Annotated[
            Literal["individual"],
            Choice(description="Directed at an individual"),
        ]
        | Annotated[
            Literal["group"],
            Choice(description="Directed at a group of people"),
        ]
        | Annotated[
            Literal["none"],
            Choice(description="No person or group is targeted"),
        ]
    ) = Field(description="Who is the post directed at?")


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

# Proposed access to the provider's response.
print(raw["answers"]["category"]["probabilities"])
print(raw["answers"]["category"]["confidence"])
print(raw["answers"]["severity"]["score"])  # e.g. 1.4 on Jev's 0–2 scale
print(raw["answers"]["severity"]["probabilities"])
print(raw["answers"]["severity"]["legend"])
```

All five questions are evaluated against the same `context`. Instructor would
return a validated `ModerationDecision` and preserve the provider response as
`raw`. Raw response access above is proposed; its exact transport
representation is an implementation decision. The review threshold is only an
example: choose thresholds using labeled data and the cost of mistakes.

To use Jev directly through TypeSafe, the corresponding proposed client is:

```python
# Set TYPESAFE_API_KEY in your environment.
client = instructor.from_provider(
    "typesafe/jev-latest", mode=instructor.Mode.DECISIONS
)
```

## Choice: literals, enums, and inline unions

Choose the form that fits how much information and reuse you need. Each form
produces one Choice question and returns one of its declared values.

### Literal

Use a Literal for a short list of options:

```python
from typing import Literal
from pydantic import BaseModel, Field


class ModerationDecision(BaseModel):
    format: Literal["text", "image_caption", "other"] = Field(
        description="What kind of content is being reviewed?"
    )
```

The result is a string narrowed to the Literal values. The field description
supplies the question.

### Standard enum

Use a regular string-valued enum for a reusable type:

```python
from enum import Enum
from pydantic import BaseModel, Field


class Category(str, Enum):
    HARASSMENT = "harassment"
    HATE = "hate"
    NONE = "none"


class ModerationDecision(BaseModel):
    category: Category = Field(description="Which policy category applies?")
```

The result is a `Category` member. The enum values become the options.

### Choices with per-option criteria

Use `Choices` and `Choice` when options need their own descriptions or
examples, as in the complete example. The enum docstring supplies the question,
while member names supply the option values: `HARASSMENT` becomes
`"harassment"`. Set `value` when you need a different value:

```python
class Category(Choices):
    """Which policy category applies?"""

    HATE = Choice(value="hate_speech", description="Attacks on protected groups")
    NONE = Choice(description="No listed category applies")
```

### Inline choices

For one-off option descriptions, annotate individual literals and join them
with `|`, as the `target` field does in the complete example. The result remains
a literal string. On Python 3.9, use the equivalent `Union[...]` syntax:

```python
from typing import Annotated, Literal, Union
from instructor.decisions import Choice

Target = Union[
    Annotated[Literal["individual"], Choice(description="A person")],
    Annotated[Literal["group"], Choice(description="A group")],
    Annotated[Literal["none"], Choice(description="No target")],
]
```

For Choice instructions, the proposed precedence is an explicit
`Question(instructions=...)`, then `Field(description=...)`, then a `Choices`
docstring. If none is supplied, fail before the request. The field name is an
identifier, not a question the model sees.

## Noul: a yes/no probability

`Noul` annotates a `float` with a yes/no question. `when_true` and `when_false`
describe the two outcomes; examples pair an input with a boolean. Both meanings
and examples are optional. The model returns the probability of `true`, not a
boolean or a separate confidence value. Choose any threshold in application
code.

For direct control over the native shape, the proposed
`Noul(criteria={"true": ..., "false": ...})` is also accepted. Use either
`criteria` or `when_true`/`when_false`, not both.

## Score: levels and output scale

A Score's levels are ordered descriptions. Jev numbers them from zero and
returns a probability-weighted position across them. Without `scale`, a
three-level Score runs from `0` to `2`.

In the complete example, `scale=(0, 10)` maps the three levels to **0, 5, and
10**. If the level probabilities are `0.1`, `0.4`, and `0.5`, the native score
is `1.4` and the validated model receives `7.0`. Intermediate values do not
create additional described levels. The raw response keeps the native score,
level probabilities, and legend; different distributions can yield the same
score.

For `scale=(lower, upper)`, Instructor would compute:

```python
scaled_score = lower + native_score / (number_of_levels - 1) * (upper - lower)
```

Bounds must be finite with `lower < upper`. Scaling happens before Pydantic
validation and is not sent to Jev. TypeSafe recommends at least two Score levels
and accepts up to ten: it cannot define separate descriptions for every integer
from 0 through 10. Choose levels you can describe distinctly. See
[TypeSafe's Score reference](https://docs.typesafe.ai/primitives/score).

A `Level` carries a description and examples. Score also accepts plain strings,
structured descriptions, and `(input, level_index)` examples:

```python
severity = Score(
    instructions="How severe is the post?",
    scale=(0, 10),
    levels=[
        "No abuse",
        Level("Insult or harassment", examples=["You're worthless."]),
        Level(
            description={"meaning": "Explicit threat", "signals": ["violence"]},
            examples=["I'm going to hurt you."],
        ),
    ],
    examples=[("I disagree.", 0), ("I'm going to hurt you.", 2)],
)
```

Example indexes always refer to the zero-based levels, even with `scale`.
Level examples and indexed examples can be combined; indexed examples are
appended to the matching level in order, including duplicates.

## Context and structured criteria

`context` is always a dictionary. In decisions mode, the entire dictionary is
sent to the provider as state and remains available to Pydantic validators. Put
strings, lists, or nested objects under descriptive keys, such as
`context={"messages": [...]}`. Do not put validator-only data there if it should
not be sent to the provider. Question instructions can use Instructor's Jinja
templating; context values are state, not templates themselves.

Instructions can be strings, objects, or arrays. The same is true of Choice
option descriptions, Score levels, and Noul true/false meanings. Use `Question`
when a Choice needs structured instructions, and the other helpers for its
criteria:

```python
from typing import Annotated
from instructor.decisions import Choice, Choices, Level, Noul, Question, Score
from pydantic import BaseModel


class Category(Choices):
    """Which policy applies?"""

    HARASSMENT = Choice(
        description={"meaning": "Abuse aimed at a person", "signals": ["insult"]},
        examples=["You're worthless."],
    )
    NONE = Choice(description=None)


class ModerationDecision(BaseModel):
    category: Annotated[
        Category,
        Question(instructions={
            "question": "Which policy applies to the post?",
            "consider": ["post", "policy"],
        }),
    ]
    violation: Annotated[
        float,
        Noul(
            instructions="Does the post violate the policy?",
            when_true={"meaning": "An attack", "signals": ["insult"]},
            when_false=["Disagreement", "Criticism without an attack"],
        ),
    ]
    severity: Annotated[
        float,
        Score(
            instructions="How severe is the post?",
            levels=["No abuse", Level(description={"meaning": "A direct attack"})],
        ),
    ]
```

`None` means a Choice option has no extra description. Use JSON-compatible
values for structured instructions and criteria. See
[TypeSafe's structured-question reference](https://docs.typesafe.ai/primitives/advanced).

### How examples are sent

Examples are an Instructor convenience, not a native Jev question field.
Instructor would convert each example input with `str(input)` and place it in
structured criteria alongside its meaning. Choice examples attach to their
option; Noul examples group under `true` or `false`; Score examples attach to
their level. Structured descriptions remain structured: only example inputs
are stringified, using Python's representation rather than JSON serialization.

For example, the complete model's `policy_violation` would compile to:

```json
{
  "type": "noul",
  "instructions": "Does the post violate the supplied content policy?",
  "criteria": {
    "true": {
      "meaning": "The post attacks or harasses a person or group.",
      "examples": ["You're worthless and nobody wants you here."]
    },
    "false": {
      "meaning": "The post expresses disagreement without an attack.",
      "examples": ["I disagree with your argument."]
    }
  }
}
```

The proposed `{ "meaning": ..., "examples": [...] }` shape also goes inside
Choice options and Score levels. Without examples, a description can pass
through unchanged. This serialization is an Instructor proposal, not a format
required by Jev.

## Types and validation

`response_model=ModerationDecision` should let type checkers infer the returned
model, its enum or Literal fields, and the `float` fields under `Annotated`.
`Question`, `Choice`, `Choices`, `Noul`, `Level`, and `Score` are proposed
Instructor helpers, not Jev SDK types.

Unsupported field types, duplicate option values, inconsistent metadata, and
examples outside the declared options or levels should fail before a request.
Jev allows up to 255 Choice options. Invalid or missing answers should fail
validation, not become plausible defaults. Jev does not generate arbitrary
free-text explanations. Implementation and behavior across sync and async
clients still need verification.
