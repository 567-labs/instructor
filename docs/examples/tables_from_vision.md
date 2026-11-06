---
title: Extracting Tables from Images Using OpenAI GPT-4
description: Learn how to convert images into markdown tables using OpenAI's GPT-4 Vision model for data extraction and analysis.
---

# Extracting Tables from Images with OpenAI's GPT-4 Vision Model

First, we define a custom type, `MarkdownDataFrame`, to handle pandas DataFrames formatted in markdown. This type uses Python's `Annotated` and `InstanceOf` types, along with decorators `BeforeValidator` and `PlainSerializer`, to process and serialize the data.

## Defining the Table Class

The `Table` class is essential for organizing the extracted data. It includes a caption and a dataframe, processed as a markdown table. Since most of the complexity is handled by the `MarkdownDataFrame` type, the `Table` class is straightforward!

This requires additional dependencies `pip install pandas tabulate`.

```python
from io import StringIO
from typing import Annotated, Any, List
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
import instructor
import pandas as pd
from rich.console import Console

console = Console()
client = instructor.from_provider("openai/gpt-4o", mode=instructor.Mode.TOOLS)


def md_to_df(data: Any) -> Any:
    if isinstance(data, str):
        return (
            pd.read_csv(
                StringIO(data),  # Get rid of whitespaces
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .map(lambda x: x.strip())
        )  # type: ignore
    return data


MarkdownDataFrame = Annotated[
    InstanceOf[pd.DataFrame],
    BeforeValidator(md_to_df),
    PlainSerializer(lambda x: x.to_markdown()),
    WithJsonSchema(
        {
            "type": "string",
            "description": """
                The markdown representation of the table,
                each one should be tidy, do not try to join tables
                that should be separate""",
        }
    ),
]


class Table(BaseModel):
    caption: str
    dataframe: MarkdownDataFrame


class MultipleTables(BaseModel):
    tables: List[Table]


example = MultipleTables(
    tables=[
        Table(
            caption="This is a caption",
            dataframe=pd.DataFrame(
                {
                    "Chart A": [10, 40],
                    "Chart B": [20, 50],
                    "Chart C": [30, 60],
                }
            ),
        )
    ]
)


def extract(url: str) -> MultipleTables:
    return client.create(
        model="gpt-5.4-mini",
        max_tokens=4000,
        response_model=MultipleTables,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                    {
                        "type": "text",
                        "text": """
                            First, analyze the image to determine the most appropriate headers for the tables.
                            Generate a descriptive h1 for the overall image, followed by a brief summary of the data it contains.
                            For each identified table, create an informative h2 title and a concise description of its contents.
                            Finally, output the markdown representation of each table.
                            Make sure to escape the markdown table properly, and make sure to include the caption and the dataframe.
                            including escaping all the newlines and quotes. Only return a markdown table in dataframe, nothing else.
                        """,
                    },
                ],
            }
        ],
    )


urls = [
    "https://a.storyblok.com/f/47007/2400x1260/f816b031cb/uk-ireland-in-three-charts_chart_a.png/m/2880x0",
    "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png/m/2880x0",
]

for url in urls:
    for table in extract(url).tables:
        console.print(table.caption, "\n", table.dataframe)
```
