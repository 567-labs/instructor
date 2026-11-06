---
title: Extracting DataFrames from Markdown using Pandas
description: Learn how to extract and convert Markdown tables directly into Pandas DataFrames in Python.
---

# Extracting directly to a DataFrame

In this example we'll show you how to extract directly to a `pandas.DataFrame`

```python
from io import StringIO
from typing import Annotated, Any
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
import pandas as pd
import instructor
import instructor


def md_to_df(data: Any) -> Any:
    # Convert markdown to DataFrame
    if isinstance(data, str):
        return (
            pd.read_csv(
                StringIO(data),  # Process data
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .applymap(lambda x: x.strip())
        )
    return data


MarkdownDataFrame = Annotated[
    # Validates final type
    InstanceOf[pd.DataFrame],
    # Converts markdown to DataFrame
    BeforeValidator(md_to_df),
    # Converts DataFrame to markdown on model_dump_json
    PlainSerializer(lambda df: df.to_markdown()),
    # Adds a description to the type
    WithJsonSchema(
        {
            "type": "string",
            "description": """
            The markdown representation of the table,
            each one should be tidy, do not try to join
            tables that should be separate""",
        }
    ),
]

client = instructor.from_provider("openai/gpt-5-nano")


def extract_df(data: str) -> pd.DataFrame:
    return client.create(
        model="gpt-5.4-mini",
        response_model=MarkdownDataFrame,
        messages=[
            {
                "role": "system",
                "content": "You are a data extraction system, table of writing perfectly formatted markdown tables.",
            },
            {
                "role": "user",
                "content": f"Extract the data into a table: {data}",
            },
        ],
    )


class Table(BaseModel):
    title: str
    data: MarkdownDataFrame


def extract_table(data: str) -> Table:
    return client.create(
        model="gpt-5.4-mini",
        response_model=Table,
        messages=[
            {
                "role": "system",
                "content": "You are a data extraction system, table of writing perfectly formatted markdown tables.",
            },
            {
                "role": "user",
                "content": f"Extract the data into a table: {data}",
            },
        ],
    )


if __name__ == "__main__":
    df = extract_df("""Create a table of the last 5 presidents of the United States,
        including their party and the years they served.""")
    assert isinstance(df, pd.DataFrame)
    print(df)
    """
                         Party          Years Served
     President
    Joe Biden                  Democrat  2021 - Present
    Donald Trump             Republican     2017 - 2021
    Barack Obama               Democrat     2009 - 2017
    George W. Bush           Republican     2001 - 2009
    Bill Clinton               Democrat     1993 - 2001
    """

    table = extract_table(
        """Create a table of the last 5 presidents of the United States,
        including their party and the years they served."""
    )
    assert isinstance(table, Table)
    assert isinstance(table.data, pd.DataFrame)
    print(table.title)
    #> Last 5 Presidents of the United States
    print(table.data)
    """
                         Party  Years Served
     President
    Joe Biden        Democratic     2021-2025
    Donald Trump     Republican     2017-2021
    Barack Obama     Democratic     2009-2017
    George W. Bush   Republican     2001-2009
    Bill Clinton     Democratic     1993-2001
    """
```

Notice that you can extract both the raw `MarkdownDataFrame` or a more complex structure like `Table` which includes a title and the data as a DataFrame. You can even request `Iterable[Table]` to get multiple tables in a single response!
