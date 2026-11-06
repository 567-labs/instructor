from openai import OpenAI
from io import StringIO
from typing import Annotated, Any
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
import instructor
import pandas as pd
from langsmith.wrappers import wrap_openai
from langsmith import traceable


client = wrap_openai(OpenAI())
client = instructor.from_openai(
    client, mode=instructor.processing.function_calls.Mode.MD_JSON
)


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
        )
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
    tables: list[Table]


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


@traceable(name="extract-table")
def extract(url: str) -> MultipleTables:
    tables = client.chat.completions.create(
        model="gpt-4-vision-preview",
        max_tokens=4000,
        response_model=MultipleTables,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe this data accurately as a table in markdown format. {example.model_dump_json(indent=2)}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                    {
                        "type": "text",
                        "text": """
                            First take a moment to reason about the best set of headers for the tables. 
                            Write a good h1 for the image above. Then follow up with a short description of the what the data is about.
                            Then for each table you identified, write a h2 tag that is a descriptive title of the table. 
                            Then follow up with a short description of the what the data is about. 
                            Lastly, produce the markdown table for each table you identified.


                            Make sure to escape the markdown table properly, and make sure to include the caption and the dataframe.
                            including escaping all the newlines and quotes. Only return a markdown table in dataframe, nothing else.
                        """,
                    },
                ],
            }
        ],
    )
    return tables.model_dump()


urls = [
    "https://a.storyblok.com/f/47007/2400x1260/f816b031cb/uk-ireland-in-three-charts_chart_a.png/m/2880x0",
    "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png/m/2880x0",
]


for url in urls:
    tables = extract(url)
    print(tables)
