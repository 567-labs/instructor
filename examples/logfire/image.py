import instructor
from io import StringIO
from typing import Annotated, Any
from collections.abc import Iterable
from pydantic import (
    BeforeValidator,
    InstanceOf,
    WithJsonSchema,
    BaseModel,
)
import pandas as pd
from openai import OpenAI
import logfire

openai_client = OpenAI()
logfire.configure()
logfire.instrument_pydantic(record="all")
logfire.instrument_openai(openai_client)
client = instructor.from_openai(openai_client, mode=instructor.Mode.MD_JSON)


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
    InstanceOf[pd.DataFrame],
    BeforeValidator(md_to_df),
    WithJsonSchema(
        {
            "type": "string",
            "description": "The markdown representation of the table, each one should be tidy, do not try to join tables that should be separate",
        }
    ),
]


class Table(BaseModel):
    caption: str
    dataframe: MarkdownDataFrame


@logfire.instrument("extract-table", extract_args=True)
def extract_table_from_image(url: str) -> Iterable[Table]:
    return client.chat.completions.create(
        model="gpt-4-vision-preview",
        response_model=Iterable[Table],
        max_tokens=1800,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract out a table from the image. Only extract out the total number of skiiers.",
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    )


url = "https://cdn.statcdn.com/Infographic/images/normal/16330.jpeg"
tables = extract_table_from_image(url)
for table in tables:
    print(table.caption, end="\n")
    print(table.dataframe.to_markdown())
