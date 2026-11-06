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
from rich.console import Console

console = Console()
client = instructor.from_openai(
    client=OpenAI(),
    mode=instructor.Mode.TOOLS,
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


def extract(url: str):
    return client.chat.completions.create(
        model="gpt-4-turbo",
        max_tokens=4000,
        response_model=Table,
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
                            Analyze the organizational chart image and extract the relevant information to reconstruct the hierarchy.
                            
                            Create a list of People objects, where each person has the following attributes:
                            - id: A unique identifier for the person
                            - name: The person's name
                            - role: The person's role or position in the organization
                            - manager_name: The name of the person who manages this person
                            - manager_role: The role of the person who manages this person
                            
                            Ensure that the relationships between people are accurately captured in the reports and manages attributes.
                            
                            Return the list of People objects as the people attribute of an Organization object.
                        """,
                    },
                ],
            }
        ],
    )


print(
    extract(
        "https://www.mindmanager.com/static/mm/images/features/org-chart/hierarchical-chart.png"
    ).model_dump()["dataframe"]
)
"""
|    id  |  name              |  role                                    |  manager_name     |  manager_role                |
|-------:|:-------------------|:-----------------------------------------|:------------------|:-----------------------------|
|    1   | Adele Morana       | Founder, Chairman & CEO                  |                   |                              |
|    2   | Winston Cole       | COO                                      | Adele Morana      | Founder, Chairman & CEO      |
|    3   | Marcus Kim         | CFO                                      | Adele Morana      | Founder, Chairman & CEO      |
|    4   | Karin Ludovicus    | CPO                                      | Adele Morana      | Founder, Chairman & CEO      |
|    5   | Lea Erastos        | Chief Business Officer                   | Winston Cole      | COO                          |
|    6   | John McKinley      | Chief Accounting Officer                 | Winston Cole      | COO                          |
|    7   | Zahida Mahtab      | VP, Global Affairs & Communication       | Winston Cole      | COO                          |
|    8   | Adelaide Zhu       | VP, Central Services                     | Winston Cole      | COO                          |
|    9   | Gabriel Drummond   | VP, Investor Relations                   | Marcus Kim        | CFO                          |
|    10  | Felicie Vasili     | VP, Finance                              | Marcus Kim        | CFO                          |
|    11  | Ayda Williams      | VP, Global Customer & Business Marketing | Karin Ludovicius  | CPO                          |
|    12  | Nicholas Brambilla | VP, Company Brand                        | Karin Ludovicius  | CPO                          |
|    13  | Sandra Herminius   | VP, Product Marketing                    | Karin Ludovicius  | CPO                          |
"""
