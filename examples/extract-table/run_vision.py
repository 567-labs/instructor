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


def extract(url: str) -> MultipleTables:
    return client.chat.completions.create(
        model="gpt-4-turbo",
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
"""
Growth in app installations and sessions across different app categories in Q3 2022 compared to Q2 2022 for Ireland and U.K. 
              Install Growth (%)  Session Growth (%) 
 Category                                           
Education                      7                   6
Games                         13                   3
Social                         4                  -3
Utilities                      6                -0.4
Top 10 Grossing Android Apps in Ireland, October 2023 
                              App Name           Category 
 Rank                                                    
1                           Google One       Productivity
2                              Disney+      Entertainment
3        TikTok - Videos, Music & LIVE      Entertainment
4                     Candy Crush Saga              Games
5       Tinder: Dating, Chat & Friends  Social networking
6                          Coin Master              Games
7                               Roblox              Games
8       Bumble - Dating & Make Friends             Dating
9                          Royal Match              Games
10         Spotify: Music and Podcasts      Music & Audio
Top 10 Grossing iOS Apps in Ireland, October 2023 
                              App Name           Category 
 Rank                                                    
1       Tinder: Dating, Chat & Friends  Social networking
2                              Disney+      Entertainment
3       YouTube: Watch, Listen, Stream      Entertainment
4         Audible: Audio Entertainment      Entertainment
5                     Candy Crush Saga              Games
6        TikTok - Videos, Music & LIVE      Entertainment
7       Bumble - Dating & Make Friends             Dating
8                               Roblox              Games
9          LinkedIn: Job Search & News           Business
10         Duolingo - Language Lessons          Education
"""
