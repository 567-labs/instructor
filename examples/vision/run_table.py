from io import StringIO
from typing import Annotated, Any
from openai import OpenAI
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
import pandas as pd
import instructor


client = instructor.from_openai(OpenAI(), mode=instructor.Mode.MD_JSON)


def to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown()


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
    PlainSerializer(to_markdown),
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


def extract_table(url: str):
    return client.chat.completions.create_iterable(
        model="gpt-4-vision-preview",
        response_model=Table,
        max_tokens=1800,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract the table from the image, and describe it. 
                        Each table should be tidy, do not try to join tables that 
                        should be separately described.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                ],
            }
        ],
    )


if __name__ == "__main__":
    url = "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png"
    tables = extract_table(url)
    for tbl in tables:
        print(tbl.caption, end="\n")
        print(tbl.dataframe)
    """
    Top 10 grossing apps in October 2023 (Ireland) for Android platforms, listing the rank, app name, and category.

                App Name                    Category         
    Rank                                                    
    1                          Google One       Productivity
    2                             Disney+      Entertainment
    3       TikTok - Videos, Music & LIVE      Entertainment
    4                    Candy Crush Saga              Games
    5      Tinder: Dating, Chat & Friends  Social networking
    6                         Coin Master              Games
    7                              Roblox              Games
    8      Bumble - Dating & Make Friends             Dating
    9                         Royal Match              Games
    10        Spotify: Music and Podcasts      Music & Audio

    Top 10 grossing apps in October 2023 (Ireland) for iOS platforms, listing the rank, app name, and category.

                App Name                    Category         
    Rank                                                    
    1      Tinder: Dating, Chat & Friends  Social networking
    2                             Disney+      Entertainment
    3      YouTube: Watch, Listen, Stream      Entertainment
    4        Audible: Audio Entertainment      Entertainment
    5                    Candy Crush Saga              Games
    6       TikTok - Videos, Music & LIVE      Entertainment
    7      Bumble - Dating & Make Friends             Dating
    8                              Roblox              Games
    9         LinkedIn: Job Search & News           Business
    10        Duolingo - Language Lessons          Education
    """
