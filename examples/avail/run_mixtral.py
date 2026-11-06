import os
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timedelta

from openai import OpenAI
import instructor

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key=os.environ["ANYSCALE_API_KEY"],
    ),
    mode=instructor.Mode.JSON_SCHEMA,
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
)


class DateRange(BaseModel):
    explain: str = Field(
        ...,
        description="Explain the date range in the context of the text before generating the date range and the repeat pattern.",
    )
    repeats: Literal["daily", "weekly", "monthly", None] = Field(
        default=None,
        description="If the date range repeats, and how often, this way we can generalize the date range to the future., if its special, then we can assume it is a one time event.",
    )
    days_of_week: list[
        Literal[
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            None,
        ]
    ] = Field(
        ...,
        description="If the date range repeats, which days of the week does it repeat on.",
    )
    time_start: datetime = Field(
        description="The start of the first time range in the day."
    )
    time_end: datetime = Field(
        description="The end of the first time range in the day."
    )


class AvailabilityResponse(BaseModel):
    availability: list[DateRange]


def prepare_dates(n=7) -> str:
    # Current date and time
    now = datetime.now()

    acc = ""
    # Loop for the next 7 days
    for i in range(n):
        # Calculate the date for each day
        day = now + timedelta(days=i)
        # Print the day of the week, date, and time
        acc += "\n" + day.strftime("%A, %Y-%m-%d %H:%M:%S")

    return acc.strip()


def parse_availability(text: str):
    return client.chat.completions.create_iterable(
        max_tokens=10000,
        messages=[
            {
                "role": "system",
                "content": "You are a state of the art date range parse designed to correctly extract availabilities.",
            },
            {
                "role": "user",
                "content": text,
            },
            {
                "role": "user",
                "content": f"To help you understand the dates, here are the next 7 days: {prepare_dates()}",
            },
        ],
        response_model=AvailabilityResponse,
        max_retries=3,
    )


if __name__ == "__main__":
    text = """
    #1
    
    12/8-12/24
    9am - 5pm Monday - Saturday
    10am - 5pm Sunday

    #2
    We are open Friday, after Thanksgiving, and then Saturdays and Sundays 9 a.m. till dusk.``
    """
    schedules = parse_availability(text)
    for schedule in schedules:
        print(schedule.model_dump_json(indent=2))
        {
            "availability": [
                {
                    "explain": "For the first date range, the availability is from December 8 to December 24, from 9 am to 5 pm on Mondays through Saturdays",
                    "repeats": "weekly",
                    "days_of_week": [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                    ],
                    "time_start": "2023-12-08T09:00:00",
                    "time_end": "2023-12-08T17:00:00",
                },
                {
                    "explain": "For the same date range, the availability on Sundays is from 10 am to 5 pm",
                    "repeats": "weekly",
                    "days_of_week": ["sunday"],
                    "time_start": "2023-12-10T10:00:00",
                    "time_end": "2023-12-10T17:00:00",
                },
            ]
        }
    {
        "availability": [
            {
                "explain": "The second date range starting from the Friday after Thanksgiving, which is November 24, 2023, and then on Saturdays and Sundays from 9 am until dusk. Assuming 'dusk' means approximately 5 pm, similar to the previous timings.",
                "repeats": "weekly",
                "days_of_week": ["friday", "saturday", "sunday"],
                "time_start": "2023-11-24T09:00:00",
                "time_end": "2023-11-24T17:00:00",
            }
        ]
    }
