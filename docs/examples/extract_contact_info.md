---
title: Contact Information Extraction - Lead Generation Automation
description: Automate customer lead extraction from text using Instructor. Extract names, phone numbers, and contact details with automatic validation.
---

# Customer Information Extraction

In this guide, we'll walk through how to extract customer lead information using OpenAI's API and Pydantic. This use case is essential for seamlessly automating the process of extracting specific information from a context.

## Motivation

You could potentially integrate this into a chatbot to extract relevant user information from user messages. With the use of machine learning driven validation it would reduce the need for a human to verify the information.

## Defining the Structure

We'll model a customer lead as a Lead object, including attributes for the name and phone number. We'll use a Pydantic PhoneNumber type to validate the phone numbers entered and provide a Field to give the model more information on correctly populating the object.

## Extracting Lead Information

To extract lead information, we create the `parse_lead_from_message` function which integrates Instructor. It calls OpenAI's API, processes the text, and returns the extracted lead information as a Lead object.

## Evaluating Lead Extraction

To showcase the `parse_lead_from_message` function we can provide sample user messages that may be obtained from a dialogue with a chatbot assistant. Also take note of the response model being set as `Iterable[Lead]` this allows for multiple leads being extracted from the same message.

```python
import instructor
from pydantic import BaseModel, Field
from pydantic_extra_types.phone_numbers import PhoneNumber
from typing import Iterable


class Lead(BaseModel):
    name: str
    phone_number: PhoneNumber = Field(
        description="Needs to be a phone number with a country code. If none, assume +1"
    )

    # Can define some function here to send Lead information to a database using an API


client = instructor.from_provider("openai/gpt-5-nano")


def parse_lead_from_message(user_message: str):
    return client.create(
        model="gpt-5.4-mini",
        response_model=Iterable[Lead],
        messages=[
            {
                "role": "system",
                "content": "You are a data extraction system that extracts a user's name and phone number from a message.",
            },
            {
                "role": "user",
                "content": f"Extract the user's lead information from this user's message: {user_message}",
            },
        ],
    )


if __name__ == "__main__":
    lead = parse_lead_from_message(
        "Yes, that would be great if someone can reach out my name is Patrick King 9175554587"
    )
    assert all(isinstance(item, Lead) for item in lead)
    for item in lead:
        print(item.model_dump_json(indent=2))
        """
        {
          "name": "Patrick King",
          "phone_number": "tel:+1-917-555-4587"
        }
        """

    # Invalid phone number example:
    try:
        lead2 = parse_lead_from_message(
            "Yes, that would be great if someone can reach out my name is Patrick King 9172234"
        )
        assert all(isinstance(item, Lead) for item in lead2)
        for item in lead2:
            print(item.model_dump_json(indent=2))
            """
            {
              "name": "Patrick King",
              "phone_number": "tel:+1-917-223-4999"
            }
            """

    except Exception as e:
        print("ERROR:", e)
        """
        ERROR:
        1 validation error for IterableLead
        tasks.0.phone_number
          value is not a valid phone number [type=value_error, input_value='+19172234', input_type=str]
        """
```

In this example, the `parse_lead_from_message` function successfully extracts lead information from a user message, demonstrating how automation can enhance the efficiency of collecting accurate customer details. It also shows how the function successfully catches that the phone number is invalid so functionality can be implemented for the user to get prompted again to give a correct phone number.
