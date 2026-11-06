from pydantic import BaseModel

import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())


class Data(BaseModel):
    index: int
    data_type: str
    pii_value: str


class PIIDataExtraction(BaseModel):
    """
    Extracted PII data from a document, all data_types should try to have consistent property names
    """

    private_data: list[Data]

    def scrub_data(self, content):
        """
        Iterates over the private data and replaces the value with a placeholder in the form of
        <{data_type}_{i}>
        """

        for i, data in enumerate(self.private_data):
            content = content.replace(data.pii_value, f"<{data.data_type}_{i}>")

        return content


EXAMPLE_DOCUMENT = """
# Fake Document with PII for Testing PII Scrubbing Model

## Personal Story

John Doe was born on 01/02/1980. His social security number is 123-45-6789. He has been using the email address john.doe@email.com for years, and he can always be reached at 555-123-4567.

## Residence

John currently resides at 123 Main St, Springfield, IL, 62704. He's been living there for about 5 years now.

## Career

At the moment, John is employed at Company A. He started his role as a Software Engineer in January 2015 and has been with the company since then.
"""

# Define the PII Scrubbing Model
pii_data: PIIDataExtraction = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=PIIDataExtraction,
    messages=[
        {
            "role": "system",
            "content": "You are a world class PII scrubbing model, Extract the PII data from the following document",
        },
        {
            "role": "user",
            "content": EXAMPLE_DOCUMENT,
        },
    ],
)  # type: ignore


print("Extracted PII Data:")
print(pii_data.model_dump_json(indent=2))
"""
{
  "private_data": [
    {
      "index": 0,
      "data_type": "date",
      "pii_value": "01/02/1980"
    },
    {
      "index": 1,
      "data_type": "ssn",
      "pii_value": "123-45-6789"
    },
    {
      "index": 2,
      "data_type": "email",
      "pii_value": "john.doe@email.com"
    },
    {
      "index": 3,
      "data_type": "phone",
      "pii_value": "555-123-4567"
    },
    {
      "index": 4,
      "data_type": "address",
      "pii_value": "123 Main St, Springfield, IL, 62704"
    }
  ]
}
"""

# Scrub the PII Data from the document
print("Scrubbed Document:")
print(pii_data.scrub_data(EXAMPLE_DOCUMENT))
"""
# Fake Document with PII for Testing PII Scrubbing Model

## Personal Story

John Doe was born on <date_of_birth_0>. His social security number is <social_security_number_1>. He has been using the email address <email_address_2> for years, and he can always be reached at <phone_number_3>.

## Residence

John currently resides at <address_4>. He's been living there for about 5 years now.

## Career

At the moment, John is employed at <employment_5>. He started his role as a <job_title_6> in <employment_start_date_7> and has been with the company since then.
"""
