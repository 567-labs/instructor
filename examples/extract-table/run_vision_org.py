from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console

import instructor

console = Console()
client = instructor.from_openai(
    client=OpenAI(),
    mode=instructor.Mode.TOOLS,
)


class People(BaseModel):
    id: str
    name: str
    role: str
    reports: list[str] = Field(
        default_factory=list, description="People who report to this person"
    )
    manages: list[str] = Field(
        default_factory=list, description="People who this person manages"
    )


class Organization(BaseModel):
    people: list[People]


def extract(url: str):
    return client.chat.completions.create_partial(
        model="gpt-4-turbo",
        max_tokens=4000,
        response_model=Organization,
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
                            - reports: A list of IDs of people who report directly to this person
                            - manages: A list of IDs of people who this person manages
                            
                            Ensure that the relationships between people are accurately captured in the reports and manages attributes.
                            
                            Return the list of People objects as the people attribute of an Organization object.
                        """,
                    },
                ],
            }
        ],
    )


console.print(
    extract(
        "https://www.mindmanager.com/static/mm/images/features/org-chart/hierarchical-chart.png"
    )
)
"""
Organization(
    people=[
        People(id='A1', name='Adele Morana', role='Founder, Chairman & CEO', reports=[], manages=['B1', 'C1', 'D1']),
        People(id='B1', name='Winston Cole', role='COO', reports=['A1'], manages=['E1']),
        People(id='C1', name='Marcus Kim', role='CFO', reports=['A1'], manages=['F1']),
        People(id='D1', name='Karin Ludovicicus', role='CPO', reports=['A1'], manages=['G1']),
        People(id='E1', name='Lea Erastos', role='Chief Business Officer', reports=['B1'], manages=['H1', 'I1']),
        People(id='F1', name='John McKinley', role='Chief Accounting Officer', reports=['C1'], manages=[]),
        People(id='G1', name='Ayda Williams', role='VP, Global Customer & Business Marketing', reports=['D1'], manages=['J1', 'K1']),
        People(id='H1', name='Zahida Mahtab', role='VP, Global Affairs & Communication', reports=['E1'], manages=[]),
        People(id='I1', name='Adelaide Zhu', role='VP, Central Services', reports=['E1'], manages=[]),
        People(id='J1', name='Gabriel Drummond', role='VP, Investor Relations', reports=['G1'], manages=[]),
        People(id='K1', name='Nicholas Brambilla', role='VP, Company Brand', reports=['G1'], manages=[]),
        People(id='L1', name='Felice Vasili', role='VP Finance', reports=['C1'], manages=[]),
        People(id='M1', name='Sandra Herminius', role='VP, Product Marketing', reports=['D1'], manages=[])
    ]
)
"""
