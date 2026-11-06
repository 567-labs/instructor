from pydantic import BaseModel

from openai import OpenAI
import instructor

client = OpenAI()

client = instructor.from_openai(client)


class User(BaseModel):
    name: str
    email: str


class MeetingInfo(BaseModel):
    user: User
    date: str
    location: str
    budget: int
    deadline: str


data = """
Jason Liu jason@gmail.com
Meeting Date: 2024-01-01
Meeting Location: 1234 Main St
Meeting Budget: $1000
Meeting Deadline: 2024-01-31
"""
stream1 = client.chat.completions.create_partial(
    model="gpt-4",
    response_model=MeetingInfo,
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the meeting and the users {data}",
        },
    ],
    stream=True,
)  # type: ignore

for message in stream1:
    print(message)
"""
ser={} date=None location=None budget=None deadline=None
user={} date=None location=None budget=None deadline=None
user={} date=None location=None budget=None deadline=None
user={} date=None location=None budget=None deadline=None
user=PartialUser(name=None, email=None) date=None location=None budget=None deadline=None
user=PartialUser(name=None, email=None) date=None location=None budget=None deadline=None
user=PartialUser(name=None, email=None) date=None location=None budget=None deadline=None
user=PartialUser(name=None, email=None) date=None location=None budget=None deadline=None
user=PartialUser(name=None, email=None) date=None location=None budget=None deadline=None
user=PartialUser(name=None, email=None) date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email=None) date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email=None) date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email=None) date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email=None) date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email=None) date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email=None) date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email=None) date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email=None) date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email=None) date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date=None location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location=None budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=None deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=100 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline=None
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline='2024-01-31'
user=PartialUser(name='Jason Liu', email='jason@gmail.com') date='2024-01-01' location='1234 Main St' budget=1000 deadline='2024-01-31'
"""
