from itertools import product
from pydantic import BaseModel
from writerai import Writer
import pytest

import instructor
from .util import models, modes


class Item(BaseModel):
    name: str
    price: float


class Order(BaseModel):
    items: list[Item]
    customer: str


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_format_nested_model(mode: instructor.Mode, model: str):
    client = instructor.from_writer(
        client=Writer(),
        mode=mode,
    )

    content = """
    Order Details:
    Customer: Jason
    Items:

    Name: Apple, Price: 0.50
    Name: Bread, Price: 2.00
    Name: Milk, Price: 1.50
    """

    response = client.chat.completions.create(
        model=model,
        response_model=Order,
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    assert len(response.items) == 3
    assert {x.name.lower() for x in response.items} == {"apple", "bread", "milk"}
    assert {x.price for x in response.items} == {0.5, 2.0, 1.5}
    assert response.customer.lower() == "jason"


class Book(BaseModel):
    title: str
    author: str
    genre: str
    isbn: str


class LibraryRecord(BaseModel):
    books: list[Book]
    visitor: str
    library_id: str


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_format_complex_nested_model(mode: instructor.Mode, model: str):
    client = instructor.from_writer(
        client=Writer(),
        mode=mode,
    )

    content = """
    Library visit details:
    Visitor: Jason
    Library ID: LIB123456
    Books checked out:
    - Title: The Great Adventure, Author: Jane Doe, Genre: Fantasy, ISBN: 1234567890
    - Title: History of Tomorrow, Author: John Smith, Genre: Non-Fiction, ISBN: 0987654321
    """

    response = client.chat.completions.create(
        model=model,
        response_model=LibraryRecord,
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    assert response.visitor.lower() == "jason"
    assert response.library_id == "LIB123456"
    assert len(response.books) == 2
    assert {book.title for book in response.books} == {
        "The Great Adventure",
        "History of Tomorrow",
    }
    assert {book.author for book in response.books} == {"Jane Doe", "John Smith"}
    assert {book.genre for book in response.books} == {"Fantasy", "Non-Fiction"}
    assert {book.isbn for book in response.books} == {"1234567890", "0987654321"}
