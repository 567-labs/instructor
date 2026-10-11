from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel

import instructor
from openai import OpenAI

class DummyModel(BaseModel):
    name: str

def test_retry_contract_on_empty_and_malformed_input():
    # Mock the OpenAI client
    mock_client = MagicMock(spec=OpenAI)
    
    # Configure the mock to return a sequence of responses
    mock_response_1 = MagicMock()
    mock_response_1.choices = [MagicMock()]
    mock_response_1.choices[0].message.content = ""
    
    mock_response_2 = MagicMock()
    mock_response_2.choices = [MagicMock()]
    mock_response_2.choices[0].message.content = "```json\n{malformed: 'json'\n```"
    
    mock_response_3 = MagicMock()
    mock_response_3.choices = [MagicMock()]
    mock_response_3.choices[0].message.content = '{"name": "test"}'
    
    # Set the side effect to return the responses in order
    mock_client.chat.completions.create.side_effect = [
        mock_response_1,
        mock_response_2,
        mock_response_3
    ]
    
    # Wrap the client with instructor
    client = instructor.from_openai(mock_client)
    
    # Call the mocked client with max_retries=3
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        response_model=DummyModel,
        max_retries=3
    )
    
    # Assert that the final output successfully parses
    assert isinstance(response, DummyModel)
    assert response.name == "test"
    # Ensure it retried the expected number of times
    assert mock_client.chat.completions.create.call_count == 3
