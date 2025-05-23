import sys
import pytest
from unittest import mock

import instructor
from pydantic import BaseModel


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Union pipe syntax requires Python 3.10+")
def test_partial_with_pipe_syntax():
    """Test that create_partial works with pipe syntax for Union types."""
    class UserWithPipe(BaseModel):
        name: str
        age: int
        category: str | None
    
    mock_client = mock.MagicMock()
    mock_client.chat.completions.create.return_value = mock.MagicMock()
    
    client = instructor.from_openai(mock_client)
    response_model = instructor.Partial[UserWithPipe]
    assert response_model is not None
