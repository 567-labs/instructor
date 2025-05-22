import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from instructor.client_vertexai import from_vertexai
from instructor.exceptions import ConfigurationError

class User(BaseModel):
    name: str
    age: int

@patch('instructor.client_vertexai.isinstance', return_value=True)
def test_deprecated_async_warning(_):
    """Test that using _async parameter raises a deprecation warning."""
    mock_model = MagicMock()
    mock_model.generate_content = MagicMock()
    mock_model.generate_content_async = MagicMock()
    
    with pytest.warns(DeprecationWarning, match="'_async' is deprecated. Use 'use_async' instead."):
        client = from_vertexai(
            mock_model, 
            _async=True
        )

@patch('instructor.client_vertexai.isinstance', return_value=True)
def test_both_async_params_error(_):
    """Test that providing both _async and use_async raises an error."""
    mock_model = MagicMock()
    mock_model.generate_content = MagicMock()
    mock_model.generate_content_async = MagicMock()
    
    with pytest.raises(ConfigurationError, match="Cannot provide both '_async' and 'use_async'. Use 'use_async' instead."):
        client = from_vertexai(
            mock_model, 
            _async=True,
            use_async=True
        )
