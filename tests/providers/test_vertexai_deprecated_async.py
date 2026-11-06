from unittest.mock import MagicMock

import pytest

from instructor import AsyncInstructor, from_vertexai
from instructor.core.exceptions import ConfigurationError


@pytest.fixture
def model():
    gm = pytest.importorskip("vertexai.generative_models")
    return MagicMock(spec=gm.GenerativeModel)


def test_deprecated_async_selects_async_client(model):
    with pytest.warns(DeprecationWarning, match="'_async' is deprecated"):
        client = from_vertexai(model, _async=True)
    assert isinstance(client, AsyncInstructor)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"_async": True, "use_async": True}, id="explicit-async"),
        pytest.param({"_async": True, "use_async": False}, id="explicit-sync"),
    ],
)
def test_conflicting_async_options(model, config):
    with pytest.raises(ConfigurationError, match="Cannot provide both"):
        from_vertexai(model, **config)
