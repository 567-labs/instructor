"""Transport fixtures for the HTTP library used by the installed OpenAI SDK."""

from importlib import import_module
from typing import TYPE_CHECKING

from openai import DefaultAsyncHttpxClient as AsyncClient
from openai import DefaultHttpxClient as Client

# The public SDK clients inherit from httpx.Client in 2.x and httpx2.Client in 3.x.
_transport = import_module(Client.__base__.__module__.split(".")[0])
MockTransport = _transport.MockTransport
if TYPE_CHECKING:
    from httpx import Request as Request, Response as Response
else:
    Request = _transport.Request
    Response = _transport.Response

__all__ = ["AsyncClient", "Client", "MockTransport", "Request", "Response"]
