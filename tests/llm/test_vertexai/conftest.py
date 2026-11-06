import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def vertex_credentials() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY environment variable not set")
    pytest.importorskip("vertexai", reason="google-cloud-aiplatform is not installed")
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    try:
        _, project = google.auth.default()
    except DefaultCredentialsError:
        pytest.skip("Google application default credentials are unavailable")
    if not project:
        pytest.skip("Google application default credentials have no project ID")
