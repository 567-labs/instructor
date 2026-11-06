from typing import NamedTuple

import pytest

import instructor


class ClientCase(NamedTuple):
    create: str
    client_type: type[instructor.Instructor] | type[instructor.AsyncInstructor]


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(ClientCase("completion", instructor.Instructor), id="sync"),
        pytest.param(ClientCase("acompletion", instructor.AsyncInstructor), id="async"),
    ],
)
def test_litellm_client(config: ClientCase, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm = pytest.importorskip("litellm")
    client = instructor.from_litellm(getattr(litellm, config.create))
    assert isinstance(client, config.client_type)
