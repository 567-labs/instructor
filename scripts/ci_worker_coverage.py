"""Measure pytest processes without instrumenting the subprocesses tests launch."""

import pytest
from coverage import Coverage

MEASUREMENT = pytest.StashKey[Coverage]()


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config: pytest.Config) -> None:
    measurement = Coverage(data_suffix=True)
    early_config.stash[MEASUREMENT] = measurement
    measurement.start()


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    measurement = session.config.stash.get(MEASUREMENT, None)
    if measurement is not None:
        measurement.stop()
        measurement.save()
