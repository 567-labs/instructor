"""Harness contracts, deliberately without machine-dependent timing gates."""

import argparse
import asyncio
import gc
import importlib.util
from pathlib import Path
import tracemalloc
import weakref

import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_local",
    Path(__file__).resolve().parents[2] / "scripts/benchmark_local.py",
)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


@pytest.mark.parametrize("case", bench.CASES)
def test_fixture_roundtrip_and_streams(case):
    fixture = bench.fixture(case, 128)
    operations = fixture["operations"]
    for name in ("parse_pydantic", "parse_instructor"):
        assert operations[name]().model_dump() == fixture["expected"]
    sync = operations["stream_sync"]()
    asynchronous = asyncio.run(operations["stream_async"]())
    for first, count, final in (sync, asynchronous):
        assert first >= 0
        assert count > 0
        assert final.model_dump() == fixture["expected"]


def test_retention_discards_returned_objects_and_stops_tracing():
    references = []

    class Result:
        def __init__(self):
            self.cycle = self

    def operation():
        result = Result()
        references.append(weakref.ref(result))
        return result

    report = bench.retained(operation, batches=3, iterations=2, warmup=1)
    gc.collect()
    assert all(ref() is None for ref in references)
    assert [point["operations"] for point in report["points"]] == [2, 4, 6]
    assert not tracemalloc.is_tracing()


def test_retention_propagates_failure_and_stops_tracing():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("invalid benchmark input")

    with pytest.raises(ValueError, match="invalid benchmark input"):
        bench.retained(operation, batches=1, iterations=1, warmup=1)
    assert not tracemalloc.is_tracing()


def test_zero_is_a_measurement_not_missing():
    assert bench.summary([0.0, 0.0])["median"] == 0
    with pytest.raises(argparse.ArgumentTypeError):
        bench.positive("0")


def test_child_preserves_error_details():
    with pytest.raises(RuntimeError, match="unrecognized arguments"):
        bench.child(["--not-a-benchmark-option"])
