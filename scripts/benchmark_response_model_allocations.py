"""Focused, offline allocation probe for issue #2603.

Run: PYTHONPATH=. python scripts/benchmark_response_model_allocations.py

The probe holds no results strongly unless --retain-results is selected to
reproduce the issue's measurement artifact.
Defaults: three batches of 300 calls, after ten warmup calls and cache clearing.
JSON includes Python/Pydantic versions, pre/post-GC and peak traced bytes, live
generated classes, cache statistics, and retention after releasing/clearing.
Use --scenario dynamic --batch-size 1000 for a longer dynamic-schema run, or
--scenario stable --retain-results for the control that intentionally holds results.
Tracemalloc includes Python bookkeeping; these are not RSS or native heap numbers.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import platform
import tracemalloc
import weakref
from collections.abc import Callable, Iterable
from typing import Any

import pydantic
from pydantic import BaseModel, Field, create_model
from typing_extensions import TypedDict

from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.providers.openai.schema import generate_openai_schema


class User(BaseModel):
    name: str
    age: int


class Nested(BaseModel):
    users: list[User]
    labels: dict[str, str] = Field(default_factory=dict)


class Record(TypedDict):
    name: str
    age: int


def scenarios() -> dict[str, Callable[[], Any]]:
    sequence = itertools.count()

    def dynamic() -> type[BaseModel]:
        index = next(sequence)
        fields: Any = {f"value{i}": (int, ...) for i in range(1 + index % 8)}
        return create_model(f"Dynamic{index}", **fields)

    wide_fields: Any = {f"value{i}": (str, ...) for i in range(8)}
    wide = create_model("Wide", **wide_fields)
    return {
        "stable": lambda: User,
        "wide": lambda: wide,
        "nested": lambda: Nested,
        "list": lambda: list[User],
        "iterable": lambda: Iterable[User],
        "typed_dict": lambda: Record,
        "list_typed_dict": lambda: list[Record],
        "scalar": lambda: int,
        "dynamic": dynamic,
    }


def measure(
    factory: Callable[[], Any],
    *,
    batch_size: int,
    batches: int,
    with_schema: bool,
    retain_results: bool,
) -> dict[str, Any]:
    def request() -> type[BaseModel]:
        model = prepare_response_model(factory())
        assert model is not None
        if with_schema:
            generate_openai_schema(model)
        return model

    for _ in range(10):
        request()
    generate_openai_schema.cache_clear()
    gc.collect()
    refs: list[weakref.ReferenceType[type[BaseModel]]] = []
    held: list[type[BaseModel]] = []
    rows = []
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    for batch in range(batches):
        tracemalloc.reset_peak()
        for _ in range(batch_size):
            model = request()
            refs.append(weakref.ref(model))
            if retain_results:
                held.append(model)
        del model
        before_gc, peak = tracemalloc.get_traced_memory()
        gc.collect()
        refs = [ref for ref in refs if ref() is not None]
        after_gc = tracemalloc.get_traced_memory()[0]
        rows.append(
            {
                "calls": (batch + 1) * batch_size,
                "before_gc_bytes": before_gc - baseline,
                "after_gc_bytes": after_gc - baseline,
                "peak_bytes": peak - baseline,
                "live_generated_classes": len(refs),
                "schema_cache": generate_openai_schema.cache_info()._asdict(),
            }
        )
    held.clear()
    generate_openai_schema.cache_clear()
    gc.collect()
    remaining = sum(ref() is not None for ref in refs)
    after_clear = tracemalloc.get_traced_memory()[0] - baseline
    tracemalloc.stop()
    return {
        "batches": rows,
        "live_classes_after_release_and_cache_clear": remaining,
        "bytes_after_release_and_cache_clear": after_clear,
    }


def main() -> None:
    cases = scenarios()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--retain-results", action="store_true")
    parser.add_argument("--scenario", choices=list(cases))
    args = parser.parse_args()
    if args.batch_size < 1 or args.batches < 1:
        parser.error("batch-size and batches must be positive")
    results = {}
    for name, factory in cases.items():
        if args.scenario and name != args.scenario:
            continue
        for with_schema in (False, True):
            results[f"{name}/{'schema' if with_schema else 'prepare'}"] = measure(
                factory,
                batch_size=args.batch_size,
                batches=args.batches,
                with_schema=with_schema,
                retain_results=args.retain_results,
            )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "pydantic": pydantic.__version__,
                "retain_results": args.retain_results,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
