"""Offline Instructor microbenchmarks; see benchmarks/README.md."""

from __future__ import annotations

import argparse
import asyncio
import gc
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[1]
CASES = {"small": (2, 1), "medium": (8, 16), "large": (32, 128)}


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "samples": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def timed(
    fn: Callable[[], Any], samples: int, iterations: int, warmup: int
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    wall, cpu = [], []
    for _ in range(samples):
        gc.collect()
        start_cpu, start = time.process_time_ns(), time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        wall.append((time.perf_counter_ns() - start) / iterations)
        cpu.append((time.process_time_ns() - start_cpu) / iterations)
    return {"wall_ns_per_op": summary(wall), "cpu_ns_per_op": summary(cpu)}


def retained(
    fn: Callable[[], Any], batches: int, iterations: int, warmup: int
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    gc.collect()
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    points = []
    try:
        for batch in range(batches):
            tracemalloc.reset_peak()
            for _ in range(iterations):
                fn()  # Deliberately do not retain the returned class/model.
            pre_gc, peak = tracemalloc.get_traced_memory()
            gc.collect()
            post_gc = tracemalloc.get_traced_memory()[0]
            points.append(
                {
                    "operations": (batch + 1) * iterations,
                    "pre_gc_delta_bytes": pre_gc - baseline,
                    "post_gc_delta_bytes": post_gc - baseline,
                    "batch_peak_delta_bytes": peak - baseline,
                }
            )
    finally:
        tracemalloc.stop()
    return {"baseline_traced_bytes": baseline, "points": points}


def fixture(case: str, chunk_chars: int) -> dict[str, Any]:
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessage,
    )
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
    )
    from pydantic import create_model
    from instructor.v2.core.function_calls import ResponseSchema
    from instructor import Partial, Mode
    from instructor.v2.core.response_model import prepare_response_model
    from instructor.v2.providers.openai.handlers import OpenAIJSONHandler
    from instructor.v2.providers.openai.schema import generate_openai_schema

    width, rows = CASES[case]
    fields: dict[str, Any] = {f"field_{i}": (int, ...) for i in range(width)}
    record = create_model("Record", **fields)
    model = create_model("Batch", records=(list[record], ...))
    expected = {
        "records": [
            {f"field_{i}": n * width + i for i in range(width)} for n in range(rows)
        ]
    }
    payload = json.dumps(expected, separators=(",", ":"))
    prepared = cast(type[ResponseSchema], prepare_response_model(model))
    # The Partial stub omits the runtime streaming classmethods.
    partial = cast(Any, Partial[model])
    handler = OpenAIJSONHandler()
    completion = ChatCompletion(
        id="local",
        created=0,
        model="offline",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=payload),
            )
        ],
    )
    chunks = [
        ChatCompletionChunk(
            id="local",
            created=0,
            model="offline",
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=payload[i : i + chunk_chars]),
                    finish_reason=None,
                )
            ],
        )
        for i in range(0, len(payload), chunk_chars)
    ]

    def prepare_schema() -> Any:
        return generate_openai_schema(prepare_response_model(model))

    def stream_sync() -> tuple[int, int, Any]:
        start = time.perf_counter_ns()
        first = None
        count = 0
        final = None
        for result in partial.from_streaming_response(
            iter(chunks), handler.extract_streaming_json
        ):
            if first is None:
                # The parser has yielded a model before this timestamp is taken.
                first = time.perf_counter_ns() - start
            count += 1
            final = result
        if first is None:
            raise RuntimeError("Stream yielded no structured result")
        return first, count, final

    async def stream_async() -> tuple[int, int, Any]:
        async def source():
            for chunk in chunks:
                yield chunk

        start = time.perf_counter_ns()
        first = None
        count = 0
        final = None
        async for result in partial.from_streaming_response_async(
            source(), handler.extract_streaming_json_async
        ):
            if first is None:
                # The parser has yielded a model before this timestamp is taken.
                first = time.perf_counter_ns() - start
            count += 1
            final = result
        if first is None:
            raise RuntimeError("Async stream yielded no structured result")
        return first, count, final

    return {
        "expected": expected,
        "input": {
            "payload_bytes": len(payload.encode()),
            "fields_per_record": width,
            "records": rows,
            "chunks": len(chunks),
        },
        "operations": {
            "prepare": lambda: prepare_response_model(model),
            "schema_pydantic": model.model_json_schema,
            "schema_cached": lambda: generate_openai_schema(prepared),
            "prepare_schema": prepare_schema,
            "parse_pydantic": lambda: model.model_validate_json(payload),
            "parse_instructor": lambda: prepared.from_response(
                completion, mode=Mode.JSON
            ),
            "stream_sync": stream_sync,
            "stream_async": stream_async,
        },
    }


def worker(args: argparse.Namespace) -> dict[str, Any]:
    f = fixture(args.case, args.chunk_chars)
    loop = asyncio.new_event_loop()
    operations = f["operations"]
    stream_async = operations["stream_async"]
    operations["stream_async"] = lambda: loop.run_until_complete(stream_async())
    try:
        # Correctness is checked outside measured regions; all SDK data is built above.
        for name in ("parse_pydantic", "parse_instructor"):
            assert operations[name]().model_dump() == f["expected"]
        for name in ("stream_sync", "stream_async"):
            _, count, final = operations[name]()
            assert count > 0 and final.model_dump() == f["expected"]
        if args.worker == "memory":
            return retained(
                operations[args.operation],
                args.memory_batches,
                args.memory_iterations
                if args.operation == "prepare_schema"
                else args.memory_stream_iterations,
                args.warmup,
            )
        results = {
            name: timed(fn, args.samples, args.iterations, args.warmup)
            for name, fn in operations.items()
        }
        for name in ("stream_sync", "stream_async"):
            # Separate first-yield samples still drain the stream to run final validation.
            results[name]["first_yield_wall_ns"] = summary(
                [operations[name]()[0] for _ in range(args.samples)]
            )
        return {
            "input": f["input"],
            "timing": results,
        }
    finally:
        loop.close()


def child(arguments: list[str]) -> dict[str, Any]:
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if result.returncode:
        raise RuntimeError(
            f"Benchmark worker failed ({result.returncode}):\n{result.stderr}"
        )
    return json.loads(result.stdout)


def cold_import(samples: int) -> dict[str, Any]:
    code = "import time; c=time.process_time_ns(); w=time.perf_counter_ns(); import instructor; print(time.perf_counter_ns()-w, time.process_time_ns()-c)"
    values = []
    for _ in range(samples):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(ROOT)),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        values.append([int(v) for v in result.stdout.split()])
    return {
        "wall_ns": summary([v[0] for v in values]),
        "cpu_ns": summary([v[1] for v in values]),
    }


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--samples", type=positive, default=7)
    parser.add_argument("--iterations", type=positive, default=10)
    parser.add_argument("--warmup", type=positive, default=3)
    parser.add_argument("--chunk-chars", type=positive, default=128)
    parser.add_argument("--memory-batches", type=positive, default=6)
    parser.add_argument("--memory-iterations", type=positive, default=100)
    parser.add_argument("--memory-stream-iterations", type=positive, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worker", choices=("timing", "memory"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--case", choices=CASES, default="small", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--operation",
        choices=("prepare_schema", "stream_sync", "stream_async"),
        default="prepare_schema",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.worker:
        sys.path.insert(0, str(ROOT))
        result = worker(args)
    else:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        result = {
            "format_version": 1,
            "environment": {
                "revision": revision,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "harness_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "dirty": bool(
                    subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
                ),
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "gc_threshold": gc.get_threshold(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "instructor",
                        "pydantic",
                        "pydantic-core",
                        "openai",
                        "jiter",
                    )
                },
            },
            "config": {
                key: value
                for key, value in vars(args).items()
                if key not in ("output", "worker", "case", "operation")
            },
            "cold_import": cold_import(args.samples),
            "cases": {},
        }
        options = [
            arg
            for key, value in result["config"].items()
            if key != "cases"
            for arg in ("--" + key.replace("_", "-"), str(value))
        ]
        for case in args.cases:
            print(f"Measuring {case}...", file=sys.stderr)
            data = child(["--worker", "timing", "--case", case, *options])
            data["memory"] = {
                operation: child(
                    [
                        "--worker",
                        "memory",
                        "--case",
                        case,
                        "--operation",
                        operation,
                        *options,
                    ]
                )
                for operation in ("prepare_schema", "stream_sync", "stream_async")
            }
            result["cases"][case] = data
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
