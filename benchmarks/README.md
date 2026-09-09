# Local performance baseline

Run from a checkout with Instructor's core dependencies installed:

```sh
python scripts/benchmark_local.py --output /tmp/instructor-benchmark.json
```

This uses only the standard library and existing core dependencies. No credentials,
clients, provider requests, network transport, or paid APIs are used. The script
forces child processes to import this checkout, even when the Python environment
has another editable Instructor installation. Installed distribution versions are
reported separately from the checkout revision; they need not match the lockfile.

For a quick correctness smoke run:

```sh
python scripts/benchmark_local.py --cases small --samples 2 --iterations 1 \
  --memory-batches 2 --memory-iterations 2 --memory-stream-iterations 2 \
  --output /tmp/benchmark-smoke.json
```

The defaults take several minutes, especially with allocation tracing on the large
stream. Change `--cases`, `--samples`, `--iterations`, `--warmup`, `--chunk-chars`,
`--memory-batches`, `--memory-iterations`, and `--memory-stream-iterations`
explicitly when comparing runs. The JSON records all settings. Run each revision
sequentially on the same idle machine and interpreter; preserve dependency versions and settings. Do not compare results
collected concurrently with other benchmarks.

## Fixed workloads and measured boundaries

Each input is compact ASCII JSON containing `records`, a list of records with
required integer fields `field_0` through `field_N`. Values are sequential integers
starting at zero. Zero is validated as data, not treated as absent.

| Case | Fields per record | Records |
| --- | ---: | ---: |
| small | 2 | 1 |
| medium | 8 | 16 |
| large | 32 | 128 |

Models, JSON, real OpenAI `ChatCompletion`/`ChatCompletionChunk` instances, and the
JSON handler are constructed outside timed regions. The default chunk is 128
characters. The report records actual payload bytes and chunk counts. These are
controlled synthetic inputs, not a representative sample of production traffic.
Schema size and payload size both vary across cases; streaming differences cannot
be attributed solely to payload length.

- **Cold import:** multiple fresh interpreters time only `import instructor`, with
  wall and process CPU clocks. Interpreter startup is excluded. The filesystem
  cache is not flushed, so this is process-cold, not disk-cold. No warmup is applied
  to import, and tracing is off.
- **Preparation:** `prepare_response_model` on the same original model each time.
- **Schema:** Pydantic `model_json_schema` separately from warmed, cached provider
  schema lookup. `prepare_schema` measures preparation followed by
  `generate_openai_schema`, matching the path discussed in issue #2603. It does not
  time full message/request preparation. No caches are cleared or replaced.
- **Parsing:** Pydantic validation of the JSON string separately from Instructor's
  `from_response(..., mode=JSON)` on the same SDK completion. Both include Pydantic
  work; their paired measurements help assess additional local dispatch/parsing
  costs, without claiming subtraction isolates an exact overhead.
- **Streaming:** synchronous and asynchronous provider JSON extraction plus Partial
  parsing. Each run drains the stream, including final validation. Total time and
  first yielded model time are separate. First-yield timing ends inside the consumer
  loop after the parser yields a model, not at generator construction. The first
  model may be incomplete or empty; this is not time to the first complete record.
  Async input is immediately ready,
  with no simulated network delay. A reused event loop's `run_until_complete` is
  included in total async timing, but not the internally measured first yield.
- **Sustained memory:** fresh worker per case and operation (`prepare_schema`, sync
  streaming, async streaming), with warmup then traced allocation snapshots after
  each batch. Return values are discarded. Each point records current allocation
  deltas before and after full GC and the batch's peak relative to the initial
  baseline. The default 600 preparation/schema calls exceed the schema cache's
  current 256-entry bound. Streaming defaults to 30 complete streams (six batches
  of five): tracing every partial model is much more expensive than schema lookup.
  Increase `--memory-stream-iterations` for longer streaming retention series.

Timing workers warm each operation and report all per-sample values plus median,
minimum, and maximum, in nanoseconds per operation. GC stays enabled; full collection
before each timing batch is excluded from timing. First-yield samples are separate
single-stream observations after warmup. Allocation tracing is off for all timing.

Memory tracing starts after fixture setup and warmup. It measures newly traced
Python allocations, not whole-process RSS, native SDK/Pydantic allocations, or
objects created before tracing. A small amount of harness bookkeeping is included.
Signed deltas are preserved. A plateau can reflect cache retention; a rising finite
series alone does not establish an unbounded leak. Extend batches and inspect
retainers before making a leak claim. The harness does not store every returned
class, unlike the initial issue reproducer.

## Regression policy

1. CI may run harness correctness tests and the small smoke command. Do not add
   machine-dependent timing or memory thresholds to CI.
2. For a proposed optimization, retain JSON from both revisions, including exact
   heads, environment and settings. Repeat the pair at least three times in
   alternating order on an otherwise idle machine. Compare distributions, not a
   single best sample. Note environmental differences and concurrent workloads.
3. Verify equal final parsed data and streamed behavior before discussing speed.
   CPU, first yield, total time, and retained memory are distinct outcomes.
4. Investigate repeatable changes outside normal run-to-run variation. Do not invent
   a universal percentage gate or claim network latency improvements from this test.
5. For retention concerns, increase total operations beyond relevant cache bounds
   and repeat in fresh workers. Demonstrate persistent post-GC growth and its
   retainers before describing a leak. Dynamic model churn, cancellation, backpressure,
   validators, other providers/modes and real transport need additional workloads.

Keep individual run reports outside the checkout, for example under `/tmp` as in
the commands above. Include measured revisions, environment and settings when
sharing results in a PR; do not commit generated one-run data.

Existing `scripts/benchmark_import.py` remains useful for a broader lazy-import
inventory. `examples/partial_streaming/benchmark.py` is a provider-backed example
whose results include network/model latency; it is not a local baseline.
