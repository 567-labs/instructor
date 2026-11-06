---
authors:
  - jxnl
categories:
  - instructor
comments: true
date: 2026-05-11
description: "What changed in Instructor v2, how the migration works, and why the new provider architecture is easier to extend and maintain."
draft: false
slug: whats-new-in-instructor-v2
tags:
  - Instructor
  - Structured Outputs
  - Providers
  - Python
---

# What's new in Instructor v2?

Instructor v2 is a large internal rewrite with a deliberately conservative public goal: the library should feel familiar to existing users, while becoming much easier to extend, reason about, and type-check.

The previous architecture accumulated a lot of provider-specific behavior in shared modules. That worked, but it made the codebase harder to grow. Adding a provider could mean touching response parsing, retry logic, multimodal handling, mode normalization, and client setup in several unrelated places.

V2 moves that logic into a provider-owned architecture. The external API stays recognizable. The internals become more explicit.

<!-- more -->

## The short version

V2 changes five things:

1. Provider-specific behavior lives with the provider.
2. Runtime dispatch goes through a registry of provider and mode handlers.
3. Old public modules remain as compatibility facades where that preserves the upgrade path.
4. Provider capabilities are described from one manifest instead of duplicated across routing and tests.
5. Sync, async, streaming, partials, and public return types are much easier to validate consistently.

## What stays familiar

The core usage model is still the one Instructor users already know:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-5-nano")

user = client.create(
    messages=[{"role": "user", "content": "Extract Jason, age 36."}],
    response_model=User,
)
```

That remains the heart of the library:

- define a Pydantic model
- create a provider-backed Instructor client
- ask for structured output
- receive validated Python objects

V2 is designed so that existing concepts such as retries, partials, iterables, multimodal inputs, and provider-specific client factories remain recognizable instead of being replaced wholesale.

## What changed under the hood

### 1. Providers now own provider behavior

In v1, provider-specific request formatting, response parsing, and reask logic could end up spread across shared modules.

In v2, that logic lives under:

```text
instructor/v2/providers/<provider>/
```

A provider package can own:

- client factories
- request preparation
- response parsing
- validation reasks
- streaming extraction
- templating
- multimodal encoding
- usage handling

This gives each provider a clear home and keeps shared runtime modules from slowly becoming giant provider switchboards.

### 2. Modes dispatch through registered handlers

V2 introduces a registry keyed by provider and mode. Instead of branching through one large shared path, the runtime asks:

```python
handlers = mode_registry.get_handlers(provider, mode)
```

Those handlers define how to:

- prepare the provider request
- parse the response
- build a reask after validation fails
- extract streaming chunks when needed

This makes behavior more explicit. It also means provider support can be extended without adding another round of conditional logic to the middle of the library.

### 3. Retry and reask logic is modular again

Instructor retries failed validations. That behavior is essential, but providers do not all want the same retry payload.

V2 keeps the generic retry loop in shared runtime code, while moving provider-specific reask formatting to the owning provider handler. That gives us the right split:

- shared orchestration stays shared
- wire-format behavior stays local

This matters for providers whose follow-up messages, tool payloads, or structured-output conventions differ in small but important ways.

### 4. Compatibility becomes intentional

One concern with a migration this broad is upgrade pain. V2 takes the opposite route: compatibility is explicit.

Old public modules under paths such as:

```text
instructor/core
instructor/processing
instructor/dsl
instructor/validation
```

remain as thin compatibility facades where users still need those imports. The real runtime behavior moves under `instructor/v2`.

That gives us two useful properties at once:

- existing import paths keep working where possible
- new implementation work has one obvious home

The same idea applies to legacy modes. Older provider-specific mode names can normalize into the smaller core mode system, preserving behavior while reducing long-term surface area.

### 5. Provider capabilities come from one manifest

V2 adds a provider specification layer that records:

- provider aliases
- handler modules
- supported modes
- unsupported modes
- legacy mode normalization
- public factory bindings
- optional SDK requirements

That manifest becomes the single place to understand what a provider supports. It also drives tests and runtime wiring, which means fewer duplicated lists and fewer places for support claims to drift.

## Why this is better

### New providers are easier to add

The path is clearer:

1. add the provider spec
2. implement provider handlers
3. wire the client factory
4. rely on shared registry and retry machinery

The codebase no longer asks each new provider to thread behavior through unrelated shared modules.

### Existing providers are easier to make feature-complete

Streaming, partials, schema conversion, and reasks often expose the places where a provider integration is shallow. V2 gives those features a provider-local place to live, so it is easier to finish the implementation instead of leaving small one-off gaps in shared utilities.

### Type inference is more trustworthy

The v2 migration also tightens public typing around:

- sync clients
- async clients
- partial streaming
- iterable streaming
- completion-returning helpers
- provider factory return types

That matters because Instructor users often lean on the IDE to understand the shape of returned data. V2 makes those guarantees easier to test directly.

### Imports can stay lightweight

Optional provider SDKs should not become a hidden tax for everyone else. The v2 public surface uses lazy exports and provider-local loading so importing Instructor does not eagerly drag every provider dependency into memory.

This is especially important for a library that wants to support many providers without forcing every user into the largest possible environment.

## What changes for contributors

The main contributor habit shift is simple:

- put provider-specific behavior in the provider package
- keep shared runtime modules provider-agnostic
- add reusable coverage to parametrized v2 tests where possible
- reserve provider-specific tests for genuinely provider-specific behavior

That last point matters. The v2 test suite is being pushed toward a smaller, more reusable structure:

- one parametrized suite for common client behavior
- one for handler registration
- one for shared provider/mode dispatch expectations
- provider-specific files only where the behavior is actually unique

The new parametrized suites reduce repeated provider boilerplate and centralize the shared client and handler assertions.

## A concrete mental model

If you want one picture of v2, use this:

```text
public factory
  -> provider spec
  -> provider client
  -> registry lookup by provider + mode
  -> request / response / reask handlers
  -> typed Instructor result
```

V1 made many of those steps work implicitly. V2 keeps the user-facing simplicity, but makes the implementation shape visible enough to maintain.

## Is this a breaking change?

The migration is designed to preserve the common public workflows and import paths where practical. That does not mean no code moved. A lot moved. But the migration is intentionally built around:

- compatibility facades
- normalized legacy modes
- preserved factory names
- continued support for existing structured-output workflows

The main breaking-pressure points are internal, not user-facing. The library is stricter about ownership boundaries because that is what keeps the public API stable over time.

## Why now

Instructor has grown from an OpenAI-focused structured-output helper into a broad provider toolkit. That growth is good, but the implementation needed to catch up with the product surface.

V2 is the cleanup that makes the next phase healthier:

- easier provider work
- more consistent feature parity
- better typing
- less duplicated routing logic
- more focused tests
- lighter imports
