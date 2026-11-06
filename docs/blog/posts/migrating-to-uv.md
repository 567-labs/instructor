---
authors:
  - ivanleomk
categories:
  - UV
comments: true
date: 2024-12-26
description: How we migrated from poetry to uv
draft: false
tags:
  - Migrations
---

## Why we migrated to uv

We recently migrated to uv from poetry because we wanted to benefit from it's many features such as

- Easier dependency management with automatic caching built in
- Significantly faster CI/CD compared to poetry, especially when we use the `caching` functionality provided by the Astral team
- Cargo-style lockfile that makes it easier to adopt new PEP features as they come out

We took around 1-2 days to handle the migration and we're happy with the results. On average, for CI/CD, we've seen a huge speed up for our jobs.

Here are some timings for jobs that I took from our CI/CD runs.

In general I'd say that we saw a ~3x speedup with approximately 67% reduction in time needed for the jobs once we implemented caching for the individual `uv` github actions.

<!-- more -->

| Job              | Time (Poetry)                                                                                 | Time (UV)                                                                                            |
| ---------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Ruff Formatting  | [1m16s](https://github.com/instructor-ai/instructor/actions/runs/12386936314)                 | [28s](https://github.com/instructor-ai/instructor/actions/runs/12501982235) (-63%)                   |
| Type checking    | [3m3s](https://github.com/instructor-ai/instructor/actions/runs/12488572568)                  | [39s](https://github.com/instructor-ai/instructor/actions/runs/12501974285) (-79%)                   |
| Test Python 3.9  | [1m21s](https://github.com/instructor-ai/instructor/actions/runs/12251767751/job/34177033359) | [32s](https://github.com/instructor-ai/instructor/actions/runs/12501974279/job/34880278051) (-61%)   |
| Test Python 3.10 | [1m32s](https://github.com/instructor-ai/instructor/actions/runs/12251767751/job/34177033359) | [33s](https://github.com/instructor-ai/instructor/actions/runs/12501974279/job/34880278299) (-64%)   |
| Test Python 3.11 | [3m19](https://github.com/instructor-ai/instructor/actions/runs/12251767751/job/34177034094)  | [2m48s](https://github.com/instructor-ai/instructor/actions/runs/12501974279/job/34880278480) (-16%) |

- Note that for 3.11 I subtracted 1m12 from the time because we added ~60 more tests for gemini so to make it a fair comparison I subtracted the time it took to run the gemini tests.

Most of our heavier jobs like the `Test Python` jobs are running multiple LLM calls in parallel and so the caching speedups of UV have some reduced benefit there.

## How we migrated

The first thing we did was to use an automated tool to convert our poetry lockfile to a uv compatible lockfile. For this, I followed [this thread](https://x.com/tiangolo/status/1839686030007361803) by Sebastian Ramirez on how to do the conversions.

**Step 1** : Use `uv` to run a `pdm` which will migrate your pyproject.toml and make sure to remove all of the `tool.poetry` sections. You can see the initial `pyproject.toml` [here](https://github.com/instructor-ai/instructor/blob/ad046fbca335b9133a704bed1900cda846caaf7c/pyproject.toml).

```
uvx pdm import pyproject.toml
```

Note that since you're using `uv`, make sure to also delete the `pdm` sections too and your optional groups

```toml
# dependency versions for extras
fastapi = { version = ">=0.109.2,<0.116.0", optional = true }
redis = { version = "^5.0.1", optional = true }
diskcache = { version = "^5.6.3", optional = true }
...


[tool.poetry.extras]
anthropic = ["anthropic", "xmltodict"]
groq = ["groq"]
cohere = ["cohere"]
...


[tool.pdm.build]
includes = ["instructor"]
[build-system]
requires = ["pdm-backend"]
build-backend = "pdm.backend"
```

**Step 2** : Once you've done so, since you're no longer using `poetry`, you need to update the build system. If you just delete it, you'll end up using `setuptools` by default and that will throw an error if you've declared your license using `license = {text = "MIT"}`. So you need to add the following to your `pyproject.toml`.

This is documented in this UV issue [here](https://github.com/astral-sh/uv/issues/9513) which documents a bug with setuptools not being able to handle Metadata 2.4 keys and so you need to use `hatchling` as your build backend.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 3** : Once you've done so, run uv sync to generate your `uv.lock` file to make sure you don't have any dependency issues.

### New Commands to know

Now that we migrated over from `poetry` to `uv`, there are a few new commands that you'll need to use.

1. `uv sync --all-extras --group <dependency groups you'd like to install>`: This should install all the dependencies for the project using `uv`, make sure to install the specific dependencies that you'd like to install. If you're writing docs for instance, you would run `uv sync --all-extras --group docs`

2. `uv run <command>` : This runs the specific command using the virtual environment you've created. When running our CI pipeline, we use this to ensure we're using the right environment for our commands.

## Migrating Your Workflows

We had a few workflows that were using `poetry` and so we needed to update them to use `uv` instead. As seen below there are a few main changes you'll need to make to your relevant workflow

```yaml
name: Test
on:
  pull_request:
  push:
    branches:
      - main

jobs:
  release:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }} # (1)!

      - name: Cache Poetry virtualenv
        uses: actions/cache@v2
        with:
          path: ~/.cache/pypoetry/virtualenvs
          key: ${{ runner.os }}-poetry-${{ hashFiles('**/poetry.lock') }}
          restore-keys: |
            ${{ runner.os }}-poetry-

      - name: Install Poetry
        uses: snok/install-poetry@v1.3.1 # (2)!

      - name: Install dependencies
        run: poetry install --with dev,anthropic # (3)!

      - name: Run tests
        if: matrix.python-version != '3.11'
        run: poetry run pytest tests/ -k 'not llm and not openai and not gemini and not anthropic and not cohere and not vertexai' && poetry run pytest tests/llm/test_cohere
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          COHERE_API_KEY: ${{ secrets.COHERE_API_KEY }}

      - name: Run Gemini Tests
        run: poetry run pytest tests/llm/test_gemini # (4)!
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}

      - name: Generate coverage report
        if: matrix.python-version == '3.11'
        run: |
          poetry run coverage run -m pytest tests/ -k "not docs and not anthropic and not gemini and not cohere and not vertexai and not fireworks"
          poetry run coverage report
          poetry run coverage html
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

1.  We switched over to using `uv` to install python

2.  We switch over to using astral's `astral-sh/setup-uv@v4` action to install `uv`

3.  Using `uv sync` was significantly faster than poetry install and with the cache I imagine it was even faster

4.  Instead of using `poetry run`, we use `uv run` which will start up the python virtual environment with the deps and then run the command you pass in.

We then modified the workflow to the following yml config

```yaml
name: Test
on:
  pull_request:
  push:
    branches:
      - main

jobs:
  release:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v2
      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true # (1)!

      - name: Set up Python
        run: uv python install ${{ matrix.python-version }}

      - name: Install the project
        run: uv sync --all-extras
      - name: Run tests
        if: matrix.python-version != '3.11'
        run: uv run pytest tests/ -k 'not llm and not openai and not gemini and not anthropic and not cohere and not vertexai' # (2)!
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          COHERE_API_KEY: ${{ secrets.COHERE_API_KEY }}

      - name: Run Gemini Tests
        if: matrix.python-version == '3.11'
        run: uv run pytest tests/llm/test_gemini
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}

      - name: Generate coverage report
        if: matrix.python-version == '3.11'
        run: |
          uv run coverage run -m pytest tests/ -k "not docs and not anthropic and not gemini and not cohere and not vertexai and not fireworks"
          uv run coverage report
          uv run coverage html
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

1.  Don't forget to enable the cache so that your jobs are faster

2.  Using `uv run` here is important because if you just run `pytest` it won't run the tests in your virtual environment causing them to fail.

And that was basically it! Most of the migration work was really trying to figure out what was causing the tests to fail and then slowly fixing them. We were able to easily upgrade many of our existing dependencies and make sure that everything was working as expected.

We also just did our first release with uv and it was a success!

## Conclusion

We're happy with the results and we're glad to have migrated to uv. It's been a smooth transition and we've been able to see a significant speedup in our CI/CD jobs. We're looking forward to continue using uv moving forward
