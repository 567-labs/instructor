---
title: OpenAI API Usage CLI Guide
description: Learn how to monitor OpenAI API usage with the CLI tool, including commands for viewing data by model, date, and cost.
---

# Using the OpenAI API Usage CLI

The OpenAI API Usage CLI tool provides functionalities for monitoring your OpenAI API usage, breaking it down by model, date, and cost.

!!! warning "Legacy estimates"
    This command uses a legacy OpenAI usage endpoint and a static model-price
    table. It does not reconcile invoices or account for current cache, tool,
    modality, batch, or service-tier pricing. Model-prefix fallback can also
    apply the wrong rate to newer model variants. An empty result does not
    establish zero usage. Use provider billing records for financial reporting.

## Monitoring API Usage

### View Usage Options

```sh
$ instructor usage --help

 Usage: instructor usage [OPTIONS] COMMAND [ARGS]...

 Check OpenAI API usage data

╭─ Options ───────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                     │
╰─────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────╮
│ list       Displays OpenAI API usage data for the past N days.  │
╰─────────────────────────────────────────────────────────────────╯
```

### List Usage for Specific Number of Days

To display API usage for the past 3 days, use the following command:

```sh
$ instructor usage list --n 3
```

This will output a table similar to:

```plaintext
                 Usage Summary by Date, Snapshot, and Cost
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Date       ┃ Snapshot ID               ┃ Total Requests ┃ Total Cost ($) ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 2023-09-04 │ gpt-5.4-mini                │             44 │           0.68 │
│ 2023-09-04 │ gpt-5.4-mini    │            195 │           0.84 │
│ 2023-09-04 │ text-embedding-ada-002-v2 │            276 │           0.00 │
│ 2023-09-04 │ gpt-4-32k-0613            │            328 │          49.45 │
└────────────┴───────────────────────────┴────────────────┴────────────────┘
```

### List Usage for Today

To display the API usage for today, simply run:

```sh
$ instructor usage list
```

# Contributions

We aim to provide a light wrapper around the API rather than offering a complete CLI. Contributions are welcome! Please feel free to make an issue at [jxnl/instructor/issues](https://github.com/jxnl/instructor/issues) or submit a pull request.
