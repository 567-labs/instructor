---
title: Managing Batch Jobs with Multi-Provider CLI
description: Learn how to create, list, cancel, and delete batch jobs using the unified Command Line Interface (CLI) across OpenAI and Anthropic providers.
---

# Using the Command Line Interface for Batch Jobs

The instructor CLI provides comprehensive functionalities for managing batch jobs across multiple providers with a unified interface. This multi-provider support allows users to leverage the strengths of different AI providers for their batch processing needs.

## Supported Providers

- **OpenAI**: Utilizes OpenAI's robust batch processing capabilities with metadata support
- **Anthropic**: Leverages Anthropic's advanced language models with cancel/delete operations

The CLI uses a unified `--provider` flag for all commands, with backward compatibility for legacy flags.

```bash
$ instructor batch --help

 Usage: instructor batch [OPTIONS] COMMAND [ARGS]...

 Manage OpenAI Batch jobs

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ cancel             Cancel a batch job                                        │
│ create             Create batch job using BatchProcessor                     │
│ create-from-file   Create a batch job from a file                            │
│ delete             Delete a completed batch job                              │
│ download-file      Download the file associated with a batch job             │
│ list               See all existing batch jobs                               │
│ results            Retrieve results from a batch job                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Creating a Batch Job

### List Jobs with Enhanced Display

```bash
$ instructor batch list --help

 Usage: instructor batch list [OPTIONS]

 See all existing batch jobs

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --limit                                  INTEGER  Total number of batch jobs │
│                                                   to show                    │
│                                                   [default: 10]              │
│ --poll                                   INTEGER  Time in seconds to wait    │
│                                                   for the batch job to       │
│                                                   complete                   │
│                                                   [default: 10]              │
│ --screen           --no-screen                    Enable or disable screen   │
│                                                   output                     │
│                                                   [default: no-screen]       │
│ --live             --no-live                      Enable live polling to     │
│                                                   continuously update the    │
│                                                   table                      │
│                                                   [default: no-live]         │
│ --provider                               TEXT     Provider to use (e.g.,     │
│                                                   'openai', 'anthropic')     │
│                                                   [default: openai]          │
│ --use-anthropic    --no-use-anthropic             [DEPRECATED] Use --model   │
│                                                   instead. Use Anthropic API │
│                                                   instead of OpenAI          │
│                                                   [default:                  │
│                                                   no-use-anthropic]          │
│ --help                                            Show this message and      │
│                                                   exit.                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

The enhanced list command now shows rich information including timestamps, duration, and provider-specific metrics:

```bash
$ instructor batch list --provider openai --limit 3

                                         Openai Batch Jobs
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ Batch ID           ┃ Status     ┃ Created    ┃ Started    ┃ Duration┃ Completed┃ Failed ┃ Total ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ batch_abc123...    │ completed  │ 07/07      │ 07/07      │ 2m      │ 15       │ 0      │ 15    │
│                    │            │ 23:48      │ 23:48      │         │          │        │       │
│ batch_def456...    │ processing │ 07/07      │ 07/07      │ 45m     │ 8        │ 0      │ 10    │
│                    │            │ 22:30      │ 22:31      │         │          │        │       │
│ batch_ghi789...    │ failed     │ 07/07      │ N/A        │ N/A     │ 0        │ 5      │ 5     │
│                    │            │ 21:15      │            │         │          │        │       │
└────────────────────┴────────────┴────────────┴────────────┴─────────┴──────────┴────────┴───────┘

$ instructor batch list --provider anthropic --limit 2

                                           Anthropic Batch Jobs
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Batch ID             ┃ Status     ┃ Created    ┃ Started    ┃ Duration┃ Succeeded┃ Errored ┃ Processing  ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━┩
│ msgbatch_abc123...   │ completed  │ 07/08      │ 07/08      │ 1m      │ 20       │ 0       │ 0           │
│                      │            │ 03:47      │ 03:47      │         │          │         │             │
│ msgbatch_def456...   │ processing │ 07/08      │ 07/08      │ 15m     │ 5        │ 0       │ 10          │
│                      │            │ 03:30      │ 03:30      │         │          │         │             │
└──────────────────────┴────────────┴────────────┴────────────┴─────────┴──────────┴─────────┴─────────────┘
```

### Create From File with Metadata Support

You can create batch jobs directly from pre-formatted .jsonl files with enhanced metadata support:

```bash
$ instructor batch create-from-file --help

 Usage: instructor batch create-from-file [OPTIONS]

 Create a batch job from a file

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --file-path                                  TEXT  File containing the    │
│                                                       batch job requests     │
│                                                       [default: None]        │
│                                                       [required]             │
│    --model                                      TEXT  Model in format        │
│                                                       'provider/model-name'  │
│                                                       (e.g., 'openai/gpt-5.4-mini', │
│                                                       'anthropic/claude-3-s… │
│                                                       [default:              │
│                                                       openai/gpt-4o-mini]    │
│    --description                                TEXT  Description/metadata   │
│                                                       for the batch job      │
│                                                       [default: Instructor   │
│                                                       batch job]             │
│    --completion-window                          TEXT  Completion window for  │
│                                                       the batch job (OpenAI  │
│                                                       only)                  │
│                                                       [default: 24h]         │
│    --use-anthropic        --no-use-anthropic          [DEPRECATED] Use       │
│                                                       --model instead. Use   │
│                                                       Anthropic API instead  │
│                                                       of OpenAI              │
│                                                       [default:              │
│                                                       no-use-anthropic]      │
│    --help                                             Show this message and  │
│                                                       exit.                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Example usage with metadata:

```bash
# OpenAI batch with custom metadata
instructor batch create-from-file \
    --file-path batch_requests.jsonl \
    --model "openai/gpt-5-nano" \
    --description "Email classification batch - production v2.1" \
    --completion-window "24h"

# Anthropic batch
instructor batch create-from-file \
    --file-path batch_requests.jsonl \
    --model "anthropic/claude-3-5-sonnet-20241022" \
    --description "Text analysis batch"
```

For creating .jsonl files, you can use the enhanced `BatchProcessor`:

```python
from instructor.batch import BatchProcessor
from pydantic import BaseModel, Field
from typing import Literal

class Classification(BaseModel):
    label: Literal["SPAM", "NOT_SPAM"] = Field(
        ..., description="Whether the email is spam or not"
    )

# Create processor
processor = BatchProcessor("openai/gpt-5-nano", Classification)

# Prepare message conversations
messages_list = [
    [
        {"role": "system", "content": "Classify the following email"},
        {"role": "user", "content": "Hello there I'm a Nigerian prince and I want to give you money"}
    ],
    [
        {"role": "system", "content": "Classify the following email"},
        {"role": "user", "content": "Meeting with Thomas has been set at Friday next week"}
    ]
]

# Create batch file
processor.create_batch_from_messages(
    messages_list=messages_list,
    file_path="batch_requests.jsonl",
    max_tokens=100,
    temperature=0.1
)
```

## Job Management Operations

### Cancelling a Batch Job

Cancel running batch jobs across all providers:

```bash
$ instructor batch cancel --help

 Usage: instructor batch cancel [OPTIONS]

 Cancel a batch job

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --batch-id                               TEXT  Batch job ID to cancel     │
│                                                   [default: None]            │
│                                                   [required]                 │
│    --provider                               TEXT  Provider to use (e.g.,     │
│                                                   'openai', 'anthropic')     │
│                                                   [default: openai]          │
│    --use-anthropic    --no-use-anthropic          [DEPRECATED] Use           │
│                                                   --provider 'anthropic'     │
│                                                   instead. Use Anthropic API │
│                                                   instead of OpenAI          │
│                                                   [default:                  │
│                                                   no-use-anthropic]          │
│    --help                                         Show this message and      │
│                                                   exit.                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Examples:

```bash
# Cancel OpenAI batch
instructor batch cancel --batch-id batch_abc123 --provider openai

# Cancel Anthropic batch
instructor batch cancel --batch-id msgbatch_def456 --provider anthropic
```

### Deleting a Batch Job

Delete completed batch jobs (supported by Anthropic):

```bash
$ instructor batch delete --help

 Usage: instructor batch delete [OPTIONS]

 Delete a completed batch job

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --batch-id        TEXT  Batch job ID to delete [default: None] [required] │
│    --provider        TEXT  Provider to use (e.g., 'openai', 'anthropic')     │
│                            [default: openai]                                 │
│    --help                  Show this message and exit.                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Examples:

```bash
# Delete Anthropic batch (supported)
instructor batch delete --batch-id msgbatch_abc123 --provider anthropic

# Try to delete OpenAI batch (shows helpful message)
instructor batch delete --batch-id batch_ghi789 --provider openai
# Note: OpenAI does not support batch deletion via API
```

### Retrieving Batch Results

Get structured results from completed batch jobs:

```bash
$ instructor batch results --help

 Usage: instructor batch results [OPTIONS]

 Retrieve results from a batch job

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --batch-id           TEXT  Batch job ID to get results from               │
│                               [default: None]                                │
│                               [required]                                     │
│ *  --output-file        TEXT  File to save the results to [default: None]    │
│                               [required]                                     │
│    --model              TEXT  Model in format 'provider/model-name' (e.g.,   │
│                               'openai/gpt-5.4-mini', 'anthropic/claude-3-sonnet')   │
│                               [default: openai/gpt-4o-mini]                  │
│    --help                     Show this message and exit.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Examples:

```bash
# Get OpenAI batch results
instructor batch results \
    --batch-id batch_abc123 \
    --output-file openai_results.jsonl \
    --model "openai/gpt-5-nano"

# Get Anthropic batch results
instructor batch results \
    --batch-id msgbatch_def456 \
    --output-file anthropic_results.jsonl \
    --model "anthropic/claude-3-5-sonnet-20241022"
```

### Downloading Raw Files (Legacy)

For compatibility, the download-file command is still available:

```bash
$ instructor batch download-file --help

 Usage: instructor batch download-file [OPTIONS]

 Download the file associated with a batch job

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --batch-id                  TEXT  Batch job ID to download                │
│                                      [default: None]                         │
│                                      [required]                              │
│ *  --download-file-path        TEXT  Path to download file to                │
│                                      [default: None]                         │
│                                      [required]                              │
│    --provider                  TEXT  Provider to use (e.g., 'openai',        │
│                                      'anthropic')                            │
│                                      [default: openai]                       │
│    --help                            Show this message and exit.             │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Provider Support Matrix

| Operation | OpenAI | Anthropic |
|-----------|--------|-----------|
| **List**  | ✅ Enhanced table | ✅ Enhanced table |
| **Create** | ✅ With metadata | ✅ File-based |
| **Cancel** | ✅ Standard API | ✅ Standard API |
| **Delete** | ❌ Not supported | ✅ Standard API |
| **Results** | ✅ Structured parsing | ✅ Structured parsing |

## Enhanced Features

- **Rich CLI Tables**: Color-coded status, timestamps, duration calculations
- **Metadata Support**: Add descriptions and custom fields to organize batches
- **Unified Commands**: Same interface works across all providers
- **Provider Detection**: Automatic provider detection from model strings
- **Error Handling**: Clear error messages and helpful notes for unsupported operations
- **Backward Compatibility**: Legacy flags still work with deprecation warnings

This comprehensive CLI interface provides efficient batch job management across all supported providers with enhanced monitoring and control capabilities.
