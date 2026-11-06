---
title: Instructor CLI Tools
description: Command-line utilities for monitoring API usage, fine-tuning models, and accessing documentation.
---

# Instructor CLI Tools

<div class="grid cards" markdown>

- :material-console: **Command Line Utilities**

    Powerful tools to enhance your Instructor workflow

    [:octicons-arrow-right-16: View Commands](#available-commands)

- :material-chart-line: **Usage Monitoring**

    Track API usage, costs, and token consumption

    [:octicons-arrow-right-16: Usage Guide](usage.md)

- :material-tune-vertical: **Model Fine-Tuning**

    Create and manage custom model versions

    [:octicons-arrow-right-16: Fine-Tuning Guide](finetune.md)

- :material-book-open-variant: **Documentation Access**

    Quickly access docs from your terminal

    [:octicons-arrow-right-16: Docs Command](#documentation-command)

</div>

## Getting Started

### Installation

The CLI tools are included with the Instructor package:

```bash
pip install instructor
```

### API Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Available Commands

Instructor provides several command-line utilities:

| Command | Description | Guide |
|---------|-------------|-------|
| `instructor usage` | Track API usage and costs | [Usage Guide](usage.md) |
| `instructor finetune` | Create and manage fine-tuned models | [Fine-Tuning Guide](finetune.md) |
| `instructor docs` | Quick access to documentation | [See below](#documentation-command) |

## Usage Command

Monitor your OpenAI API usage directly from the terminal:

```bash
# View total usage for the current month
instructor usage

# View usage breakdown by day
instructor usage --by-day

# Calculate cost for a specific model
instructor usage --model gpt-5.4-mini
```

For detailed usage statistics and options, see the [Usage Guide](usage.md).

## Fine-Tuning Command

Create and manage fine-tuned models with an interactive interface:

```bash
# Start the fine-tuning interface
instructor finetune
```

This launches an interactive application that guides you through the fine-tuning process. Learn more in the [Fine-Tuning Guide](finetune.md).

## Documentation Command

Quickly access Instructor documentation from your terminal:

```bash
# Open main documentation
instructor docs

# Search for specific topic
instructor docs validation

# Open specific page
instructor docs concepts/models
```

This command opens the Instructor documentation in your default web browser, making it easy to find information when you need it.

## Support & Contribution

- **GitHub**: Visit our [GitHub Repository](https://github.com/jxnl/instructor)
- **Issues**: Report bugs or request features on our [Issue Tracker](https://github.com/jxnl/instructor/issues)
- **Discord**: Join our [Discord Community](https://discord.gg/bD9YE9JArw) for support
