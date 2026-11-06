---
authors:
  - jxnl
categories:
  - Technical
  - Documentation
comments: true
date: 2025-08-29
description:
  Deep dive into how we integrated the mkdocs-llmstxt plugin to automatically generate llms.txt files for better AI documentation consumption.
draft: false
slug: mkdocs-llmstxt-plugin-integration
tags:
  - MkDocs
  - Plugins
  - Documentation
  - AI
  - Automation
---

# Automating llms.txt Generation with mkdocs-llmstxt Plugin

Today we integrated the `mkdocs-llmstxt` plugin into Instructor's documentation pipeline. This powerful plugin automatically generates `llms.txt` files from our MkDocs documentation, making our comprehensive guides instantly accessible to AI language models.

<!-- more -->

## About the mkdocs-llmstxt Plugin

The [`mkdocs-llmstxt` plugin](https://github.com/pawamoy/mkdocs-llmstxt) by Timothée Mazzucotelli is a brilliant solution to a common problem: how do you keep an `llms.txt` file synchronized with your evolving documentation?

### Key Features

**Automatic Generation**: The plugin generates `llms.txt` files directly from your MkDocs source files during the build process. No manual maintenance required.

**Flexible Section Control**: You can specify exactly which parts of your documentation to include:

```yaml
plugins:
  - llmstxt:
      sections:
        Getting Started:
          - index.md: Introduction to structured outputs
          - installation.md: Setup instructions
        Core Concepts:
          - concepts/*.md
```

**Clean Markdown Conversion**: The plugin converts your documentation to clean, LLM-friendly markdown format, removing HTML artifacts and navigation elements.

**Customizable Descriptions**: You can provide both short and long descriptions of your project, giving AI models the context they need.

## Our Implementation

Here's how we configured the plugin for Instructor:

```yaml
plugins:
  - llmstxt:
      markdown_description: >
        Instructor is a Python library that makes it easy to work with structured outputs 
        from large language models (LLMs). Built on top of Pydantic, it provides a simple, 
        type-safe way to extract structured data from LLM responses across multiple providers 
        including OpenAI, Anthropic, Google, and many others.
      sections:
        Getting Started:
          - index.md: Introduction to structured outputs with LLMs
          - getting-started.md: Quick start guide
          - installation.md: Installation instructions
        Core Concepts:
          - concepts/*.md
        Integrations:
          - integrations/*.md
```

### Why These Sections?

We carefully selected these sections because they provide AI models with the essential information needed to understand and use Instructor:

- **Getting Started**: Core concepts and installation
- **Core Concepts**: Deep dive into features like validation, streaming, and patterns
- **Integrations**: Provider-specific guidance for OpenAI, Anthropic, Google, and others

## Technical Benefits

### Build Integration

The plugin seamlessly integrates into our existing MkDocs build pipeline. Every time we deploy documentation updates, the `llms.txt` file is automatically regenerated with the latest content.

### Content Freshness

Unlike manually maintained `llms.txt` files, our generated version is always up-to-date. When we add new integration guides or update existing concepts, the changes are automatically reflected.

### Glob Pattern Support

The plugin supports glob patterns like `concepts/*.md`, making it easy to include entire directories without manually listing each file.

## Plugin Architecture

The `mkdocs-llmstxt` plugin works by:

1. **Parsing Configuration**: Reading your `sections` configuration during the MkDocs build
2. **File Processing**: Converting specified markdown files to clean, LLM-friendly format
3. **Content Assembly**: Combining sections with metadata into the standard llms.txt format
4. **Output Generation**: Writing the final `llms.txt` file to your site root

## Installation and Setup

Adding the plugin to your own MkDocs project is straightforward:

```bash
pip install mkdocs-llmstxt
```

Then add it to your `mkdocs.yml`:

```yaml
site_url: https://your-site.com/  # Required for the plugin

plugins:
  - llmstxt:
      markdown_description: Description of your project
      sections:
        Documentation:
          - docs/*.md
```

## Resources

- [mkdocs-llmstxt Plugin](https://github.com/pawamoy/mkdocs-llmstxt)
- [llms.txt Specification](https://github.com/AnswerDotAI/llms-txt)
- [Instructor Documentation](https://python.useinstructor.com/)

Special thanks to Timothée Mazzucotelli for creating this excellent plugin!
