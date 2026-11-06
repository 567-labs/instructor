---
authors:
  - jxnl
categories:
  - Announcements
comments: true
date: 2025-08-29
description:
  Instructor now automatically generates llms.txt files for better AI documentation access.
draft: false
slug: llms-txt-support
tags:
  - Documentation
  - AI
---

# Instructor Now Supports llms.txt

We've added automatic `llms.txt` generation to Instructor's documentation using the [`mkdocs-llmstxt`](https://github.com/pawamoy/mkdocs-llmstxt) plugin.

<!-- more -->

## What is llms.txt?

The [`llms.txt` specification](https://github.com/AnswerDotAI/llms-txt) helps AI coding assistants access clean documentation without parsing complex HTML. Think "robots.txt for LLMs."

## What This Means

Your AI coding assistant (Copilot, Claude, Cursor) now gets better access to:
- Getting started guides
- Core concepts and patterns  
- Provider integration docs

This should result in more accurate suggestions and better understanding of Instructor's features.

## Implementation

We're using the `mkdocs-llmstxt` plugin to automatically generate our `llms.txt` from our existing markdown documentation. Every time we update our docs, the `llms.txt` file stays current automatically.

No manual maintenance, always up-to-date.

## Resources

- [llms.txt Specification](https://github.com/AnswerDotAI/llms-txt)
- [mkdocs-llmstxt Plugin](https://github.com/pawamoy/mkdocs-llmstxt)