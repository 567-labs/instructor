---
title: Python lessons on LLM extraction with Pydantic
description: Follow Python lessons on extracting lists and nested records, validating fields, retrying failed requests, and streaming LLM responses.
---

# Learn structured data extraction with Instructor {#instructor-llm-tutorial-complete-guide-to-structured-outputs}

Start by extracting a single record. Then choose a lesson based on the shape of your data, the rules it must satisfy, or how you want to stream it.

## Getting started

* [**Provider setup**](getting_started/installation.md) - Install a provider SDK and configure your API key
* [**Your First LLM Extraction**](getting_started/first_extraction.md) - Build your first structured output with OpenAI, Anthropic, or Google LLMs
* [**Response Models Tutorial**](getting_started/response_models.md) - Master Pydantic models for LLM outputs
* [**LLM Client Setup**](../concepts/from_provider.md) - Configure Instructor for OpenAI, Anthropic, Gemini, and 15+ LLM providers

## LLM Data Extraction Patterns

Learn essential patterns for extracting structured data from language models:

* [**Simple Object Extraction**](patterns/simple_object.md) - Extract structured objects from LLM responses
* [**List Extraction Tutorial**](patterns/list_extraction.md) - Generate lists and arrays with LLMs
* [**Nested Data Structures**](patterns/nested_structure.md) - Handle complex, hierarchical LLM outputs
* [**Optional Fields**](patterns/optional_fields.md) - Manage missing data in LLM responses
* [**Field Validation**](patterns/field_validation.md) - Validate LLM outputs with Pydantic
* [**Prompt Engineering Templates**](patterns/prompt_templates.md) - Optimize prompts for better LLM extraction

## LLM Output Validation Tutorial

Ensure reliability with these validation tutorials:

* [**Validation Fundamentals**](validation/basics.md) - Core concepts for validating LLM outputs
* [**Field-Level Validation**](validation/field_level_validation.md) - Granular validation for LLM data
* [**Custom Validators**](validation/custom_validators.md) - Build domain-specific LLM validators
* [**Retry Strategies**](validation/retry_mechanisms.md) - Handle LLM failures gracefully

## Streaming LLM Responses

Real-time LLM output processing tutorials:

* [**Streaming Basics**](streaming/basics.md) - Stream LLM responses for better UX
* [**Streaming Lists**](streaming/lists.md) - Process LLM arrays in real-time
