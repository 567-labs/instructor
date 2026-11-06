---
title: API Reference Guide
description: Explore the comprehensive API reference with details on instructors, validation, iteration, and function calls.
---

# API Reference

Core modes are the recommended default. Legacy provider-specific modes still
work but are deprecated and will show warnings. See the
[Mode Migration Guide](concepts/mode-migration.md) for details.

## Core Clients

The main client classes for interacting with LLM providers.

::: instructor.core.client.Instructor

::: instructor.core.client.AsyncInstructor

::: instructor.core.client.Response

## Client Creation

Functions to create Instructor clients from various providers.

::: instructor.auto_client.from_provider

::: instructor.v2.providers.openai.client.from_openai

::: instructor.v2.providers.litellm.client.from_litellm

## DSL Components

Domain-specific language components for advanced patterns and data handling.

::: instructor.dsl.iterable

::: instructor.dsl.partial

::: instructor.dsl.parallel

::: instructor.dsl.maybe

::: instructor.dsl.citation

## Function Calls & Schema

Classes and functions for defining and working with function call schemas.

::: instructor.function_calls

::: instructor.v2.core.function_calls.OpenAISchema

::: instructor.v2.core.function_calls.openai_schema

::: instructor.v2.core.schema.generate_openai_schema

::: instructor.v2.core.schema.generate_anthropic_schema

::: instructor.v2.core.schema.generate_gemini_schema

## Validation

Validation utilities for LLM outputs and async validation support.

::: instructor.validation

::: instructor.v2.validation.llm_validator

::: instructor.v2.validation.openai_moderation

## Batch Processing

Batch processing utilities for handling multiple requests efficiently.

::: instructor.batch

::: instructor.batch.BatchProcessor

::: instructor.batch.BatchRequest

::: instructor.batch.BatchJob

## Distillation

Tools for distillation and fine-tuning workflows.

::: instructor.distil

::: instructor.distil.FinetuneFormat

::: instructor.distil.Instructions

## Multimodal

Support for image and audio content in LLM requests.

::: instructor.processing.multimodal

::: instructor.v2.core.multimodal.Image

::: instructor.v2.core.multimodal.Audio

## Mode & Provider

Enumerations for modes and providers.

::: instructor.mode.Mode

::: instructor.utils.providers.Provider

## Exceptions

Exception classes for error handling.

::: instructor.core.exceptions

## Hooks

Event hooks system for monitoring and intercepting LLM interactions.

::: instructor.core.hooks

::: instructor.core.hooks.Hooks

::: instructor.core.hooks.HookName

## Patch Functions

Decorators for patching LLM client methods.

::: instructor.core.patch

::: instructor.core.patch.apatch
