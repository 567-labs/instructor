---
authors:
- jxnl
categories:
- Production
- Financial Services
comments: true
date: 2025-09-11
description: London Stock Exchange Group uses Instructor in production for AI-powered market surveillance, achieving 100% precision in detecting price-sensitive news
draft: false
tags:
- Production
- Finance
- Amazon Bedrock
- Market Surveillance
- Anthropic
---

# London Stock Exchange Group Powers Market Surveillance with Instructor

London Stock Exchange Group (LSEG) has deployed Instructor in production to power their AI-driven market surveillance system, demonstrating the library's capability in mission-critical financial applications.

<!-- more -->

## Production Impact at Scale

LSEG processes over £1 trillion of securities annually from 400 members, requiring sophisticated market abuse detection systems. Their new AI-powered "Surveillance Guide" uses Instructor to integrate with Anthropic's Claude Sonnet 3.5 model through Amazon Bedrock.

## Remarkable Results

The system achieved exceptional performance metrics:
- **100% precision** in identifying non-sensitive news
- **100% recall** for detecting price-sensitive content
- Automated analysis of 250,000+ regulatory news articles
- Significant reduction in manual analyst workload

## Technical Architecture

LSEG's implementation leverages Instructor's structured output capabilities in their technical stack:

- **Instructor library**: Seamless integration with Claude Sonnet 3.5
- **Amazon Bedrock**: Scalable foundation model infrastructure
- **Custom Python pipelines**: Data processing and analysis

The system processes regulatory news through a two-step classification approach, using Instructor to ensure reliable, structured responses from the LLM for downstream analysis.

## Why This Matters

This production deployment showcases Instructor being used where accuracy and reliability are paramount - financial regulatory compliance. The system helps analysts efficiently review trades flagged for potential market abuse by automatically analyzing news sensitivity and market impact.

As Charles Kellaway from LSEG noted, the solution transforms market surveillance operations by reducing manual review time while improving consistency in price-sensitivity assessment.

## Learn More

Read the full case study: [How London Stock Exchange Group is detecting market abuse with their AI-powered Surveillance Guide on Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/how-london-stock-exchange-group-is-detecting-market-abuse-with-their-ai-powered-surveillance-guide-on-amazon-bedrock/)

Ready to build your own production-ready structured output applications? [Get started with Instructor](../../getting-started.md).
