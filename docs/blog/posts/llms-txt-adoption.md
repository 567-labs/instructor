---
authors:
  - jxnl
categories:
  - Announcements
comments: true
date: 2025-03-19
description:
  Instructor adopts llms.txt to make documentation more accessible to AI language models.
draft: false
slug: instructor-adopts-llms-txt
tags:
  - Documentation
  - AI
  - LLMs
  - Standards
---

# Instructor Adopts llms.txt: Making Documentation AI-Friendly

We're excited to announce that Instructor now implements the llms.txt specification! You can now find our llms.txt file at [python.useinstructor.com/llms.txt](https://python.useinstructor.com/llms.txt). This adoption marks an important step in making our documentation more accessible to AI language models.

<!-- more -->

## What is llms.txt?

The llms.txt specification, [developed by Jeremy Howard and the Answer.AI team](https://github.com/AnswerDotAI/llms-txt), addresses a critical challenge in AI-documentation interaction: context windows are too small for most websites, and HTML pages with navigation, ads, and JavaScript are difficult for LLMs to process effectively.

Think of llms.txt as robots.txt for AI language models - a standardized way to help AI systems understand and navigate your documentation. While robots.txt tells search engines what they can index, llms.txt helps AI models find and understand the most relevant information about your project.

## Why Instructor Adopted llms.txt

As a library focused on structured outputs from LLMs, it made perfect sense for us to implement this standard. Here's why:

1. **Better AI Integration**: Our users often interact with Instructor through AI coding assistants. Having a llms.txt file helps these tools better understand our documentation.

2. **Cleaner Documentation Access**: Instead of parsing our full HTML documentation, AI models can now access clean markdown versions of our docs.

3. **Supporting the Standard**: We believe in the importance of standardizing how AI models interact with documentation. By adopting llms.txt early, we're helping establish best practices for AI-friendly documentation.

## What This Means for Users

If you're using AI coding assistants like GitHub Copilot, Claude, or Cursor with Instructor, you should notice:

- More accurate code suggestions
- Better understanding of Instructor's features
- More relevant documentation references

For example, when you ask an AI assistant about Instructor's features, it can now directly access our markdown documentation through the llms.txt file, rather than trying to parse our HTML documentation.

## How It Works

Our llms.txt file provides:

- A concise overview of Instructor
- Links to key documentation in markdown format
- Important notes about usage and best practices
- References to example code and tutorials

AI models can use this information to better understand:

- Core concepts of Instructor
- How to use our key features
- Best practices for implementation
- Where to find detailed documentation

## Implementing llms.txt

The llms.txt specification is gaining adoption, and we encourage other Python libraries and frameworks to implement it. Here's how you can add llms.txt to your project:

1. Create a `/llms.txt` file in your documentation root
2. Follow the [standard format](https://github.com/AnswerDotAI/llms-txt#format)
3. Include key information and markdown links
4. Test with various AI assistants

## Looking Forward

This is just the beginning. As more projects adopt llms.txt, we expect to see:

- Better AI-assisted coding experiences
- More standardized documentation access
- Improved AI understanding of codebases
- Enhanced collaboration between humans and AI

We're excited to be part of establishing this standard and look forward to seeing how it evolves. If you're interested in learning more about llms.txt or want to discuss its implementation, reach out to us on [GitHub](https://github.com/instructor-ai/instructor) or [Twitter](https://x.com/jxnl.co).

For more details about the llms.txt specification, check out the [official repository](https://github.com/AnswerDotAI/llms-txt) and join the discussion about making documentation more AI-friendly.

Happy coding!