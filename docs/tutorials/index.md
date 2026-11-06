---
title: Instructor Tutorials
description: Interactive, step-by-step tutorials for learning how to use Instructor effectively
---

# Instructor Tutorials

<div class="grid cards" markdown>

- :material-school: **Learning Path**

    Follow our structured learning path to become an Instructor expert

    [:octicons-arrow-right-16: Start Learning](#tutorial-pathway)

- :material-notebook-edit: **Interactive Formats**

    Run our Jupyter notebooks in your preferred environment

    [:octicons-arrow-right-16: Run Options](#running-options)

- :material-certificate: **Skill Building**

    Gain practical skills for real-world AI applications

    [:octicons-arrow-right-16: What You'll Learn](#skills-gained)

- :material-help: **Support**

    Get help when you need it

    [:octicons-arrow-right-16: Get Help](#getting-help)

</div>

## Tutorial Pathway {#tutorial-pathway}

Our tutorials follow a carefully designed learning path from basic concepts to advanced applications. Each tutorial builds on previous concepts while introducing new techniques.

| Tutorial | Topic | Key Skills | Difficulty |
|----------|-------|------------|------------|
| 1. [Introduction to Structured Outputs](./1-introduction.ipynb) | Basic extraction | Pydantic models, basic prompting | 🟢 Beginner |
| 2. [Tips and Tricks](./2-tips.ipynb) | Best practices | Advanced models, optimization | 🟢 Beginner |
| 3. [Applications: RAG](./3-0-applications-rag.ipynb) | Retrieval-augmented generation | Information retrieval, context handling | 🟡 Intermediate |
| 4. [Applications: RAG Validation](./3-1-validation-rag.ipynb) | Validating RAG outputs | Quality control, validation hooks | 🟡 Intermediate |
| 5. [Validation Techniques](./4-validation.ipynb) | Deep validation | Custom validators, error handling | 🟡 Intermediate |
| 6. [Knowledge Graphs](./5-knowledge-graphs.ipynb) | Graph building | Entity relationships, graph visualization | 🔴 Advanced |
| 7. [Chain of Density](./6-chain-of-density.ipynb) | Summarization techniques | Iterative refinement, content density | 🔴 Advanced |
| 8. [Synthetic Data Generation](./7-synthetic-data-generation.ipynb) | Creating datasets | Data augmentation, testing data | 🔴 Advanced |

## Running Options {#running-options}

Choose your preferred environment to work through these interactive Jupyter notebooks:

<div class="grid cards" markdown>

- :material-laptop: **Run Locally**

    ```bash
    git clone https://github.com/jxnl/instructor.git
    cd instructor
    pip install -e ".[all]"
    jupyter notebook docs/tutorials/
    ```

- :material-google: **Google Colab**

    Look for the "Open in Colab" button at the top of each notebook

    Perfect for cloud execution without local setup

- :simple-mybinder: **Binder**

    Click the "Launch Binder" button to run instantly in your browser

    No installation or API keys required for basic examples

</div>

## Skills Gained {#skills-gained}

By completing this tutorial series, you'll gain practical skills in:

- **Structured Extraction**: Define Pydantic models that capture exactly the data you need
- **Advanced Validation**: Ensure LLM outputs meet your data quality requirements
- **Streaming Responses**: Process data in real-time with partial and iterative outputs
- **Complex Applications**: Build RAG systems, knowledge graphs, and more
- **Multi-Provider Support**: Work with different LLM providers using a consistent interface
- **Production Techniques**: Learn optimization strategies for real-world applications

## Setup Requirements

Before starting, make sure you have:

- **Python Environment**: Python 3.8+ installed
- **Dependencies**: Install with `pip install "instructor[all]"`
- **API Keys**: Access to OpenAI API or other supported providers
- **Basic Knowledge**: Familiarity with Python and basic LLM concepts

## Getting Help {#getting-help}

We're here to support your learning journey:

- **Documentation**: Check the [core concepts](../concepts/index.md) for detailed explanations
- **FAQ**: Browse our [frequently asked questions](../faq.md)
- **Community**: Join our [Discord server](https://discord.gg/bD9YE9JArw) for real-time help
- **Issues**: Report problems on [GitHub](https://github.com/jxnl/instructor/issues)
- **Examples**: See [practical examples](../examples/index.md) of Instructor in action

<div class="grid cards" markdown>

- :material-play-circle: **Ready to Begin?**

    Start your journey with our first tutorial on structured outputs

    [:octicons-arrow-right-16: Start Learning](./1-introduction.ipynb){: .md-button .md-button--primary }

</div>

