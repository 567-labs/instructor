---
title: Instructor Cookbook Collection
description: Practical examples and recipes for solving real-world problems with structured outputs
---

# Instructor Cookbooks

<div class="grid cards" markdown>

- :material-text-box-multiple: **Text Processing**

    Extract structured information from text documents

    [:octicons-arrow-right-16: View Recipes](#text-processing)

- :material-image: **Multi-Modal**

    Work with images and other media types

    [:octicons-arrow-right-16: View Recipes](#multi-modal-examples)

- :material-database: **Data Tools**

    Integrate with databases and data processing tools

    [:octicons-arrow-right-16: View Recipes](#data-tools)

- :material-server: **Deployment**

    Options for local and cloud deployment

    [:octicons-arrow-right-16: View Recipes](#deployment-options)

</div>

Our cookbooks demonstrate how to use Instructor to solve real-world problems with structured outputs. Each example includes complete code and explanations to help you implement similar solutions in your own projects.

## Text Processing

### Classification Examples

| Example | Description | Use Case |
|---------|-------------|----------|
| [Single Classification](single_classification.md) | Basic classification with a single category | Content categorization |
| [Multiple Classification](multiple_classification.md) | Handling multiple classification categories | Multi-label document tagging |
| [Enum-Based Classification](classification.md) | Using Python enums for structured classification | Standardized taxonomies |
| [Batch Classification](bulk_classification.md) | Process multiple items efficiently | High-volume text processing |
| [Batch Classification with LangSmith](batch_classification_langsmith.md) | Using LangSmith for batch processing | Performance monitoring |
| [Local Classification](local_classification.md) | Classification without external APIs | Offline processing |

### Information Extraction

| Example | Description | Use Case |
|---------|-------------|----------|
| [Entity Resolution](entity_resolution.md) | Identify and disambiguate entities | Name standardization |
| [Contact Information](extract_contact_info.md) | Extract structured contact details | CRM data entry |
| [PII Sanitization](pii.md) | Detect and redact sensitive information | Privacy compliance |
| [Citation Extraction](exact_citations.md) | Accurately extract formatted citations | Academic research |
| [Action Items](action_items.md) | Extract tasks from text | Meeting follow-ups |
| [Search Query Processing](search.md) | Structure complex search queries | Search enhancement |
| [Medication Adherence Triage](medication_adherence.md) | Convert pharmacist call notes into typed triage records with Claude | Medication management |

### Document Processing

| Example | Description | Use Case |
|---------|-------------|----------|
| [Document Segmentation](document_segmentation.md) | Divide documents into meaningful sections | Long-form content analysis |
| [Planning and Tasks](planning-tasks.md) | Break down complex queries into subtasks | Project management |
| [Knowledge Graph Generation](knowledge_graph.md) | Create relationship graphs from text | Information visualization |
| [Knowledge Graph Building](../examples/building_knowledge_graphs.md) | Build and query knowledge graphs | Semantic data modeling |
| [Chain of Density](../tutorials/6-chain-of-density.ipynb) | Implement iterative summarization | Content distillation |

## Multi-Modal Examples

### Vision Processing

| Example | Description | Use Case |
|---------|-------------|----------|
| [Table Extraction](tables_from_vision.md) | Convert image tables to structured data | Data entry automation |
| [Table Extraction with GPT-4](extracting_tables.md) | Advanced table extraction | Complex table processing |
| [Receipt Information](extracting_receipts.md) | Extract data from receipt images | Expense management |
| [Slide Content Extraction](extract_slides.md) | Convert slides to structured text | Presentation analysis |
| [Image to Ad Copy](image_to_ad_copy.md) | Generate ad text from images | Marketing automation |
| [YouTube Clip Analysis](youtube_clips.md) | Extract info from video clips | Content moderation |

### Multi-Modal Processing

| Example | Description | Use Case |
|---------|-------------|----------|
| [Gemini Multi-Modal](multi_modal_gemini.md) | Process text, images, and other data | Mixed-media analysis |

## Data Tools

### Database Integration

| Example | Description | Use Case |
|---------|-------------|----------|
| [SQLModel Integration](sqlmodel.md) | Store AI-generated data in SQL databases | Persistent storage |
| [Pandas DataFrame](pandas_df.md) | Work with structured data in Pandas | Data analysis |

### Streaming and Processing

| Example | Description | Use Case |
|---------|-------------|----------|
| [Partial Response Streaming](partial_streaming.md) | Stream partial results in real-time | Interactive applications |
| [Self-Critique and Correction](self_critique.md) | Implement self-assessment | Quality improvement |

### API Integration

| Example | Description | Use Case |
|---------|-------------|----------|
| [Content Moderation](moderation.md) | Implement content filtering | Trust & safety |
| [Cost Optimization with Batch API](batch_job_oai.md) | Reduce API costs | Production efficiency |
| [Few-Shot Learning](examples.md) | Use contextual examples in prompts | Performance tuning |

### Observability & Tracing

| Example | Description | Use Case |
|---------|-------------|----------|
| [Langfuse Tracing](tracing_with_langfuse.md) | Open-source LLM engineering | Observability & Debugging

## Deployment Options

### Model Providers

| Example | Description | Use Case |
|---------|-------------|----------|
| [Groq Cloud API](groq.md) | High-performance inference | Low-latency applications |
| [Mistral/Mixtral Models](mistral.md) | Open-source model integration | Cost-effective deployment |
| [IBM watsonx.ai](watsonx.md) | Enterprise AI platform | Business applications |

### Local Deployment

| Example | Description | Use Case |
|---------|-------------|----------|
| [Ollama Integration](ollama.md) | Local open-source models | Privacy-focused applications |

## Stay Updated

Subscribe to our newsletter for updates on new features and usage tips:

<iframe src="https://embeds.beehiiv.com/2faf420d-8480-4b6e-8d6f-9c5a105f917a?slim=true" data-test-id="beehiiv-embed" height="52" frameborder="0" scrolling="no" style="margin: 0; border-radius: 0px !important; background-color: transparent;"></iframe>

Looking for more structured learning? Check out our [Tutorial series](../tutorials/index.md) for step-by-step guides.
