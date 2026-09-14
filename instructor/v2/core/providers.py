"""Provider detection and registry utilities.

This module contains provider-related enums and detection logic.
"""

from enum import Enum

from instructor.v2.core.mode import DEPRECATED_TO_CORE, Mode


class Provider(Enum):
    """Supported provider identifiers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    VERTEXAI = "vertexai"
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    TOGETHER = "together"
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"
    GEMINI = "gemini"
    GENAI = "genai"
    DATABRICKS = "databricks"
    CEREBRAS = "cerebras"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    WRITER = "writer"
    XAI = "xai"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    GOOGLE = "google"
    GENERATIVE_AI = "generative-ai"
    UNKNOWN = "unknown"
    BEDROCK = "bedrock"
    PERPLEXITY = "perplexity"
    OPENROUTER = "openrouter"
    HUBRIS = "hubris"


def provider_from_mode(mode: Mode, default: Provider = Provider.OPENAI) -> Provider:
    """Infer provider from a provider-specific Mode."""
    mapping = {
        Mode.ANTHROPIC_TOOLS: Provider.ANTHROPIC,
        Mode.ANTHROPIC_JSON: Provider.ANTHROPIC,
        Mode.ANTHROPIC_PARALLEL_TOOLS: Provider.ANTHROPIC,
        Mode.ANTHROPIC_REASONING_TOOLS: Provider.ANTHROPIC,
        Mode.COHERE_TOOLS: Provider.COHERE,
        Mode.COHERE_JSON_SCHEMA: Provider.COHERE,
        Mode.MISTRAL_TOOLS: Provider.MISTRAL,
        Mode.MISTRAL_STRUCTURED_OUTPUTS: Provider.MISTRAL,
        Mode.VERTEXAI_TOOLS: Provider.VERTEXAI,
        Mode.VERTEXAI_JSON: Provider.VERTEXAI,
        Mode.VERTEXAI_PARALLEL_TOOLS: Provider.VERTEXAI,
        Mode.GEMINI_TOOLS: Provider.GEMINI,
        Mode.GEMINI_JSON: Provider.GEMINI,
        Mode.GENAI_TOOLS: Provider.GENAI,
        Mode.GENAI_JSON: Provider.GENAI,
        Mode.GENAI_STRUCTURED_OUTPUTS: Provider.GENAI,
        Mode.XAI_TOOLS: Provider.XAI,
        Mode.XAI_JSON: Provider.XAI,
        Mode.CEREBRAS_TOOLS: Provider.CEREBRAS,
        Mode.CEREBRAS_JSON: Provider.CEREBRAS,
        Mode.FIREWORKS_TOOLS: Provider.FIREWORKS,
        Mode.FIREWORKS_JSON: Provider.FIREWORKS,
        Mode.WRITER_TOOLS: Provider.WRITER,
        Mode.WRITER_JSON: Provider.WRITER,
        Mode.BEDROCK_TOOLS: Provider.BEDROCK,
        Mode.BEDROCK_JSON: Provider.BEDROCK,
        Mode.PERPLEXITY_JSON: Provider.PERPLEXITY,
        Mode.OPENROUTER_STRUCTURED_OUTPUTS: Provider.OPENROUTER,
    }
    return mapping.get(mode, default)


def normalize_mode_for_provider(mode: Mode, _provider: Provider) -> Mode:
    """Apply provider-specific mode overrides before registry lookup."""
    if mode in DEPRECATED_TO_CORE:
        Mode.warn_deprecated_mode(mode)
        return DEPRECATED_TO_CORE[mode]
    return mode


def get_provider(base_url: str) -> Provider:
    """
    Detect the provider based on the base URL.

    Args:
        base_url: The base URL to analyze

    Returns:
        Provider: The detected provider enum value
    """
    normalized = str(base_url).lower()
    providers = (
        ("azure", Provider.AZURE_OPENAI),
        ("anyscale", Provider.ANYSCALE),
        ("together", Provider.TOGETHER),
        ("anthropic", Provider.ANTHROPIC),
        ("cerebras", Provider.CEREBRAS),
        ("fireworks", Provider.FIREWORKS),
        ("groq", Provider.GROQ),
        ("openai", Provider.OPENAI),
        ("mistral", Provider.MISTRAL),
        ("cohere", Provider.COHERE),
        ("generative-ai", Provider.GENERATIVE_AI),
        ("vertexai", Provider.VERTEXAI),
        ("gemini", Provider.GEMINI),
        ("genai", Provider.GENAI),
        ("google", Provider.GOOGLE),
        ("databricks", Provider.DATABRICKS),
        ("deepseek", Provider.DEEPSEEK),
        ("bedrock", Provider.BEDROCK),
        ("writer", Provider.WRITER),
        ("perplexity", Provider.PERPLEXITY),
        ("ollama", Provider.OLLAMA),
        ("localhost:11434", Provider.OLLAMA),
        ("litellm", Provider.LITELLM),
        ("openrouter", Provider.OPENROUTER),
        ("hubris", Provider.HUBRIS),
        ("x.ai", Provider.XAI),
        ("xai", Provider.XAI),
    )
    for token, provider in providers:
        if token in normalized:
            return provider
    return Provider.UNKNOWN
