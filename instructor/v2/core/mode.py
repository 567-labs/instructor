import enum
import warnings


# Track if deprecation warnings have been shown
_functions_deprecation_shown = False
_reasoning_tools_deprecation_shown = False


class Mode(enum.Enum):
    """
    Mode enumeration for patching LLM API clients.

    Each mode determines how the library formats and structures requests
    to different provider APIs and how it processes their responses.
    """

    # OpenAI modes
    FUNCTIONS = "function_call"  # Deprecated
    PARALLEL_TOOLS = "parallel_tool_call"
    TOOLS = "tool_call"
    TOOLS_STRICT = "tools_strict"
    JSON = "json_mode"
    JSON_O1 = "json_o1"
    MD_JSON = "markdown_json_mode"
    JSON_SCHEMA = "json_schema_mode"

    # Add new modes to support responses api
    RESPONSES_TOOLS = "responses_tools"
    RESPONSES_TOOLS_WITH_INBUILT_TOOLS = "responses_tools_with_inbuilt_tools"

    # XAI modes
    XAI_JSON = "xai_json"
    XAI_TOOLS = "xai_tools"

    # Anthropic modes
    ANTHROPIC_TOOLS = "anthropic_tools"
    ANTHROPIC_REASONING_TOOLS = "anthropic_reasoning_tools"
    ANTHROPIC_JSON = "anthropic_json"
    ANTHROPIC_PARALLEL_TOOLS = "anthropic_parallel_tools"

    # Mistral modes
    MISTRAL_TOOLS = "mistral_tools"
    MISTRAL_STRUCTURED_OUTPUTS = "mistral_structured_outputs"

    # Vertex AI & Google modes
    VERTEXAI_TOOLS = "vertexai_tools"
    VERTEXAI_JSON = "vertexai_json"
    VERTEXAI_PARALLEL_TOOLS = "vertexai_parallel_tools"
    GEMINI_JSON = "gemini_json"
    GEMINI_TOOLS = "gemini_tools"
    GENAI_TOOLS = "genai_tools"
    GENAI_JSON = "genai_json"
    GENAI_STRUCTURED_OUTPUTS = (
        "genai_structured_outputs"  # Backwards compatibility alias
    )

    # Cohere modes
    COHERE_TOOLS = "cohere_tools"
    COHERE_JSON_SCHEMA = "json_object"

    # Cerebras modes
    CEREBRAS_TOOLS = "cerebras_tools"
    CEREBRAS_JSON = "cerebras_json"

    # Fireworks modes
    FIREWORKS_TOOLS = "fireworks_tools"
    FIREWORKS_JSON = "fireworks_json"

    # Other providers
    WRITER_TOOLS = "writer_tools"
    WRITER_JSON = "writer_json"
    BEDROCK_TOOLS = "bedrock_tools"
    BEDROCK_JSON = "bedrock_json"
    PERPLEXITY_JSON = "perplexity_json"
    OPENROUTER_STRUCTURED_OUTPUTS = "openrouter_structured_outputs"

    # MiniMax modes
    MINIMAX_TOOLS = "minimax_tools"
    MINIMAX_JSON = "minimax_json"

    # Classification helpers
    @classmethod
    def tool_modes(cls) -> set["Mode"]:
        """Returns a set of all tool-based modes."""
        return {
            cls.FUNCTIONS,
            cls.PARALLEL_TOOLS,
            cls.TOOLS,
            cls.TOOLS_STRICT,
            cls.ANTHROPIC_TOOLS,
            cls.ANTHROPIC_REASONING_TOOLS,
            cls.ANTHROPIC_PARALLEL_TOOLS,
            cls.MISTRAL_TOOLS,
            cls.VERTEXAI_TOOLS,
            cls.VERTEXAI_PARALLEL_TOOLS,
            cls.GEMINI_TOOLS,
            cls.COHERE_TOOLS,
            cls.CEREBRAS_TOOLS,
            cls.FIREWORKS_TOOLS,
            cls.WRITER_TOOLS,
            cls.BEDROCK_TOOLS,
            cls.OPENROUTER_STRUCTURED_OUTPUTS,
            cls.MISTRAL_STRUCTURED_OUTPUTS,
            cls.XAI_TOOLS,
            cls.GENAI_TOOLS,
            cls.RESPONSES_TOOLS,
            cls.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
            cls.MINIMAX_TOOLS,
        }

    @classmethod
    def json_modes(cls) -> set["Mode"]:
        """Returns a set of all JSON-based modes."""
        return {
            cls.JSON,
            cls.JSON_O1,
            cls.MD_JSON,
            cls.JSON_SCHEMA,
            cls.ANTHROPIC_JSON,
            cls.VERTEXAI_JSON,
            cls.GEMINI_JSON,
            cls.COHERE_JSON_SCHEMA,
            cls.CEREBRAS_JSON,
            cls.FIREWORKS_JSON,
            cls.WRITER_JSON,
            cls.BEDROCK_JSON,
            cls.PERPLEXITY_JSON,
            cls.OPENROUTER_STRUCTURED_OUTPUTS,
            cls.MISTRAL_STRUCTURED_OUTPUTS,
            cls.XAI_JSON,
            cls.MINIMAX_JSON,
        }

    @classmethod
    def parallel_modes(cls) -> set["Mode"]:
        """Return canonical and compatibility aliases for parallel tool modes."""
        return {
            cls.PARALLEL_TOOLS,
            cls.ANTHROPIC_PARALLEL_TOOLS,
            cls.VERTEXAI_PARALLEL_TOOLS,
        }

    @classmethod
    def warn_mode_functions_deprecation(cls):
        """
        Warn about FUNCTIONS mode deprecation.

        Shows the warning only once per session to avoid spamming logs
        with the same message.
        """
        global _functions_deprecation_shown
        if not _functions_deprecation_shown:
            warnings.warn(
                "The FUNCTIONS mode is deprecated and will be removed in future versions",
                DeprecationWarning,
                stacklevel=2,
            )
            _functions_deprecation_shown = True

    @classmethod
    def warn_anthropic_reasoning_tools_deprecation(cls):
        """
        Warn about ANTHROPIC_REASONING_TOOLS mode deprecation.

        ANTHROPIC_TOOLS now supports extended thinking/reasoning via the
        'thinking' parameter. Use Mode.ANTHROPIC_TOOLS with thinking={'type': 'enabled'}
        instead of Mode.ANTHROPIC_REASONING_TOOLS.

        Shows the warning only once per session to avoid spamming logs
        with the same message.
        """
        global _reasoning_tools_deprecation_shown
        if not _reasoning_tools_deprecation_shown:
            warnings.warn(
                "Mode.ANTHROPIC_REASONING_TOOLS is deprecated. "
                "Use Mode.ANTHROPIC_TOOLS with thinking={'type': 'enabled', 'budget_tokens': ...} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _reasoning_tools_deprecation_shown = True

    @classmethod
    def warn_deprecated_mode(cls, mode: "Mode") -> None:
        """Warn about provider-specific mode deprecation.

        Uses a single warning per mode per process to reduce noise.
        """
        if mode not in DEPRECATED_TO_CORE:
            return
        if mode in _deprecated_modes_warned:
            return
        _deprecated_modes_warned.add(mode)
        replacement = DEPRECATED_TO_CORE[mode]
        warnings.warn(
            f"Mode.{mode.name} is deprecated and will be removed in v3.0. "
            f"Use Mode.{replacement.name} instead. "
            "The provider is determined by the client (from_openai, from_anthropic, etc.), "
            "not by the mode.",
            DeprecationWarning,
            stacklevel=3,
        )


_deprecated_modes_warned: set[Mode] = set()

# Maps deprecated modes to their core replacements.
# NOTE: Mode.JSON is not deprecated because GenAI uses it.
DEPRECATED_TO_CORE: dict[Mode, Mode] = {
    # OpenAI legacy modes
    Mode.FUNCTIONS: Mode.TOOLS,
    Mode.TOOLS_STRICT: Mode.TOOLS,
    Mode.JSON_O1: Mode.JSON_SCHEMA,
    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: Mode.RESPONSES_TOOLS,
    # Anthropic legacy modes
    Mode.ANTHROPIC_TOOLS: Mode.TOOLS,
    Mode.ANTHROPIC_REASONING_TOOLS: Mode.TOOLS,
    Mode.ANTHROPIC_JSON: Mode.JSON,
    Mode.ANTHROPIC_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
    # GenAI legacy modes
    Mode.GENAI_TOOLS: Mode.TOOLS,
    Mode.GENAI_JSON: Mode.JSON,
    Mode.GENAI_STRUCTURED_OUTPUTS: Mode.JSON,
    # Mistral legacy modes
    Mode.MISTRAL_TOOLS: Mode.TOOLS,
    Mode.MISTRAL_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
    # Cohere legacy modes
    Mode.COHERE_TOOLS: Mode.TOOLS,
    Mode.COHERE_JSON_SCHEMA: Mode.JSON_SCHEMA,
    # xAI legacy modes
    Mode.XAI_TOOLS: Mode.TOOLS,
    Mode.XAI_JSON: Mode.MD_JSON,
    # Fireworks legacy modes
    Mode.FIREWORKS_TOOLS: Mode.TOOLS,
    Mode.FIREWORKS_JSON: Mode.MD_JSON,
    # Cerebras legacy modes
    Mode.CEREBRAS_TOOLS: Mode.TOOLS,
    Mode.CEREBRAS_JSON: Mode.MD_JSON,
    # Writer legacy modes
    Mode.WRITER_TOOLS: Mode.TOOLS,
    Mode.WRITER_JSON: Mode.MD_JSON,
    # Bedrock legacy modes
    Mode.BEDROCK_TOOLS: Mode.TOOLS,
    Mode.BEDROCK_JSON: Mode.MD_JSON,
    # Perplexity legacy modes
    Mode.PERPLEXITY_JSON: Mode.MD_JSON,
    # VertexAI legacy modes
    Mode.VERTEXAI_TOOLS: Mode.TOOLS,
    Mode.VERTEXAI_JSON: Mode.MD_JSON,
    Mode.VERTEXAI_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
    # Gemini legacy modes
    Mode.GEMINI_TOOLS: Mode.TOOLS,
    Mode.GEMINI_JSON: Mode.MD_JSON,
    # OpenRouter legacy modes
    Mode.OPENROUTER_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
    # MiniMax legacy modes
    Mode.MINIMAX_TOOLS: Mode.TOOLS,
    Mode.MINIMAX_JSON: Mode.MD_JSON,
}


def reset_deprecated_mode_warnings() -> None:
    """Reset deprecation warning tracking."""
    global _deprecated_modes_warned
    _deprecated_modes_warned = set()
