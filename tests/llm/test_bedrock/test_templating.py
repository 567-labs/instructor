"""Jinja ``context=`` templating on the Bedrock provider.

The Bedrock request handler converts messages to Converse shape before ``handle_templating``
runs: the system prompt moves to ``system=[{"text": ...}]`` and content parts become
``{"text": ...}`` blocks with no ``type``. Templating has to recognise both shapes, or every
``{{ variable }}`` reaches the model verbatim.
"""

from __future__ import annotations

from instructor import Mode
from instructor.v2.core.templating import handle_templating
from instructor.v2.providers.bedrock.handlers import (
    _prepare_bedrock_converse_kwargs_internal,
)

CONTEXT = {"topic": "airports", "text": "Runway closed for snow."}


def test_converse_system_and_text_blocks_are_templated():
    kwargs = {
        "modelId": "anthropic.claude-3-5-sonnet",
        "system": [{"text": "You classify {{ topic }} reports."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "Report: {{ text }}"},
                    {"cachePoint": {"type": "default"}},
                ],
            }
        ],
    }

    result = handle_templating(kwargs, Mode.BEDROCK_JSON, context=CONTEXT)

    assert result["system"] == [{"text": "You classify airports reports."}]
    assert result["messages"][0]["content"] == [
        {"text": "Report: Runway closed for snow."},
        {"cachePoint": {"type": "default"}},
    ]


def test_templating_does_not_mutate_the_caller_kwargs():
    system = [{"text": "{{ topic }}"}]
    messages = [{"role": "user", "content": [{"text": "{{ text }}"}]}]
    kwargs = {"system": system, "messages": messages}

    handle_templating(kwargs, Mode.BEDROCK_TOOLS, context=CONTEXT)

    assert system == [{"text": "{{ topic }}"}]
    assert messages == [{"role": "user", "content": [{"text": "{{ text }}"}]}]


def test_string_content_is_templated_before_conversion():
    kwargs = {"messages": [{"role": "user", "content": "Report: {{ text }}"}]}

    result = handle_templating(kwargs, Mode.BEDROCK_JSON, context=CONTEXT)

    assert result["messages"][0]["content"] == "Report: Runway closed for snow."


def test_blocks_without_text_pass_through():
    image = {"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}
    document = {
        "document": {"format": "pdf", "name": "report", "source": {"bytes": b"%PDF"}}
    }
    kwargs = {
        "system": [{"text": "{{ topic }}"}, {"cachePoint": {"type": "default"}}],
        "messages": [
            {"role": "user", "content": [image, document, {"text": "{{ text }}"}]}
        ],
    }

    result = handle_templating(kwargs, Mode.BEDROCK_JSON, context=CONTEXT)

    assert result["system"] == [
        {"text": "airports"},
        {"cachePoint": {"type": "default"}},
    ]
    assert result["messages"][0]["content"] == [
        image,
        document,
        {"text": "Runway closed for snow."},
    ]


def test_templating_survives_the_bedrock_kwargs_preparation():
    """The create() order: the request handler converts to Converse shape first, templating second."""
    call_kwargs = {
        "model": "anthropic.claude-3-5-sonnet",
        "messages": [
            {"role": "system", "content": "You classify {{ topic }} reports."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Report: {{ text }}"},
                    {"cachePoint": {"type": "default"}},
                ],
            },
        ],
    }

    prepared = _prepare_bedrock_converse_kwargs_internal(call_kwargs)
    result = handle_templating(prepared, Mode.BEDROCK_JSON, context=CONTEXT)

    assert result["system"] == [{"text": "You classify airports reports."}]
    assert result["messages"] == [
        {
            "role": "user",
            "content": [
                {"text": "Report: Runway closed for snow."},
                {"cachePoint": {"type": "default"}},
            ],
        }
    ]


def test_other_providers_keep_their_templating():
    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Report: {{ text }}"}],
            },
        ]
    }

    result = handle_templating(kwargs, Mode.ANTHROPIC_TOOLS, context=CONTEXT)

    assert result["messages"][0]["content"] == [
        {"type": "text", "text": "Report: Runway closed for snow."}
    ]
