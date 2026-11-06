import json
import pytest
from instructor.utils import (
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)


def test_extract_json_from_codeblock():
    example = """
    Here is a response

    ```json
    {
        "key": "value"
    }
    ```
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {"key": "value"}


def test_extract_json_from_codeblock_no_end():
    example = """
    Here is a response

    ```json
    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}]
    }
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}],
    }


def test_stream_json():
    text = """here is the json for you!

    ```json
    , here
    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}]
    }
    ```

    What do you think?
    """

    def batch_strings(chunks, n=2):
        batch = ""
        for chunk in chunks:
            for char in chunk:
                batch += char
                if len(batch) == n:
                    yield batch
                    batch = ""
        if batch:  # Yield any remaining characters in the last batch
            yield batch

    result = json.loads(
        "".join(list(extract_json_from_stream(batch_strings(text, n=3))))
    )
    assert result == {"key": "value", "another_key": [{"key": {"key": "value"}}]}


@pytest.mark.asyncio
async def test_stream_json_async():
    text = """here is the json for you!

    ```json
    , here
    {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}, {"key": "value"}]
    }
    ```

    What do you think?
    """

    async def batch_strings_async(chunks, n=2):
        batch = ""
        for chunk in chunks:
            for char in chunk:
                batch += char
                if len(batch) == n:
                    yield batch
                    batch = ""
        if batch:  # Yield any remaining characters in the last batch
            yield batch

    result = json.loads(
        "".join(
            [
                chunk
                async for chunk in extract_json_from_stream_async(
                    batch_strings_async(text, n=3)
                )
            ]
        )
    )
    assert result == {
        "key": "value",
        "another_key": [{"key": {"key": "value"}}, {"key": "value"}],
    }
