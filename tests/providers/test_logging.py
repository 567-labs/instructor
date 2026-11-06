import logging
from instructor.auto_client import from_provider


def test_from_provider_logging(caplog):
    caplog.set_level(logging.INFO)
    from_provider("ollama/llama3.2")
    assert any(
        "Initializing ollama provider" in record.getMessage()
        for record in caplog.records
    )
    assert any("Client initialized" in record.getMessage() for record in caplog.records)
