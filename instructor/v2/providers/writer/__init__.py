"""v2 Writer provider."""

try:
    from instructor.v2.providers.writer.client import from_writer
except ImportError:
    from_writer = None

__all__ = ["from_writer"]
