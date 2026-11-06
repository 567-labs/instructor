"""Backwards compatibility module for instructor.function_calls.

This module re-exports everything from instructor.processing.function_calls
for backwards compatibility.
"""

# Re-export everything from the actual function_calls module
from .processing.function_calls import *  # noqa: F401, F403
