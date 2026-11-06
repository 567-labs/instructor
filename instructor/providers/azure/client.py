"""Compatibility shim — backed by v2 Azure provider."""

from instructor.v2.providers.azure.client import from_azure

__all__ = ["from_azure"]
