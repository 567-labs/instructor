import os
import pytest

if not os.getenv("MINIMAX_API_KEY"):
    pytest.skip("MINIMAX_API_KEY environment variable not set", allow_module_level=True)
