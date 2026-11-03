import pytest
from tenacity import AsyncRetrying, Retrying

from instructor.v2.core.errors import ConfigurationError
from instructor.v2.core.retry import _validate_max_retries


def test_retry_counts_and_policies_are_accepted():
    _validate_max_retries(0, Retrying)
    _validate_max_retries(3, Retrying)
    _validate_max_retries(Retrying(), Retrying)
    _validate_max_retries(AsyncRetrying(), AsyncRetrying)


@pytest.mark.parametrize("bad", [1.5, "3", None, [2]])
def test_a_value_that_is_not_a_count_or_a_policy_is_named(bad):
    with pytest.raises(ConfigurationError, match="max_retries must be a non-negative int"):
        _validate_max_retries(bad, Retrying)


def test_the_sync_path_rejects_an_async_policy():
    with pytest.raises(ConfigurationError, match="Retrying instance"):
        _validate_max_retries(AsyncRetrying(), Retrying)


def test_a_negative_count_is_named():
    with pytest.raises(ConfigurationError, match="must not be negative"):
        _validate_max_retries(-1, Retrying)
