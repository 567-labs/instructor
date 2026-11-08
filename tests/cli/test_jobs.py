"""Tests for instructor.cli.jobs."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("pages", "limit", "expected"),
    [
        ([], 2, []),
        ([["job-only"]], 2, ["job-only"]),
        ([["job-new"], ["job-old", "job-older"]], 2, ["job-new", "job-old"]),
    ],
)
def test_get_jobs_consumes_cursor(monkeypatch, pages, limit, expected) -> None:
    from instructor.cli import jobs

    page_items = [[SimpleNamespace(id=job_id) for job_id in page] for page in pages]
    expected_items = [SimpleNamespace(id=job_id) for job_id in expected]
    yielded = []
    requested_limits = []

    class FakeCursor:
        def __iter__(self):
            for page in page_items:
                for job in page:
                    yielded.append(job)
                    yield job

    class FakeJobs:
        def list(self, *, limit):
            requested_limits.append(limit)
            return FakeCursor()

    monkeypatch.setattr(
        jobs,
        "client",
        SimpleNamespace(fine_tuning=SimpleNamespace(jobs=FakeJobs())),
    )

    assert jobs.get_jobs(limit=limit) == expected_items
    assert [job.id for job in yielded] == expected
    assert requested_limits == [limit]
