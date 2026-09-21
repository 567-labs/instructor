"""Behavior tests for the Mistral batch provider.

Covers both Mistral SDK export layouts, the request/response contracts and the
job-status normalization, without contacting the API.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.batch import BatchError, BatchProcessor, BatchSuccess
from instructor.batch.models import BatchJobInfo, BatchStatus
from instructor.batch.providers import mistral as mistral_module
from instructor.batch.providers.mistral import MistralProvider

pytestmark = pytest.mark.unit


CREATED_AT = 1735784800  # 2025-01-02T02:26:40Z


class Person(BaseModel):
    name: str
    age: int


class Team(BaseModel):
    members: list[Person]


class FakeJob:
    """Stand-in for BatchJobOut (SDK 1.x) and BatchJob (SDK 2.x)"""

    def __init__(self, data: dict[str, Any]) -> None:
        self.__dict__.update(data)

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)


def make_job(
    job_id: str = "batch_123",
    status: str = "SUCCESS",
    succeeded: int = 2,
    failed: int = 0,
    output_file: str | None = "file_out",
    **overrides: Any,
) -> FakeJob:
    data: dict[str, Any] = {
        "id": job_id,
        "status": status,
        "created_at": CREATED_AT,
        "started_at": CREATED_AT + 5,
        "completed_at": CREATED_AT + 65,
        "input_files": ["file_in"],
        "output_file": output_file,
        "error_file": None,
        "errors": [],
        "total_requests": succeeded + failed,
        "completed_requests": succeeded + failed,
        "succeeded_requests": succeeded,
        "failed_requests": failed,
        "model": "mistral-small-latest",
        "endpoint": "/v1/chat/completions",
        "metadata": {"description": "Instructor batch job"},
    }
    data.update(overrides)
    return FakeJob(data)


class RecordingJobs:
    def __init__(
        self,
        job: FakeJob | None = None,
        listed: tuple[FakeJob, ...] = (),
        supports_delete: bool = True,
        failures: dict[str, Exception] | None = None,
    ) -> None:
        self.job = job or make_job()
        self.listed = listed
        self.failures = failures or {}
        self.created_kwargs: dict[str, Any] | None = None
        self.requested_ids: list[str] = []
        self.cancelled_ids: list[str] = []
        self.deleted_ids: list[str] = []
        self.list_kwargs: list[dict[str, Any]] = []
        if supports_delete:
            self.delete = self._delete

    def _raise_if_needed(self, operation: str) -> None:
        if operation in self.failures:
            raise self.failures[operation]

    def create(self, **kwargs: Any) -> FakeJob:
        self._raise_if_needed("create")
        self.created_kwargs = kwargs
        return self.job

    def get(self, *, job_id: str) -> FakeJob:
        self._raise_if_needed("get")
        self.requested_ids.append(job_id)
        return self.job

    def cancel(self, *, job_id: str) -> FakeJob:
        self._raise_if_needed("cancel")
        self.cancelled_ids.append(job_id)
        return make_job(job_id=job_id, status="CANCELLATION_REQUESTED")

    def _delete(self, *, job_id: str) -> dict[str, Any]:
        self._raise_if_needed("delete")
        self.deleted_ids.append(job_id)
        return {"id": job_id, "object": "batch", "deleted": True}

    def list(self, **kwargs: Any) -> SimpleNamespace:
        self._raise_if_needed("list")
        self.list_kwargs.append(kwargs)
        return SimpleNamespace(data=list(self.listed))


class RecordingFiles:
    def __init__(
        self, payload: bytes = b"", failures: dict[str, Exception] | None = None
    ) -> None:
        self.payload = payload
        self.failures = failures or {}
        self.uploads: list[dict[str, Any]] = []
        self.downloaded_ids: list[str] = []

    def upload(self, *, file: dict[str, Any], purpose: str) -> SimpleNamespace:
        if "upload" in self.failures:
            raise self.failures["upload"]
        self.uploads.append(
            {
                "file_name": file["file_name"],
                "content": file["content"].read(),
                "purpose": purpose,
            }
        )
        return SimpleNamespace(id="file_in")

    def download(self, *, file_id: str) -> io.BytesIO:
        if "download" in self.failures:
            raise self.failures["download"]
        self.downloaded_ids.append(file_id)
        return io.BytesIO(self.payload)


class FakeClient:
    def __init__(
        self,
        jobs: RecordingJobs | None = None,
        files: RecordingFiles | None = None,
    ) -> None:
        self.api_key: str | None = None
        self.jobs = jobs or RecordingJobs()
        self.files = files or RecordingFiles()
        self.batch = SimpleNamespace(jobs=self.jobs)


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


def install_client(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> FakeClient:
    def factory(**kwargs: Any) -> FakeClient:
        client.api_key = kwargs.get("api_key")
        return client

    monkeypatch.setattr(mistral_module, "_load_mistral_client_type", lambda: factory)
    return client


def write_requests(tmp_path: Path, *lines: dict[str, Any]) -> str:
    path = tmp_path / "batch_requests.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return str(path)


# --- SDK layout resolution -------------------------------------------------


class FakeClientType:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key


def fake_module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.mark.parametrize(
    "exporting, non_exporting",
    [
        ("mistralai.client", "mistralai"),  # SDK 2.x
        ("mistralai", "mistralai.client"),  # SDK 1.x
    ],
)
def test_client_type_resolves_both_sdk_layouts(
    monkeypatch: pytest.MonkeyPatch, exporting: str, non_exporting: str
) -> None:
    monkeypatch.setitem(
        sys.modules, exporting, fake_module(exporting, Mistral=FakeClientType)
    )
    monkeypatch.setitem(sys.modules, non_exporting, fake_module(non_exporting))

    assert mistral_module._load_mistral_client_type() is FakeClientType


def test_client_type_skips_unimportable_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry makes import_module raise ImportError
    monkeypatch.setitem(sys.modules, "mistralai.client", None)
    monkeypatch.setitem(
        sys.modules, "mistralai", fake_module("mistralai", Mistral=FakeClientType)
    )

    assert mistral_module._load_mistral_client_type() is FakeClientType


def test_client_type_is_none_when_no_layout_exports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules, "mistralai.client", fake_module("mistralai.client")
    )
    monkeypatch.setitem(
        sys.modules, "mistralai", fake_module("mistralai", Mistral="not-a-type")
    )

    assert mistral_module._load_mistral_client_type() is None


def test_client_type_resolves_the_installed_sdk() -> None:
    pytest.importorskip("mistralai")

    client_type = mistral_module._load_mistral_client_type()

    assert client_type is not None
    assert client_type.__name__ == "Mistral"


def test_missing_sdk_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mistral_module, "_load_mistral_client_type", lambda: None)

    with pytest.raises(RuntimeError, match="require the mistralai package"):
        MistralProvider().get_status("batch_123")


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(monkeypatch, FakeClient())
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        MistralProvider().get_status("batch_123")


def test_api_key_is_passed_to_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = install_client(monkeypatch, FakeClient())

    MistralProvider().get_status("batch_123")

    assert client.api_key == "test-key"


# --- submit ----------------------------------------------------------------


def test_submit_uploads_file_and_creates_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = install_client(monkeypatch, FakeClient())
    request = {"custom_id": "request-0", "body": {"messages": []}}
    file_path = write_requests(tmp_path, request)

    batch_id = MistralProvider().submit_batch(
        file_path, metadata={"description": "job"}, model="mistral-small-latest"
    )

    assert batch_id == "batch_123"
    assert client.files.uploads == [
        {
            "file_name": "batch.jsonl",
            "content": json.dumps(request).encode() + b"\n",
            "purpose": "batch",
        }
    ]
    assert client.jobs.created_kwargs == {
        "input_files": ["file_in"],
        "model": "mistral-small-latest",
        "endpoint": "/v1/chat/completions",
        "metadata": {"description": "job"},
    }


def test_submit_accepts_buffer_and_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = install_client(monkeypatch, FakeClient())
    buffer = io.BytesIO(b'{"custom_id": "request-0"}\n')
    buffer.seek(len(buffer.getvalue()))

    MistralProvider().submit_batch(
        buffer, model="mistral-large-latest", endpoint="/v1/embeddings"
    )

    # The buffer is rewound before upload
    assert client.files.uploads[0]["content"] == b'{"custom_id": "request-0"}\n'
    assert client.jobs.created_kwargs is not None
    assert client.jobs.created_kwargs["endpoint"] == "/v1/embeddings"
    assert client.jobs.created_kwargs["metadata"] == {}


def test_submit_requires_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(monkeypatch, FakeClient())

    with pytest.raises(ValueError, match="require a model"):
        MistralProvider().submit_batch("batch_requests.jsonl")


def test_submit_rejects_unsupported_input(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(monkeypatch, FakeClient())

    with pytest.raises(ValueError, match="Unsupported file_path_or_buffer type"):
        MistralProvider().submit_batch(object(), model="mistral-small-latest")  # type: ignore[arg-type]


def test_submit_wraps_sdk_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(
        monkeypatch,
        FakeClient(jobs=RecordingJobs(failures={"create": RuntimeError("boom")})),
    )
    buffer = io.BytesIO(b"{}\n")

    with pytest.raises(RuntimeError, match="Failed to submit Mistral batch: boom"):
        MistralProvider().submit_batch(buffer, model="mistral-small-latest")


# --- status, results and lifecycle -----------------------------------------


def test_get_status_normalizes_request_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(
        monkeypatch,
        FakeClient(
            jobs=RecordingJobs(job=make_job(status="RUNNING", succeeded=1, failed=1))
        ),
    )

    status = MistralProvider().get_status("batch_123")

    assert status == {
        "id": "batch_123",
        "status": "RUNNING",
        "created_at": CREATED_AT,
        "request_counts": {"total": 2, "succeeded": 1, "failed": 1},
    }


def test_get_status_wraps_sdk_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(
        monkeypatch,
        FakeClient(jobs=RecordingJobs(failures={"get": RuntimeError("nope")})),
    )

    with pytest.raises(RuntimeError, match="Failed to get Mistral batch status: nope"):
        MistralProvider().get_status("batch_123")


def test_retrieve_results_downloads_the_output_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"custom_id": "request-0"}\n'
    client = install_client(
        monkeypatch, FakeClient(files=RecordingFiles(payload=payload))
    )

    assert MistralProvider().retrieve_results("batch_123") == payload.decode()
    assert client.files.downloaded_ids == ["file_out"]


def test_download_results_writes_the_output_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b'{"custom_id": "request-0"}\n'
    install_client(monkeypatch, FakeClient(files=RecordingFiles(payload=payload)))
    destination = tmp_path / "results.jsonl"

    MistralProvider().download_results("batch_123", str(destination))

    assert destination.read_bytes() == payload


@pytest.mark.parametrize(
    "job, message",
    [
        (make_job(status="RUNNING"), "Batch not completed, status: RUNNING"),
        (make_job(status="FAILED"), "Batch not completed, status: FAILED"),
        (make_job(succeeded=0, failed=3), "All 3 batch requests failed"),
        (make_job(output_file=None), "Batch has no output file ID available"),
    ],
)
def test_results_refuse_jobs_without_usable_output(
    monkeypatch: pytest.MonkeyPatch, job: FakeJob, message: str
) -> None:
    install_client(monkeypatch, FakeClient(jobs=RecordingJobs(job=job)))

    with pytest.raises(RuntimeError, match=message):
        MistralProvider().retrieve_results("batch_123")


def test_cancel_batch_returns_the_updated_job(monkeypatch: pytest.MonkeyPatch) -> None:
    client = install_client(monkeypatch, FakeClient())

    cancelled = MistralProvider().cancel_batch("batch_123")

    assert client.jobs.cancelled_ids == ["batch_123"]
    assert cancelled["status"] == "CANCELLATION_REQUESTED"


def test_delete_batch_uses_the_sdk2_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    client = install_client(monkeypatch, FakeClient())

    assert MistralProvider().delete_batch("batch_123") == {
        "id": "batch_123",
        "object": "batch",
        "deleted": True,
    }
    assert client.jobs.deleted_ids == ["batch_123"]


def test_delete_batch_reports_unsupported_on_sdk1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_client(monkeypatch, FakeClient(jobs=RecordingJobs(supports_delete=False)))

    with pytest.raises(NotImplementedError, match="requires mistralai>=2.0.0"):
        MistralProvider().delete_batch("batch_123")


def test_list_batches_normalizes_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    listed = tuple(make_job(job_id=f"batch_{index}") for index in range(5))
    client = install_client(monkeypatch, FakeClient(jobs=RecordingJobs(listed=listed)))

    jobs = MistralProvider().list_batches(limit=2)

    assert client.jobs.list_kwargs == [{"page_size": 2}]
    assert [job.id for job in jobs] == ["batch_0", "batch_1"]
    assert all(job.provider == "mistral" for job in jobs)


def test_list_batches_handles_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    install_client(monkeypatch, FakeClient(jobs=RecordingJobs(listed=())))

    assert MistralProvider().list_batches() == []


# --- job normalization -----------------------------------------------------


@pytest.mark.parametrize(
    "raw_status, expected",
    [
        ("QUEUED", BatchStatus.PENDING),
        ("RUNNING", BatchStatus.PROCESSING),
        ("SUCCESS", BatchStatus.COMPLETED),
        ("FAILED", BatchStatus.FAILED),
        ("TIMEOUT_EXCEEDED", BatchStatus.EXPIRED),
        ("CANCELLATION_REQUESTED", BatchStatus.CANCELLED),
        ("CANCELLED", BatchStatus.CANCELLED),
        ("SOMETHING_NEW", BatchStatus.PENDING),
    ],
)
def test_from_mistral_maps_every_status(raw_status: str, expected: BatchStatus) -> None:
    info = BatchJobInfo.from_mistral(make_job(status=raw_status).model_dump())

    assert info.status is expected
    assert info.raw_status == raw_status


def test_from_mistral_normalizes_job_fields() -> None:
    info = BatchJobInfo.from_mistral(make_job().model_dump())

    assert info.provider == "mistral"
    assert info.model == "mistral-small-latest"
    assert info.endpoint == "/v1/chat/completions"
    assert info.files.input_file_id == "file_in"
    assert info.files.output_file_id == "file_out"
    assert info.request_counts.total == 2
    assert info.request_counts.succeeded == 2
    assert info.request_counts.failed == 0
    assert info.timestamps.created_at is not None
    assert info.timestamps.created_at.timestamp() == CREATED_AT
    assert info.timestamps.completed_at is not None
    assert info.timestamps.completed_at.timestamp() == CREATED_AT + 65
    assert info.error is None


def test_from_mistral_surfaces_job_errors_and_missing_fields() -> None:
    info = BatchJobInfo.from_mistral(
        {
            "id": "batch_123",
            "status": "FAILED",
            "errors": [{"message": "model not found", "count": 2}],
            "input_files": [],
            "metadata": None,
        }
    )

    assert info.error is not None
    assert info.error.error_message == "model not found"
    assert info.error.error_type == "mistral_error"
    assert info.files.input_file_id is None
    assert info.metadata == {}
    assert info.timestamps.created_at is None


@pytest.mark.parametrize(
    "created_at, expected",
    [
        ("2025-01-02T02:26:40Z", CREATED_AT),
        (CREATED_AT, CREATED_AT),
        ("not-a-date", None),
        (None, None),
    ],
)
def test_from_mistral_parses_timestamp_variants(
    created_at: Any, expected: int | None
) -> None:
    info = BatchJobInfo.from_mistral(
        {"id": "batch_123", "status": "QUEUED", "created_at": created_at}
    )

    if expected is None:
        assert info.timestamps.created_at is None
    else:
        assert info.timestamps.created_at is not None
        assert info.timestamps.created_at.timestamp() == expected


# --- request and response contracts ----------------------------------------


def test_request_format_carries_schema_and_no_model() -> None:
    processor: BatchProcessor[Person] = BatchProcessor(
        "mistral/mistral-small-latest", Person
    )
    buffer = processor.create_batch_from_messages(
        [[{"role": "user", "content": "Ada is 36"}]], max_tokens=90, temperature=0.2
    )

    assert isinstance(buffer, io.BytesIO)
    request = json.loads(buffer.read().decode())

    assert request["custom_id"] == "request-0"
    assert set(request) == {"custom_id", "body"}
    body = request["body"]
    assert body["messages"] == [{"role": "user", "content": "Ada is 36"}]
    assert body["max_tokens"] == 90
    assert body["temperature"] == 0.2
    # Mistral sets the model on the job, not per request
    assert "model" not in body
    json_schema = body["response_format"]["json_schema"]
    assert body["response_format"]["type"] == "json_schema"
    assert json_schema["name"] == "Person"
    assert json_schema["strict"] is True
    assert json_schema["schema"]["additionalProperties"] is False
    assert set(json_schema["schema"]["properties"]) == {"name", "age"}


def test_request_format_applies_strict_schema_to_nested_models() -> None:
    processor: BatchProcessor[Team] = BatchProcessor(
        "mistral/mistral-small-latest", Team
    )
    buffer = processor.create_batch_from_messages(
        [[{"role": "user", "content": "Ada is 36"}]]
    )

    assert isinstance(buffer, io.BytesIO)
    schema = json.loads(buffer.read().decode())["body"]["response_format"][
        "json_schema"
    ]["schema"]

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Person"]["additionalProperties"] is False


def test_submit_batch_defaults_the_model_from_the_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = install_client(monkeypatch, FakeClient())
    processor: BatchProcessor[Person] = BatchProcessor(
        "mistral/mistral-small-latest", Person
    )

    processor.submit_batch(io.BytesIO(b"{}\n"))

    assert client.jobs.created_kwargs is not None
    assert client.jobs.created_kwargs["model"] == "mistral-small-latest"


def test_submit_batch_keeps_an_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = install_client(monkeypatch, FakeClient())
    processor: BatchProcessor[Person] = BatchProcessor(
        "mistral/mistral-small-latest", Person
    )

    processor.submit_batch(io.BytesIO(b"{}\n"), model="mistral-large-latest")

    assert client.jobs.created_kwargs is not None
    assert client.jobs.created_kwargs["model"] == "mistral-large-latest"


def mistral_result_line(custom_id: str, content: str) -> str:
    return json.dumps(
        {
            "id": f"result_{custom_id}",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {
                    "id": "cmpl_1",
                    "object": "chat.completion",
                    "model": "mistral-small-latest",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                },
            },
            "error": None,
        }
    )


def test_parse_results_reads_the_mistral_response_shape() -> None:
    processor: BatchProcessor[Person] = BatchProcessor(
        "mistral/mistral-small-latest", Person
    )

    results = processor.parse_results(
        "\n".join(
            [
                mistral_result_line("request-0", '{"name": "Ada", "age": 36}'),
                mistral_result_line("request-1", "not json"),
                json.dumps(
                    {
                        "custom_id": "request-2",
                        "response": None,
                        "error": {
                            "message": "model not found",
                            "type": "invalid_request_error",
                        },
                    }
                ),
                json.dumps({"custom_id": "request-3", "error": "rate limited"}),
            ]
        )
    )

    assert isinstance(results[0], BatchSuccess)
    assert results[0].result == Person(name="Ada", age=36)

    assert isinstance(results[1], BatchError)
    assert results[1].custom_id == "request-1"

    assert isinstance(results[2], BatchError)
    assert results[2].error_type == "invalid_request_error"
    assert results[2].error_message == "model not found"

    assert isinstance(results[3], BatchError)
    assert results[3].error_type == "mistral_error"
    assert results[3].error_message == "rate limited"
