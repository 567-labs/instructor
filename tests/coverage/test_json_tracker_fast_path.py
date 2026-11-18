from typing import Any

import pytest

from instructor.v2.dsl import json_tracker


@pytest.mark.parametrize(
    ("value", "strict_attempted", "complete_paths"),
    [
        (' \t{"a": 1, "b": "élève  ', False, {"a"}),
        (" \n[1,2 \r\t", False, {"[0]"}),
        ('{"a": 1, "b": {}', True, {"a"}),
        ("[[0], [1]", True, {"[0]", "[0][0]"}),
        (' \t[1, {"b": 2}] \t\n', True, {"", "[0]", "[1]", "[1].b"}),
        ("\u2003[1,2]", True, set()),
        ("[1,2]\u2003", True, {"[0]"}),
        ('{"a": 1, "b": "\ud800', False, set()),
        (' "é"\t', True, {""}),
        ("[1,2}", False, {"[0]"}),
    ],
)
def test_tracker_only_skips_provably_incomplete_containers(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    strict_attempted: bool,
    complete_paths: set[str],
) -> None:
    original_parse = json_tracker.from_json
    calls: list[dict[str, Any]] = []

    def capture_parse(data: bytes, **kwargs: Any) -> Any:
        assert data == value.encode()
        calls.append(kwargs)
        return original_parse(data, **kwargs)

    monkeypatch.setattr(json_tracker, "from_json", capture_parse)
    tracker = json_tracker.JsonCompleteness()
    tracker.analyze(value)

    assert tracker.is_root_complete() is ("" in complete_paths)
    assert tracker.get_complete_paths() == complete_paths
    assert ({} in calls) is strict_attempted
