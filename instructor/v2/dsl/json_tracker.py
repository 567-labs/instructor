"""
JSON Completeness Tracker for Partial Streaming.

Tracks which parts of accumulated JSON are "closed" (complete) vs "open" (incomplete).
Uses jiter for parsing and a simple heuristic: if a value has a next sibling,
it must be complete (because jiter had to finish parsing it to find the next one).
"""

from __future__ import annotations

from typing import Any

from jiter import from_json

_CONTAINER_ENDS = {"{": "}", "[": "]"}


def is_json_complete(json_str: str) -> bool:
    """
    Check if a JSON string represents a complete structure.

    Uses jiter in strict mode - parsing fails if JSON is incomplete.
    """
    if not json_str or not json_str.strip():
        return False
    try:
        from_json(json_str.encode())  # No partial_mode = strict parsing
        return True
    except ValueError:
        return False


class JsonCompleteness:
    """
    Track completeness of JSON structures during streaming.

    Uses a simple heuristic: if a value has a next sibling in the parsed
    structure, it must be complete. For the last sibling, we don't know
    until the parent completes - but that's fine because parent validation
    will cover it.

    Example:
        tracker = JsonCompleteness()

        # Incomplete - missing closing brace
        tracker.analyze('{"name": "Alice", "address": {"city": "NY')
        tracker.is_path_complete("")  # False - root incomplete
        tracker.is_path_complete("name")  # True - has next sibling "address"
        tracker.is_path_complete("address")  # False - last sibling, unknown

        # Complete
        tracker.analyze('{"name": "Alice"}')
        tracker.is_path_complete("")  # True - root complete
    """

    def __init__(self) -> None:
        self._complete_paths: set[str] = set()
        self._complete_subtrees: dict[str, list[Any]] = {}
        self._shared_json: str | None = None

    def analyze(self, json_str: str) -> None:
        """Analyze a JSON string and determine completeness of each path."""
        self._analyze(json_str)

    def _analyze(self, json_str: str, parsed_partial: Any = None) -> None:
        """Optionally reuse a trailing-strings parse of this exact input."""
        self._shared_json = None
        self._complete_paths.clear()
        self._complete_subtrees.clear()

        if not json_str or not (trimmed := json_str.strip()):
            return

        # A complete container must end with its closing delimiter. Strip
        # only for the check so jiter still decides which whitespace is valid.
        try:
            raw = json_str.encode()
        except UnicodeEncodeError:
            return
        closing = _CONTAINER_ENDS.get(trimmed[0])
        if closing is None or trimmed.endswith(closing):
            try:
                parsed = from_json(raw)
            except ValueError:
                pass
            else:
                self._mark_complete(parsed, "")
                return

        # Compute sibling paths before validators see the tree. Most snapshots
        # need only those paths, so defer parsing a private tree until expansion.
        if parsed_partial is not None:
            self._shared_json = json_str
            self._check_siblings(parsed_partial, "")
            return

        try:
            parsed = from_json(raw, partial_mode="trailing-strings")
        except ValueError:
            return

        self._check_siblings(parsed, "")

    def _mark_complete(self, data: Any, path: str) -> None:
        self._complete_paths.add(path)
        if isinstance(data, (dict, list)) and data:
            # Store roots first. Most callers validate the entire completed
            # model, so they never ask about its descendant paths.
            self._complete_subtrees.setdefault(path, []).append(data)

    def _expand(self, path: str) -> None:
        if self._shared_json is not None:
            # A validator may have mutated a shared subtree. Rebuild once from
            # the original JSON before inspecting any descendants.
            self.analyze(self._shared_json)
        for data in self._complete_subtrees.pop(path, ()):
            self._mark_all(data, path)

    def _mark_all(self, data: Any, path: str) -> None:
        """Recursively mark path and all children as complete."""
        self._complete_paths.add(path)
        if isinstance(data, dict):
            for key, value in data.items():
                child_path = f"{path}.{key}" if path else str(key)
                self._mark_all(value, child_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._mark_all(item, f"{path}[{i}]")

    def _check_siblings(self, data: Any, path: str) -> None:
        """
        Check completeness using sibling heuristic.

        If a value has a next sibling, it's complete (jiter had to finish
        parsing it to find the next sibling). Last sibling is unknown.
        """
        if isinstance(data, dict):
            keys = list(data.keys())
            for i, key in enumerate(keys):
                child_path = f"{path}.{key}" if path else str(key)
                if i < len(keys) - 1:
                    # Has next sibling → complete
                    self._mark_complete(data[key], child_path)
                else:
                    # Last sibling → recurse to check children
                    self._check_siblings(data[key], child_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                child_path = f"{path}[{i}]"
                if i < len(data) - 1:
                    # Has next sibling → complete
                    self._mark_complete(item, child_path)
                else:
                    # Last sibling → recurse
                    self._check_siblings(item, child_path)

    def is_path_complete(self, path: str) -> bool:
        """
        Check if the sub-structure at the given path is complete.

        Args:
            path: Dot-separated path (e.g., "user.address.city", "items[0]")
                  Use "" for root object.

        Returns:
            True if the structure at path is complete (closed), False otherwise.
        """
        if path in self._complete_paths:
            return True
        # Expand only ancestors that can produce this flattened path. An
        # empty key shares the root path under the legacy path representation.
        if "" in self._complete_subtrees:
            self._expand("")
        end = len(path)
        while (end := max(path.rfind(".", 0, end), path.rfind("[", 0, end))) > 0:
            if path[:end] in self._complete_subtrees:
                self._expand(path[:end])
        return path in self._complete_paths

    def get_complete_paths(self) -> set[str]:
        """Return all paths that are complete."""
        for path in list(self._complete_subtrees):
            self._expand(path)
        return self._complete_paths.copy()

    def is_root_complete(self) -> bool:
        """Check if the root JSON structure is complete."""
        return "" in self._complete_paths
