from typing import Any
import mkdocs.plugins
from pymdownx import highlight  # type: ignore


@mkdocs.plugins.event_priority(0)
# pylint: disable=unused-argument
def on_startup(command: str, dirty: bool) -> None:  # noqa: ARG001
    """Monkey patch Highlight extension to hide lines in code blocks."""
    original = highlight.Highlight.highlight  # type: ignore

    def patched(self: Any, src: str, *args: Any, **kwargs: Any) -> Any:
        lines = src.splitlines(keepends=True)

        final_lines = []

        remove_lines = False
        for line in lines:
            if line.strip() == "# <%hide%>":
                remove_lines = not remove_lines
            elif not remove_lines:
                final_lines.append(line)

        return original(self, "".join(final_lines), *args, **kwargs)

    highlight.Highlight.highlight = patched
