"""Textual TUI that runs the FlakeHunter pipeline live.

Three panels, per the project's design intent:

* a **test list** (left) showing each target's live status as it advances
  ``detecting -> flaky -> classifying -> fixing -> verifying -> verified``;
* a **detail** panel (right) for the selected test: its root cause and the
  proposed fix's diff + rationale;
* a **scoreboard** (bottom): before/after failure rate, Wilson CI, verdict.

The pipeline is synchronous and subprocess-heavy, so it runs on a *thread*
worker (never the event loop). Status updates cross back to the UI thread via
``post_message`` (thread-safe); widgets are only ever touched in the message
handler, which runs on the UI thread.
"""

from __future__ import annotations

from typing import Callable, Protocol

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import DataTable, Footer, Header, Static

from flakehunter.orchestrator import PipelineEvent

class SupportsRun(Protocol):
    """Anything with a blocking ``run(test_ids)`` — the real Pipeline or a fake."""

    def run(self, test_ids: list[str]) -> object: ...


# Factory the app calls to build something with ``.run(test_ids)``; it is handed
# the app's thread-safe emit callback so pipeline events flow back to the UI.
# Kept structural (a Protocol, not ``Pipeline``) so tests can inject a fake.
PipelineFactory = Callable[[Callable[[PipelineEvent], None]], SupportsRun]


def _short(test_id: str) -> str:
    """A compact label for the test-list column."""
    return test_id.split("::", 1)[-1]


def _status_label(event: PipelineEvent) -> str:
    """Human-readable status for the test-list row, from the latest event."""
    phase = event.phase
    if phase == "flaky" and event.verdict is not None:
        return f"flaky p={event.verdict.failure_rate:.0%}"
    if phase == "classified" and event.cause is not None:
        return f"cause: {event.cause.category}"
    if phase == "rejected" and event.result is not None:
        return f"rejected ({event.result.verdict.removeprefix('rejected_')})"
    if phase == "error":
        return "error"
    return {
        "detecting": "detecting...",
        "not_flaky": "not flaky",
        "classifying": "classifying...",
        "fixing": "fixing...",
        "fixed": "fix ready",
        "verifying": "verifying...",
        "verified": "verified",
        "suggest_only": "suggested (manual)",
    }.get(phase, phase)


class PipelineUpdate(Message):
    """Carries one ``PipelineEvent`` from the worker thread to the UI thread."""

    def __init__(self, event: PipelineEvent) -> None:
        self.event = event
        super().__init__()


class FlakeHunterApp(App[None]):
    """Live 3-panel view over a pipeline run."""

    CSS = """
    #tests { width: 45%; border: solid $accent; }
    #detail { width: 55%; border: solid $accent; padding: 0 1; }
    #scoreboard { height: auto; border: solid $secondary; padding: 0 1; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        *,
        test_ids: list[str],
        pipeline_factory: PipelineFactory,
    ) -> None:
        super().__init__()
        self._test_ids = list(test_ids)
        self._pipeline_factory = pipeline_factory
        # Latest event seen per test, for re-rendering the detail/scoreboard
        # when the selection changes.
        self._latest: dict[str, PipelineEvent] = {}
        self._selected: str | None = test_ids[0] if test_ids else None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="tests")
            yield Static("Waiting for the pipeline...", id="detail")
        yield Static("Scoreboard: (idle)", id="scoreboard")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#tests", DataTable)
        table.cursor_type = "row"
        table.add_column("Test", key="test")
        table.add_column("Status", key="status")
        for test_id in self._test_ids:
            table.add_row(_short(test_id), "queued", key=test_id)
        # Drive the (blocking, subprocess-heavy) pipeline off the event loop.
        self.run_worker(self._drive, thread=True, exclusive=True)

    # -- worker thread -----------------------------------------------------
    def _drive(self) -> None:
        pipeline = self._pipeline_factory(self._emit)
        pipeline.run(self._test_ids)

    def _emit(self, event: PipelineEvent) -> None:
        # Runs on the worker thread: hand off to the UI thread, do not touch
        # widgets here.
        self.post_message(PipelineUpdate(event))

    # -- UI thread ---------------------------------------------------------
    def on_pipeline_update(self, message: PipelineUpdate) -> None:
        event = message.event
        self._latest[event.test_id] = event
        self._update_row(event)
        if self._selected is None:
            self._selected = event.test_id
        if event.test_id == self._selected:
            self._render_detail(event.test_id)
            self._render_scoreboard(event.test_id)

    def on_data_table_row_highlighted(
        self, message: DataTable.RowHighlighted
    ) -> None:
        self._selected = str(message.row_key.value)
        self._render_detail(self._selected)
        self._render_scoreboard(self._selected)

    def _update_row(self, event: PipelineEvent) -> None:
        table = self.query_one("#tests", DataTable)
        table.update_cell(event.test_id, "status", _status_label(event))

    def _render_detail(self, test_id: str) -> None:
        event = self._latest.get(test_id)
        detail = self.query_one("#detail", Static)
        if event is None:
            detail.update(f"{test_id}\n\n(queued)")
            return

        lines = [test_id, ""]
        if event.error is not None:
            lines.append(f"ERROR: {event.error}")
        if event.cause is not None:
            c = event.cause
            lines.append(f"Root cause: {c.category}  (confidence {c.confidence:.2f})")
            lines.append(f"Auto-fixable: {c.auto_fixable}")
            if c.evidence:
                lines.append("Evidence:")
                lines.extend(f"  - {item}" for item in c.evidence)
            lines.append("")
        if event.proposal is not None:
            p = event.proposal
            lines.append(f"Rationale: {p.rationale}")
            lines.append(f"Files touched: {', '.join(p.files_touched) or '(none)'}")
            lines.append("")
            lines.append("Diff:")
            lines.append(p.diff or "(empty)")
        elif event.cause is None:
            lines.append(f"Status: {_status_label(event)}")
        detail.update("\n".join(lines))

    def _render_scoreboard(self, test_id: str) -> None:
        event = self._latest.get(test_id)
        board = self.query_one("#scoreboard", Static)
        if event is None:
            board.update("Scoreboard: (idle)")
            return
        if event.phase == "verifying":
            # Verifier.verify() blocks with no per-run progress hook, and we
            # deliberately do not modify verifier.py, so this is a coarse status.
            board.update("Scoreboard: running verification battery...")
            return
        if event.result is not None:
            r = event.result
            board.update(
                f"Scoreboard [{test_id}]  "
                f"before {r.failure_rate_before:.0%} -> after {r.failure_rate_after:.0%}  |  "
                f"after 95% CI upper {r.ci95_upper_after:.3f}  |  "
                f"verdict: {r.verdict}"
            )
            return
        if event.verdict is not None:
            v = event.verdict
            board.update(
                f"Scoreboard [{test_id}]  "
                f"failure rate {v.failure_rate:.0%} over {v.n_runs} runs  |  "
                f"95% CI upper {v.ci95_upper:.3f}"
            )
            return
        board.update(f"Scoreboard [{test_id}]: {_status_label(event)}")
