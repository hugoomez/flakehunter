"""Headless TUI test via Textual's ``run_test`` pilot.

A fake pipeline (no subprocess, no Codex) emits a scripted sequence of events;
we assert the three panels render the live status, root cause, diff, and
scoreboard. Driven through ``asyncio.run`` so no pytest-async plugin is needed.
"""

from __future__ import annotations

import asyncio

from textual.widgets import DataTable, Static

from flakehunter.orchestrator import PipelineEvent
from flakehunter.tui import FlakeHunterApp

from _builders import make_cause, make_proposal, make_result, make_verdict

TEST_ID = "demo/tests/test_deck.py::test_shuffle_preserves_first"


def _verified_script() -> list[PipelineEvent]:
    verdict = make_verdict(TEST_ID)
    cause = make_cause(TEST_ID, category="randomness")
    proposal = make_proposal(TEST_ID)
    result = make_result(TEST_ID, verdict="verified_fix")
    return [
        PipelineEvent(TEST_ID, "detecting"),
        PipelineEvent(TEST_ID, "flaky", verdict=verdict),
        PipelineEvent(TEST_ID, "classifying", verdict=verdict),
        PipelineEvent(TEST_ID, "classified", verdict=verdict, cause=cause),
        PipelineEvent(TEST_ID, "fixing", verdict=verdict, cause=cause),
        PipelineEvent(TEST_ID, "fixed", verdict=verdict, cause=cause, proposal=proposal),
        PipelineEvent(TEST_ID, "verifying", verdict=verdict, cause=cause, proposal=proposal),
        PipelineEvent(
            TEST_ID, "verified", verdict=verdict, cause=cause, proposal=proposal, result=result
        ),
    ]


class _FakePipeline:
    def __init__(self, on_event, script):  # type: ignore[no-untyped-def]
        self._on_event = on_event
        self._script = script

    def run(self, test_ids):  # type: ignore[no-untyped-def]
        for event in self._script:
            self._on_event(event)


def _run_scenario(script: list[PipelineEvent]) -> dict[str, str]:
    async def scenario() -> dict[str, str]:
        app = FlakeHunterApp(
            test_ids=[TEST_ID],
            pipeline_factory=lambda on_event: _FakePipeline(on_event, script),
        )
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            table = app.query_one("#tests", DataTable)
            detail = app.query_one("#detail", Static)
            board = app.query_one("#scoreboard", Static)
            return {
                "status": str(table.get_cell(TEST_ID, "status")),
                "detail": str(detail.render()),
                "scoreboard": str(board.render()),
            }

    return asyncio.run(scenario())


def test_tui_renders_verified_run() -> None:
    panels = _run_scenario(_verified_script())

    assert panels["status"] == "verified"
    assert "Root cause: randomness" in panels["detail"]
    assert "seed the RNG before shuffling" in panels["detail"]
    assert "+new" in panels["detail"]  # the proposed diff body is shown
    assert "verified_fix" in panels["scoreboard"]
    assert "0%" in panels["scoreboard"]  # after failure rate


def test_tui_shows_suggest_only_for_non_auto_fixable() -> None:
    verdict = make_verdict(TEST_ID)
    cause = make_cause(TEST_ID, category="timing")
    script = [
        PipelineEvent(TEST_ID, "detecting"),
        PipelineEvent(TEST_ID, "flaky", verdict=verdict),
        PipelineEvent(TEST_ID, "classifying", verdict=verdict),
        PipelineEvent(TEST_ID, "classified", verdict=verdict, cause=cause),
        PipelineEvent(TEST_ID, "suggest_only", verdict=verdict, cause=cause),
    ]
    panels = _run_scenario(script)

    assert panels["status"] == "suggested (manual)"
    assert "Root cause: timing" in panels["detail"]
    assert "Auto-fixable: False" in panels["detail"]
