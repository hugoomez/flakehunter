\# AGENTS.md



\## Project

FlakeHunter: a CLI tool that detects, classifies, fixes, and statistically

verifies flaky tests in Python test suites.



\## Structure

\- `src/flakehunter/contracts.py` — frozen dataclasses and interfaces. DO NOT

&#x20; modify signatures here without explicit instruction; treat as a fixed spec.

\- `src/flakehunter/stats.py` — statistical primitives (Wilson CI, Fisher exact,

&#x20; power analysis, BH-FDR). Must have its own tests against known values.

\- `src/flakehunter/runner.py`, `detector.py`, `classifier.py`, `fixer.py`,

&#x20; `verifier.py` — implement against `contracts.py`.

\- `demo/` — a seeded demo repo with labeled flaky tests. Never "fix" its

&#x20; flakiness directly; it exists to be flaky on purpose.



\## Conventions

\- Strict type hints everywhere. Dataclasses use `frozen=True`.

\- Never parse pytest's text output. Always use `pytest-json-report` and parse

&#x20; the JSON.

\- Test commands: `uv run pytest src/flakehunter/tests -q`

\- Every new module needs its own test file under `src/flakehunter/tests/`.



\## Workflow

\- Work on a `codex/<module>` branch for substantial features; I will review

&#x20; and merge.

\- Keep this file stable during a session — don't propose edits to AGENTS.md

&#x20; mid-task.

