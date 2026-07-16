\# Demo Repo Spec — `demo/`



\## Purpose

A small, plausible Python project (`shopcart`: a shopping cart library) whose

test suite contains deliberately seeded flakiness, one case per auto-detectable

category, plus distractors. This is what FlakeHunter is tested and demoed

against. Never "fix" this repo's flakiness directly — it must stay flaky on

purpose.



\## Source project: `demo/src/shopcart/`

A minimal cart library with: `Catalog` (loads items), `Cart` (add items, add

discounts, compute total), and a module-level cache/singleton used by some

tests on purpose (needed for cases 1 and 2 below). Keep it small — a few dozen

lines total is enough. Business logic correctness is irrelevant; it only needs

to exist so the tests have something real to call.



\## Seeded test cases — `demo/tests/`



\*\*Case 1 — Order dependency (category A), `test\_catalog.py`\*\*

Two tests where the second silently depends on module-level state populated by

the first (a dict at module scope). Passes when run in suite order, fails when

run isolated or in reverse order. Expected failure rate when suite order is

randomized: high (\~50%, order-dependent).



\*\*Case 2 — Shared mutable state (category B), `test\_cart\_discounts.py`\*\*

A test that mutates a module-level singleton cart (not reset between tests),

so its assertion on discount count depends on whether other tests already

mutated the same singleton. Failure rate similarly order-dependent, but

resolved by `--forked` (process isolation) rather than by reordering.



\*\*Case 3 — Unseeded randomness, high rate (category C), `test\_deck.py`\*\*

`random.shuffle` on a list without setting a seed; asserts the first element

equals a specific value. Expected failure rate: very high (\~51/52 ≈ 98%).

Deliberately obvious — used for the "big, undeniable" demo moment.



\*\*Case 4 — Unseeded randomness, subtle rate (category C), `test\_pricing\_rounding.py`\*\*

A test relying on `random.choice` over a small set combined with a rounding

assumption, seeded implicitly by hash order or an unfixed RNG call, tuned so

the failure rate is around 4-6%. This is the case that exercises the

statistical significance test (Fisher exact) rather than being obvious to the

naked eye — keep it subtle on purpose.



\*\*Case 5 — Timing dependency, detect-only (category D), `test\_async\_worker.py`\*\*

A test that starts a fake background worker and asserts completion after a

fixed `time.sleep(0.05)`, which is too short under load. Not auto-fixed —

FlakeHunter should detect and classify it via AST signals (call to

`time.sleep`) but mark `auto\_fixable=False`.



\*\*Distractor 1 — Deterministic failure, `test\_broken\_feature.py`\*\*

A test that always fails (a real bug, not flakiness). Must NEVER be classified

as flaky by the Detector (this validates no false positives).



\*\*Distractor 2 — Always passes, `test\_stable\_total.py`\*\*

A simple, fully deterministic test. Must never be flagged as flaky either.



\## `demo/GROUND\_TRUTH.json`

A JSON file (not read by the tool itself, only used by humans/tests to check

FlakeHunter's own accuracy) mapping each test id to its true category and an

approximate expected failure rate range, e.g.:

```json

{

&#x20; "tests/test\_catalog.py::test\_read\_from\_cache": {"category": "order", "expected\_rate": \[0.3, 0.7]},

&#x20; "tests/test\_cart\_discounts.py::test\_discount\_count": {"category": "shared\_state", "expected\_rate": \[0.3, 0.7]},

&#x20; "tests/test\_deck.py::test\_shuffle\_preserves\_first": {"category": "randomness", "expected\_rate": \[0.9, 1.0]},

&#x20; "tests/test\_pricing\_rounding.py::test\_round\_trip": {"category": "randomness", "expected\_rate": \[0.03, 0.08]},

&#x20; "tests/test\_async\_worker.py::test\_worker\_completes": {"category": "timing", "expected\_rate": \[0.05, 0.3]},

&#x20; "tests/test\_broken\_feature.py::test\_feature\_x": {"category": "deterministic\_failure", "expected\_rate": \[1.0, 1.0]},

&#x20; "tests/test\_stable\_total.py::test\_total\_is\_correct": {"category": "stable", "expected\_rate": \[0.0, 0.0]}

}

```



\## Acceptance check

Running `pytest --count=20 -p no:randomly` (or an equivalent repeated-run

plugin) over `demo/tests/` should show cases 1-5 fluctuating between pass/fail

across runs, while the two distractors are perfectly consistent (always fail /

always pass respectively).

