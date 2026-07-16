\# FlakeHunter — Technical Spec



\## Mission

Detect flaky tests (pass/fail nondeterministically on the same commit), classify

their root cause via controlled experiments, generate a fix with the Codex SDK,

and verify the fix with a strict statistical contract before accepting it.



\## Root-cause taxonomy (auto-fixable: A, B, C. Detect-only: D, E.)

\- \*\*A — Order dependency:\*\* test depends on state left by another test.

&#x20; Detect: runs isolated vs. in full suite; Fisher exact test on the 2x2

&#x20; (failures/passes isolated vs. failures/passes in-suite).

\- \*\*B — Shared mutable state:\*\* globals/singletons/temp files not reset.

&#x20; Detect: runs with vs. without `--forked` (pytest-forked); Fisher exact.

\- \*\*C — Unseeded randomness:\*\* `random`/`numpy.random`/hash order used without

&#x20; a fixed seed. Detect: runs with vs. without fixed `PYTHONHASHSEED` and

&#x20; `--randomly-seed`; Fisher exact.

\- \*\*D — Timing/async dependency:\*\* `time.sleep`, race conditions, clock-crossing.

&#x20; Detect: AST signals (calls to `time.sleep`, `datetime.now`, async primitives).

&#x20; Fix is suggested, not auto-applied.

\- \*\*E — External/network/IO dependency:\*\* hits real network/disk/DNS.

&#x20; Detect: AST signals (`requests.\*`, `socket.\*`, fixed file paths) + exception

&#x20; patterns. Fix is suggested, not auto-applied.



\## The five contracts (implement exactly against these; do not change signatures)



```python

@dataclass(frozen=True)

class RunResult:

&#x20;   test\_id: str

&#x20;   outcome: Literal\["passed", "failed", "error", "skipped"]

&#x20;   duration\_s: float

&#x20;   error\_repr: str | None

&#x20;   seed\_env: dict\[str, str]

&#x20;   order\_hash: str



@dataclass(frozen=True)

class FlakeVerdict:

&#x20;   test\_id: str

&#x20;   n\_runs: int

&#x20;   n\_failures: int

&#x20;   failure\_rate: float

&#x20;   ci95\_upper: float          # Wilson score interval upper bound

&#x20;   is\_flaky: bool             # True iff 0 < n\_failures < n\_runs

&#x20;   sample\_tracebacks: list\[str]



@dataclass(frozen=True)

class RootCause:

&#x20;   test\_id: str

&#x20;   category: Literal\["order", "shared\_state", "randomness", "timing", "external"]

&#x20;   confidence: float

&#x20;   evidence: list\[str]        # e.g. "Fisher exact p=0.001, isolated vs suite"

&#x20;   ast\_signals: list\[str]

&#x20;   auto\_fixable: bool          # True only for order/shared\_state/randomness



@dataclass(frozen=True)

class FixProposal:

&#x20;   test\_id: str

&#x20;   diff: str                  # unified diff, applicable via `git apply`

&#x20;   rationale: str

&#x20;   files\_touched: list\[str]

&#x20;   codex\_session\_id: str



@dataclass(frozen=True)

class VerificationResult:

&#x20;   test\_id: str

&#x20;   contract\_passed: bool

&#x20;   assertion\_count\_preserved: bool

&#x20;   no\_skip\_introduced: bool

&#x20;   failure\_rate\_before: float

&#x20;   failure\_rate\_after: float

&#x20;   ci95\_upper\_after: float

&#x20;   suite\_still\_green: bool

&#x20;   verdict: Literal\["verified\_fix", "rejected\_weakens\_test",

&#x20;                     "rejected\_still\_flaky", "rejected\_breaks\_suite"]

```



\## The Verification Contract (never relax this)

A fix is accepted only if ALL of:

1\. \*\*Structural:\*\* AST assertion count does not decrease; no `skip`/`xfail`

&#x20;  decorators or masking retries introduced.

2\. \*\*Statistical:\*\* re-run N\_verify times; failure rate's Wilson CI upper bound

&#x20;  falls below the target threshold (e.g. ≤2%); improvement vs. before is

&#x20;  significant via Fisher exact test (p < 0.05).

3\. \*\*No regression:\*\* the rest of the relevant suite still passes after

&#x20;  applying the fix.



\## Statistics module (`stats.py`) — required functions

\- `wilson\_interval(k, n, confidence=0.95) -> (float, float)`

\- `rule\_of\_three\_upper(n) -> float`  # ≈ 3/n, for k=0 case

\- `n\_for\_detection(p, beta=0.05) -> int`  # n ≥ log(beta)/log(1-p)

\- `n\_for\_verification(threshold) -> int`  # ≈ 3/threshold when expecting k=0

\- `fisher\_exact\_pvalue(k1, n1, k2, n2) -> float`

\- `bh\_fdr\_correction(pvalues: list\[float]) -> list\[float]`  # Benjamini-Hochberg

Each function must have its own tests against known reference values.



\## Repo layout

```

src/flakehunter/

&#x20; contracts.py   # the dataclasses above — written by hand, never auto-edited

&#x20; stats.py       # statistics primitives — written by hand, has its own tests

&#x20; runner.py      # SandboxRunner — pytest-json-report, PYTHONHASHSEED, --forked, timeouts

&#x20; detector.py    # Detector — orchestrates N runs, computes FlakeVerdict

&#x20; classifier.py  # Classifier — differential experiments + AST analysis

&#x20; fixer.py       # Fixer — Codex SDK integration, category-specific prompts

&#x20; verifier.py    # Verifier — the Verification Contract

&#x20; tui.py         # Textual interface

&#x20; cli.py         # entrypoint: `flakehunter run|detect|classify|reproduce`

demo/            # seeded demo repo, see docs/DEMO\_CASES.md

```



\## Non-negotiables

\- Never parse pytest's text output — always `pytest-json-report`, parse JSON.

\- Every sandbox execution has a hard timeout; never hang indefinitely.

\- Diffs are applied to a temporary copy of the target repo, never in place.

