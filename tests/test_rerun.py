"""Guard the one operation that can silently corrupt a finished run.

Grafting a re-run into a run is the only way to correct one cell without paying
for the other twenty-four, and three of its four failure modes are silent: a
missing marker destroys the run's clock, a badly-sorted name leaves the failed
attempt winning, and a settings or catalog difference puts a cell measured under
one configuration into a run measured under another.

    python tests/test_rerun.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench.collect import RERUN_MARKER, load_run  # noqa: E402
from bench.rerun import (  # noqa: E402
    COMPARABILITY_FIELDS,
    comparability_gap,
    find_run,
    graft,
    runner_argv,
    trials_for,
)

failures: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<58} got={got!r} want={want!r}")
    if not ok:
        failures.append(label)


def check_true(label: str, got: object) -> None:
    check(label, bool(got), True)


MANIFEST = {
    "harness": "hermes",
    "harness_version": "v2026.8.3",
    "harness_label": "Hermes Agent",
    "model": {"label": "M", "fingerprint": "fp1", "total_slots": 1},
    "dataset": "terminal-bench@2.0",
    "context_window": 131072,
    "agent_max_tokens": 16384,
    "agent_timeout_multiplier": 8.0,
    "n_attempts": 1,
    "n_concurrent": 2,
    "n_concurrent_agents": 1,
    "reasoning_effort": "high",
    "reasoning_effort_applied": True,
    "started_at": "2026-08-19T04:00:00+00:00",
}


def _run(tmp: Path, name: str = "hermes__m__20260819T040000Z") -> Path:
    job = tmp / name
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (job / "result.json").write_text(
        json.dumps({"finished_at": "2026-08-19T04:03:00"}), encoding="utf-8"
    )
    return job


def _trial(job: Path, task: str, *, suffix: str, resolved: bool,
           exception: str | None = None, started: str = "2026-08-19T04:00:00Z",
           finished: str = "2026-08-19T04:01:00Z", agent_s: int = 60,
           tokens: int | None = 1000) -> Path:
    d = job / f"{task}__{suffix}"
    (d / "agent").mkdir(parents=True)
    result = {
        "task_name": task,
        "verifier_result": {"rewards": {"reward": 1.0 if resolved else 0.0}},
        "started_at": started,
        "finished_at": finished,
        "agent_execution": {"started_at": started, "finished_at": finished},
        "agent_result": {
            "n_input_tokens": None if tokens is None else tokens * 4,
            "n_cache_tokens": None,
            "n_output_tokens": tokens,
            "cost_usd": None,
        },
    }
    if exception:
        result["exception_info"] = {
            "exception_type": exception, "exception_message": "boom"
        }
    (d / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Reproducing the run, rather than the catalog's current defaults
# ---------------------------------------------------------------------------

print("\n-- the replay command --")

_argv = runner_argv(MANIFEST, "crack-7z-hash", Path("/scratch"))
_line = " ".join(_argv)
# Every one of these comes off the manifest, not the catalog. The catalog is
# exactly what may have moved since the run -- that is why the manifest records
# them in the first place.
check_true("the harness is taken from the run", "--harness hermes" in _line)
check_true("...and only the one task runs", "--task crack-7z-hash" in _line)
check_true("...at the run's timeout multiplier",
           "--agent-timeout-multiplier 8.0" in _line)
check_true("...its attempt count", "--n-attempts 1" in _line)
check_true("...its benchmark", "--dataset terminal-bench@2.0" in _line)
check_true("...and its reasoning effort", "--reasoning-effort high" in _line)
# A harness that never received an effort must not be handed one now: that
# would make the graft a thinking trial in a run of non-thinking ones.
_no_effort = {**MANIFEST, "reasoning_effort_applied": False}
check_true("a harness that was never told an effort is still not told",
           "--reasoning-effort" not in " ".join(
               runner_argv(_no_effort, "t", Path("/s"))))
check_true("the graft never writes into the original run",
           "--jobs-dir" in _line and "scratch" in _line)


# ---------------------------------------------------------------------------
# Refusing a graft that would not be the same experiment
# ---------------------------------------------------------------------------

print("\n-- comparability --")

check("an identical replay has no gaps", comparability_gap(MANIFEST, MANIFEST), [])

# The one this rig exists to hold constant.
_other_weights = {**MANIFEST, "model": {**MANIFEST["model"], "fingerprint": "fp2"}}
check("different weights are refused",
      comparability_gap(MANIFEST, _other_weights),
      ["the weights being served: was 'fp1', now 'fp2'"])

# The catalog drift case: hermes gained {reasoning_effort} between runs, so a
# graft taken afterwards is a thinking trial in a run of non-thinking ones.
_now_reasons = {**MANIFEST, "reasoning_effort_applied": False}
check_true("a harness that has since gained an effort knob is refused",
           any("effort reached" in g
               for g in comparability_gap(_now_reasons, MANIFEST)))

_bumped = {**MANIFEST, "harness_version": "v2026.9.1"}
check_true("a bumped harness pin is refused",
           any("pinned harness build" in g
               for g in comparability_gap(MANIFEST, _bumped)))
_uncapped = {**MANIFEST, "agent_max_tokens": None}
check_true("a changed output ceiling is refused",
           any("output ceiling" in g
               for g in comparability_gap(MANIFEST, _uncapped)))

# A field the original never recorded cannot be compared against. Refusing on
# it would make every run written before that field ungraftable, for a reason
# about this rig rather than about the run.
_old = {k: v for k, v in MANIFEST.items() if k != "reasoning_effort_applied"}
check("a field the original never recorded is not a gap",
      comparability_gap(_old, MANIFEST), [])

check("every guarded field is documented with a reason",
      all(isinstance(label, str) and label for _, label in COMPARABILITY_FIELDS),
      True)


# ---------------------------------------------------------------------------
# The graft itself: it must win, and it must not end the run
# ---------------------------------------------------------------------------

print("\n-- grafting --")

with tempfile.TemporaryDirectory() as scratch:
    tmp = Path(scratch)
    job = _run(tmp)
    _trial(job, "kept", suffix="aaa", resolved=True)
    # Failed on the day: a package host declined to serve the installer.
    failed = _trial(job, "broken", suffix="mmm", resolved=False,
                    exception="NetworkConnectionError", tokens=None)

    # The correction, produced a day later and therefore carrying its own clock.
    source_job = tmp / "scratch-run"
    (source_job).mkdir()
    fixed = _trial(source_job, "broken", suffix="zzz", resolved=True,
                   started="2026-08-20T04:00:00Z",
                   finished="2026-08-20T04:30:00Z", agent_s=1800)

    target = graft(fixed, job, "broken", [failed], "package host was throttling")

    check_true("the graft sorts after the attempt it supersedes",
               target.name > failed.name)
    check_true("...and is marked as a graft", (target / RERUN_MARKER).exists())
    marker = json.loads((target / RERUN_MARKER).read_text(encoding="utf-8"))
    check("...naming what it supersedes", marker["supersedes"], [failed.name])
    check("...and why", marker["why"], "package host was throttling")
    check_true("the superseded attempt is left on disk as evidence",
               failed.exists())

    run = load_run(job)
    by_task = {t["task_name"]: t for t in run["tasks"]}
    check("the graft is the attempt that counts", by_task["broken"]["resolved"], True)
    check("...and its failure does not follow it", run["n_errors"], 0)
    check("...and it is flagged as a graft", by_task["broken"]["spliced"], True)
    check("the run scores both tasks", run["n_resolved"], 2)

    # The whole reason the marker exists, asserted by taking it away. The run
    # itself is one minute long; the graft finished a day later. A measured
    # 3.7-hour run once reported 27.1 hours this way, with its LLM-busy share
    # falling from 93% to 13% -- both wrong, and nothing on screen said so.
    check("the clock ends with the run, not with the graft",
          run["wall_clock_s"], 60.0)
    check_true("...and the busy share stays a share",
               0 < run["llm_busy_pct"] <= 100)

    (target / RERUN_MARKER).unlink()
    unmarked = load_run(job)
    check_true("without the marker the graft would end the run a day later",
               unmarked["wall_clock_s"] > 86_000)
    check_true("...and the busy share would collapse",
               unmarked["llm_busy_pct"] < 5)
    (target / RERUN_MARKER).write_text(
        json.dumps(marker), encoding="utf-8"
    )
    check("restoring the marker restores the clock",
          load_run(job)["wall_clock_s"], 60.0)
    # The graft's tokens count -- it is a real result, and the cell it replaces
    # reported none at all.
    check("the graft's tokens are counted",
          by_task["broken"]["n_output_tokens"], 1000)

    # A second correction of the same task must not overwrite the first.
    again = graft(fixed, job, "broken", [failed], "second fix")
    check_true("a second graft gets its own directory", again != target)
    check_true("...and still sorts last", again.name > target.name)
    # The failed original, the first correction, and the second: nothing is
    # overwritten, because a superseded attempt is the evidence for the graft.
    check("...and every attempt remains on disk", len(trials_for(job, "broken")), 3)


# ---------------------------------------------------------------------------
# Finding the run, and refusing to guess
# ---------------------------------------------------------------------------

print("\n-- resolving a run --")

with tempfile.TemporaryDirectory() as scratch:
    tmp = Path(scratch)
    a = _run(tmp, "hermes__m__20260819T040000Z")
    _run(tmp, "hermes__m__20260819T090000Z")
    check("an exact name resolves",
          find_run("hermes__m__20260819T040000Z", tmp).name, a.name)
    check("a unique fragment resolves", find_run("040000", tmp).name, a.name)
    try:
        find_run("hermes", tmp)
        check("an ambiguous fragment is refused", "no error", "SystemExit")
    except SystemExit as exc:
        check_true("an ambiguous fragment is refused, naming the candidates",
                   "matches 2 runs" in str(exc))
    try:
        find_run("nothing-like-this", tmp)
        check("an unknown run is refused", "no error", "SystemExit")
    except SystemExit as exc:
        check_true("an unknown run is refused", "No run matching" in str(exc))


print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
