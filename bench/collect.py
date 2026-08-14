"""Turn Harbor's on-disk job output into one comparable results index.

Harbor writes each trial's result.json the moment that trial finishes, and the
job-level result.json only at the end. Reading the trial files directly is what
makes the dashboard live: a run in flight is just a job directory whose trial
count has not caught up to its task count yet.

Nothing here mutates the runs/ tree -- it is read-only over Harbor's output plus
the harness-bench.json manifest the runner drops alongside it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench import RUNS_DIR
from bench.activity import find_log, tail
from bench.config import load, strip_ansi
from bench.watchdog import health_around

MANIFEST_NAME = "harness-bench.json"

#: Written beside a trial that was re-run later and grafted into a run that had
#: already finished. A graft is not a measurement the run made -- it happened on
#: another day, often after a fix -- so the run has to be able to say which of
#: its trials are grafts. See _wall_clock for the one statistic that has to
#: exclude them.
RERUN_MARKER = "rerun.json"


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Terminal-Bench is 89 tasks. At that size the difference between two bare
    percentages is routinely inside the noise, so every pass rate this module
    emits carries an interval and the dashboard draws it.
    """
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    margin = (
        z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _parse_time(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _utc_time(value: Any) -> datetime | None:
    """A timestamp only if it says which zone it is in.

    Not every writer here does. Our manifest records aware UTC and Harbor's
    per-trial results end in Z, but Harbor's *job* result writes naive local
    time -- the same instant, five hours apart on this machine, with nothing in
    the string to say so. Comparing across them silently turns a five-hour run
    into a twenty-minute one, so an ambiguous timestamp is refused rather than
    guessed at.
    """
    parsed = _parse_time(value)
    return parsed if parsed and parsed.tzinfo else None


def _duration(start: Any, end: Any) -> float | None:
    a, b = _parse_time(start), _parse_time(end)
    if a and b:
        return max(0.0, (b - a).total_seconds())
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # A trial that is mid-write; it will be complete on the next poll.
        return None
    return parsed if isinstance(parsed, dict) else None


# ----------------------------------------------------------------------------
# Trial normalization
# ----------------------------------------------------------------------------


def _is_resolved(verifier_result: dict[str, Any] | None) -> tuple[bool, float | None]:
    """Harbor rewards are a dict of named floats; 'resolved' is all-of-them-1.0.

    Different datasets name their reward keys differently, so we treat any
    reward strictly below 1.0 as unresolved rather than hunting for a magic key.
    """
    if not verifier_result:
        return False, None
    rewards = verifier_result.get("rewards")
    if not isinstance(rewards, dict) or not rewards:
        return False, None
    values = [float(v) for v in rewards.values() if isinstance(v, (int, float))]
    if not values:
        return False, None
    primary = rewards.get("reward")
    score = float(primary) if isinstance(primary, (int, float)) else min(values)
    return all(v >= 1.0 for v in values), score


def _repaired_input(ctx: dict[str, Any]) -> float:
    """``n_input_tokens`` for one agent result, repairing the pre-0.1.10 form.

    Cache reads are a *subset* of a request's prompt, so a well-formed record
    always has ``n_input_tokens >= n_cache_tokens``. Until 0.1.10 the hermes and
    opencode adapters reported input *net* of cache while omp, minion, codex and
    claude-code reported it inclusive, so those two runs violate that invariant
    and understate their prompt totals -- by 18x on a measured 25-task opencode
    run (1.87M reported against 32.6M cache read).

    The adapters are fixed, but runs already on disk keep whatever they were
    written with, and those are exactly the runs someone is comparing. The
    violated invariant identifies them without a version stamp or a guess: no
    correctly-recorded trial can land in this branch, because doing so would
    mean a request read more cache than it had prompt.
    """
    value = ctx.get("n_input_tokens")
    if not isinstance(value, (int, float)):
        return 0
    cache = ctx.get("n_cache_tokens")
    if isinstance(cache, (int, float)) and cache > value:
        return value + cache
    return value


def _token_totals(result: dict[str, Any]) -> dict[str, Any]:
    contexts: list[dict[str, Any]] = []
    agent_result = result.get("agent_result")
    if isinstance(agent_result, dict):
        contexts.append(agent_result)
    for step in result.get("step_results") or []:
        if isinstance(step, dict) and isinstance(step.get("agent_result"), dict):
            contexts.append(step["agent_result"])

    totals = {"n_input_tokens": 0, "n_output_tokens": 0, "cost_usd": 0.0}
    seen = False
    for ctx in contexts:
        for key in ("n_input_tokens", "n_output_tokens", "cost_usd"):
            value = ctx.get(key)
            if isinstance(value, (int, float)):
                totals[key] += _repaired_input(ctx) if key == "n_input_tokens" else value
                seen = True
    if not seen:
        return {"n_input_tokens": None, "n_output_tokens": None, "cost_usd": None}
    if not totals["cost_usd"]:
        totals["cost_usd"] = None
    return totals


def _pretty_check(name: str) -> str:
    """`test_outputs.py::test_input_data_integrity` -> `input data integrity`."""
    name = name.rsplit("::", 1)[-1]
    name = re.sub(r"^test_", "", name)
    return name.replace("_", " ")


def read_checks(trial_dir: Path) -> list[dict[str, str]]:
    """Individual test outcomes from the verifier's CTRF report.

    Scoring is all-or-nothing, so a task that passes five of six checks scores
    identically to one that wrote nothing. That distinction is the difference
    between "nearly solved it" and "had no idea", and it is invisible in the
    reward alone -- so surface the checks themselves.
    """
    report = _read_json(trial_dir / "verifier" / "ctrf.json")
    if not report:
        return []
    tests = ((report.get("results") or {}).get("tests")) or []
    checks: list[dict[str, str]] = []
    for test in tests:
        if not isinstance(test, dict) or not test.get("name"):
            continue
        checks.append(
            {
                "name": _pretty_check(str(test["name"])),
                "status": str(test.get("status") or "unknown"),
            }
        )
    return checks


#: Transport-level failures between the agent and the model endpoint: a
#: refused, reset or timed-out connection. These are infrastructure. They say
#: nothing about the harness, and charging them to it is how a working adapter
#: comes to look broken.
_ENDPOINT_SIGNATURES = (
    "error sending request", "connection refused", "connection reset",
    "connection aborted", "connection error", "max retries exceeded",
    "econnrefused", "econnreset", "etimedout", "socket hang up",
    "remote end closed connection", "server disconnected", "read timed out",
    "failed to connect", "no route to host", "bad gateway",
    "service unavailable", "gateway timeout", "gateway time-out",
    "apiconnectionerror", "upstream connect error",
)

#: Paths only a model endpoint serves. Requiring one on the *same line* as the
#: signature keeps a task's own network work -- an agent curling a dead port is
#: ordinary Terminal-Bench material -- from reading as the benchmark's endpoint
#: falling over. Precision matters more than recall: a missed endpoint fault is
#: scored the way it is today, while a false one silently drops a real harness
#: failure out of the denominator, which is the more damaging mistake.
_API_PATHS = (
    "/chat/completions", "/completions", "/v1/messages", "/v1/responses",
    "/api/chat", "/api/generate", "/v1/models",
)

#: The fatal error is the last thing written; the rest is the agent's own work.
_FAULT_TAIL_LINES = 40


def _fault_kind(result: dict[str, Any], trial_dir: Path | None) -> str | None:
    """Whose fault the trial's exception was, or None if it did not raise."""
    exception = result.get("exception_info") or {}
    if not exception.get("exception_type"):
        return None

    lines = strip_ansi(str(exception.get("exception_message") or "")).splitlines()
    log = find_log(trial_dir) if trial_dir else None
    if log:
        lines += strip_ansi(tail(log, 8_000)).splitlines()[-_FAULT_TAIL_LINES:]

    for line in lines:
        lowered = line.lower()
        if any(sig in lowered for sig in _ENDPOINT_SIGNATURES) and any(
            path in lowered for path in _API_PATHS
        ):
            return "transport"
    return "harness"


def _failed_at(result: dict[str, Any]) -> str:
    """When the trial stopped -- the moment to check the endpoint against.

    The agent's own finish time, not the trial's: verification and teardown run
    afterwards, and a window centred on those would miss the failure entirely.
    """
    agent = result.get("agent_execution") or {}
    return str(agent.get("finished_at") or result.get("finished_at") or "")


def normalize_trial(
    result: dict[str, Any], trial_dir: Path | None = None
) -> dict[str, Any]:
    resolved, reward = _is_resolved(result.get("verifier_result"))
    fault = _fault_kind(result, trial_dir)
    exception = result.get("exception_info") or {}
    tokens = _token_totals(result)
    checks = read_checks(trial_dir) if trial_dir else []
    n_passed = sum(1 for c in checks if c["status"] == "passed")

    return {
        "checks": checks,
        "n_checks": len(checks),
        "n_checks_passed": n_passed,
        "task_name": result.get("task_name") or result.get("trial_name") or "unknown",
        "trial_name": result.get("trial_name"),
        "resolved": resolved,
        "reward": reward,
        "error_type": exception.get("exception_type"),
        # "transport" | "harness" | None -- see _fault_kind. Only harness faults
        # are the harness's to answer for.
        "fault": fault,
        # What the watchdog saw around this trial's last moment, when it was
        # running. This is the difference between "the endpoint dropped" as a
        # guess and as a finding: samples showing the endpoint answering while
        # the harness could not send puts the fault on this side of the wire.
        # Absent means diagnostics were off -- no claim either way.
        "endpoint_health": (
            health_around(trial_dir.parent, _failed_at(result))
            if fault == "transport" and trial_dir
            else {}
        ),
        # Stripped before truncating: a failing harness's stderr ends up in
        # here, and color codes would otherwise eat the 400 characters that
        # were supposed to explain what went wrong.
        "error_message": strip_ansi(exception.get("exception_message") or "")[:400]
        or None,
        # A trial re-run later and grafted into a finished run. It scores like
        # any other, but the run's clock cannot include it -- see _wall_clock.
        "spliced": bool(trial_dir and (trial_dir / RERUN_MARKER).exists()),
        "duration_s": _duration(result.get("started_at"), result.get("finished_at")),
        "agent_s": _duration(
            (result.get("agent_execution") or {}).get("started_at"),
            (result.get("agent_execution") or {}).get("finished_at"),
        ),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        **tokens,
    }


# ----------------------------------------------------------------------------
# Run assembly
# ----------------------------------------------------------------------------


def _wall_clock(
    manifest: dict[str, Any],
    job_result: dict[str, Any] | None,
    tasks: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    """How long the run took, and how much of it the model spent generating.

    Anchored on the job's own start and ended at the last trial to finish --
    not at the sum of trial durations, which double-counts whenever two trials
    overlap and would report a 2-concurrent run as taking twice as long as it
    did.

    A run still going is measured to *now*, so the number grows while you watch
    instead of standing still until the last trial lands. That makes an
    in-flight figure a lower bound rather than a wrong one.
    """
    # Only sources that carry a zone: our own manifest, and the trials, which
    # Harbor writes with a Z. Its job-level result is deliberately not used --
    # see _utc_time.
    started = _utc_time(manifest.get("started_at"))
    if not started:
        return None, None

    # A grafted re-run (see RERUN_MARKER) is left out of both halves of this.
    # It is a real result and it counts everywhere else -- toward the score, the
    # checks, the tokens -- but it ran on its own another day, so letting it end
    # the clock would report a 3.7 hour run as having taken 27, and divide the
    # busy percentage by that same wrong number. Excluded from the generating
    # time too, or the ratio would mix a window with work done outside it.
    measured = [t for t in tasks if not t.get("spliced")]

    finishes = [f for f in (_utc_time(t.get("finished_at")) for t in measured) if f]
    # A run that is over ends at its last trial; one still going runs to now,
    # so the figure grows while you watch instead of standing still.
    over = manifest.get("stopped_at") or (job_result or {}).get("finished_at")
    ended = max(finishes, default=started) if over else datetime.now(UTC)

    wall = max(0.0, (ended - started).total_seconds())
    agent = sum(
        t["agent_s"] for t in measured if isinstance(t.get("agent_s"), (int, float))
    )
    return wall, (agent / wall * 100 if wall else None)


def _expected_task_count(job_dir: Path, job_result: dict[str, Any] | None) -> int | None:
    if job_result and isinstance(job_result.get("n_total_trials"), int):
        return job_result["n_total_trials"]
    config = _read_json(job_dir / "config.json") or {}
    for key in ("n_total_trials", "n_tasks"):
        if isinstance(config.get(key), int):
            return config[key]
    return None


def _job_level_errors(job_result: dict[str, Any] | None) -> dict[str, int]:
    """Exception counts Harbor recorded at the job level, by exception type.

    ``stats.evals[<key>].exception_stats`` maps an exception type to the list of
    trial names that raised it. This is the only record of a trial that failed
    before it could write its own result.json.
    """
    if not job_result:
        return {}
    counts: dict[str, int] = {}
    evals = ((job_result.get("stats") or {}).get("evals") or {})
    if not isinstance(evals, dict):
        return {}
    for eval_stats in evals.values():
        if not isinstance(eval_stats, dict):
            continue
        exception_stats = eval_stats.get("exception_stats") or {}
        if not isinstance(exception_stats, dict):
            continue
        for name, trials in exception_stats.items():
            counts[name] = counts.get(name, 0) + (
                len(trials) if isinstance(trials, list) else 1
            )
    return counts


def load_run(job_dir: Path) -> dict[str, Any] | None:
    """Read one Harbor job directory into a normalized run record."""
    manifest = _read_json(job_dir / MANIFEST_NAME) or {}
    job_result = _read_json(job_dir / "result.json")

    trials: list[dict[str, Any]] = []
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        result = _read_json(trial_dir / "result.json")
        if result:
            trials.append(normalize_trial(result, trial_dir))

    # A job with neither a manifest nor any trials is not ours (or not started).
    if not manifest and not trials and not job_result:
        return None

    # Deduplicate retries: keep the last attempt per task.
    by_task: dict[str, dict[str, Any]] = {}
    for trial in trials:
        by_task[trial["task_name"]] = trial
    tasks = sorted(by_task.values(), key=lambda t: t["task_name"])

    # An endpoint that drops a connection is infrastructure failing, not the
    # harness being wrong, so those trials leave the denominator instead of
    # counting as failures. They stay in `tasks`: the matrix still shows them,
    # marked unscored, because silently dropping a trial is its own kind of lie.
    scored = [t for t in tasks if t["fault"] != "transport"]
    n_attempted = len(tasks)
    n_done = len(scored)
    n_unscored = n_attempted - n_done
    n_total = _expected_task_count(job_dir, job_result) or n_attempted
    n_resolved = sum(1 for t in scored if t["resolved"])
    n_errors = sum(1 for t in scored if t["error_type"])

    low, high = wilson_interval(n_resolved, n_done)
    solved = [t for t in scored if t["resolved"]]
    solved_output = [
        t["n_output_tokens"] for t in solved if isinstance(t["n_output_tokens"], (int, float))
    ]
    # The same tokens over every trial, not just the ones that worked. A harness
    # that solves the cheap tasks and burns six figures on the ones it fails
    # looks economical by the per-solve figure alone -- measured here, one run
    # reported 26k per solve against 125k per trial, while another sat at 0.9x.
    # The gap between the two is the part neither number tells you by itself.
    #
    # A trial the harness never reported tokens for is left out rather than
    # counted as zero: an unreported turn is a missing measurement, and folding
    # it in as free would drag the mean toward whichever harness logs worst.
    all_output = [
        t["n_output_tokens"] for t in scored if isinstance(t["n_output_tokens"], (int, float))
    ]
    # A trial that aborted on a dead connection stopped early for a reason that
    # has nothing to do with how long the harness takes to work.
    durations = [t["duration_s"] for t in scored if isinstance(t["duration_s"], (int, float))]

    # Partial credit: every check the run passed, over every check it faced.
    # Scoring is all-or-nothing, so a run that passes five of six checks on
    # twenty tasks scores zero on all twenty -- indistinguishable here from one
    # that wrote nothing. This says how close it got.
    #
    # Not a score, and not comparable to the pass rate: tasks carry different
    # numbers of checks, so this weights a nine-check task nine times a
    # one-check task, and the checks are not equally hard. It answers "how much
    # of the work landed", not "how good is this harness".
    n_checks_total = sum(t.get("n_checks") or 0 for t in scored)
    n_checks_passed = sum(t.get("n_checks_passed") or 0 for t in scored)

    # The same sum over the tasks it did *not* solve, which is the half that
    # carries information the pass rate has not already given you.
    #
    # A solved task passes every check by definition, so it contributes a
    # guaranteed 100% to the totals above. On real data that dominates them:
    # one run here showed 62.1% overall, of which 43 of 43 checks came from
    # solved tasks and the partial credit on everything else was 35.0%. Ranking
    # by the overall figure put that run first; ranking by what it salvaged from
    # its failures put it last. The overall number is mostly the pass rate
    # wearing a different denominator, so both are reported and neither is
    # allowed to stand in for the other.
    missed = [t for t in scored if not t.get("resolved")]
    n_checks_missed_total = sum(t.get("n_checks") or 0 for t in missed)
    n_checks_missed_passed = sum(t.get("n_checks_passed") or 0 for t in missed)

    wall_clock_s, llm_busy_pct = _wall_clock(manifest, job_result, tasks)

    errors: dict[str, int] = {}
    for trial in scored:
        if trial["error_type"]:
            errors[trial["error_type"]] = errors.get(trial["error_type"], 0) + 1

    # A trial that dies during agent *setup* never writes its own result.json, so
    # a run that failed to install the harness would otherwise render as an empty
    # run rather than a broken one -- the single most misleading way this could
    # fail. Harbor still records those in the job-level stats, so fall back to it.
    setup_errors = _job_level_errors(job_result)
    if setup_errors and not errors:
        errors = setup_errors
        n_errors = sum(setup_errors.values())

    model = manifest.get("model") or {}
    # A job killed mid-flight never writes finished_at, so without the explicit
    # stop marker it would read as "running" forever and its partial results
    # would look like a benchmark still in progress rather than a baseline.
    stopped_at = manifest.get("stopped_at")
    # Against attempts, not scored trials: an unscored trial still ran, and
    # measuring completion by the denominator would leave a finished run
    # reporting "running" forever the moment one connection dropped.
    finished = bool(job_result and job_result.get("finished_at")) or (
        n_total > 0 and n_attempted >= n_total
    )
    if stopped_at and not finished:
        status = "stopped"
    elif finished:
        status = "complete"
    else:
        status = "running"

    return {
        "run_id": job_dir.name,
        "path": str(job_dir),
        "harness": manifest.get("harness") or job_dir.name.split("__")[0],
        "harness_label": manifest.get("harness_label")
        or manifest.get("harness")
        or job_dir.name.split("__")[0],
        # Which sweep this harness ran in. None on anything recorded before
        # sweeps had an identity.
        "batch_id": manifest.get("batch_id"),
        "harness_vendor": manifest.get("harness_vendor"),
        "harness_repo": manifest.get("harness_repo"),
        "agent_ref": manifest.get("agent_ref"),
        # A job dir with no manifest was produced by a bare `harbor run` rather
        # than by bench.runner -- still worth showing (an oracle baseline is the
        # "is this task even solvable" ceiling), but it carries no model identity.
        "model_label": model.get("label") or "unmanaged run (no manifest)",
        "model_fingerprint": model.get("fingerprint"),
        "model_served_id": model.get("served_id"),
        "model_quant": model.get("ftype"),
        "model_params": model.get("n_params"),
        "model_n_ctx": model.get("n_ctx"),
        # What the server said it would sample with. Absent on runs
        # recorded before this was captured, and on any endpoint that
        # does not report it.
        "model_sampling": model.get("sampling"),
        "dataset": manifest.get("dataset"),
        "n_concurrent": manifest.get("n_concurrent"),
        # The cap that actually matters: how many agents talk to the endpoint at
        # once. Overshooting the slot count queues requests at the server and
        # charges the wait to whichever harness makes more calls -- the variable
        # under test -- so a run is not interpretable without both numbers.
        "n_concurrent_agents": manifest.get("n_concurrent_agents"),
        "endpoint_slots": model.get("total_slots"),
        # What the harnesses were told the window was. Two runs at different
        # windows are not the same experiment: it sets when a harness compresses
        # or truncates, and nothing else on screen would reveal a mismatch.
        "context_window": manifest.get("context_window"),
        # Where that number came from. Recorded since the window was first
        # resolved, but never read, so a 4096 nobody knew and a 4096 the server
        # reported rendered identically -- which is the one distinction the
        # field exists to make.
        "context_window_source": manifest.get("context_window_source"),
        # Whether the run offered thinking, and how that was decided. Only some
        # harnesses read it, but it separates two runs that are otherwise
        # described identically -- and the difference is not a small one.
        # Absent on runs recorded before the effort was resolved per endpoint.
        "reasoning_effort": manifest.get("reasoning_effort"),
        "reasoning_effort_source": manifest.get("reasoning_effort_source"),
        "agent_max_tokens": manifest.get("agent_max_tokens"),
        "debug_capture": bool(manifest.get("debug_capture")),
        "n_attempts": manifest.get("n_attempts"),
        # True when the run was restricted to a subset of the dataset. Such a run
        # is a smoke test, not a measurement, and must not be ranked against a
        # full-dataset run as though the two were the same experiment.
        "is_partial": bool(manifest.get("is_partial")),
        "n_tasks_requested": manifest.get("n_tasks_requested"),
        "agent_timeout_multiplier": manifest.get("agent_timeout_multiplier"),
        "subset": manifest.get("subset"),
        "n_timeouts": sum(
            1
            for t in tasks
            if t["error_type"] and "timeout" in t["error_type"].lower()
        ),
        "harbor_version": manifest.get("harbor_version"),
        "started_at": manifest.get("started_at")
        or (job_result or {}).get("started_at"),
        "finished_at": (job_result or {}).get("finished_at"),
        "stopped_at": stopped_at,
        "stopped_reason": manifest.get("stopped_reason"),
        "status": status,
        "n_total": n_total,
        "n_done": n_done,
        "n_attempted": n_attempted,
        # Trials removed from the denominator because the endpoint, not the
        # harness, is what failed.
        "n_unscored": n_unscored,
        "n_resolved": n_resolved,
        "n_errors": n_errors,
        "max_retries": manifest.get("max_retries"),
        "pass_rate": (n_resolved / n_done) if n_done else 0.0,
        "ci_low": low,
        "ci_high": high,
        "mean_output_tokens_per_solve": (
            statistics.fmean(solved_output) if solved_output else None
        ),
        "mean_output_tokens_per_trial": (
            statistics.fmean(all_output) if all_output else None
        ),
        # How many trials the per-trial mean is actually over. It is not always
        # n_done: a harness can finish a trial without reporting usage for it.
        "n_token_samples": len(all_output),
        "median_duration_s": statistics.median(durations) if durations else None,
        # Elapsed time for the whole run, and the share of it the model spent
        # generating. Not the sum of trial durations: that double-counts
        # overlapping trials and would make a 2-concurrent run look twice as
        # slow as it was.
        "n_checks_total": n_checks_total,
        "n_checks_passed": n_checks_passed,
        "check_rate": (n_checks_passed / n_checks_total) if n_checks_total else None,
        "n_checks_missed_total": n_checks_missed_total,
        "n_checks_missed_passed": n_checks_missed_passed,
        "missed_check_rate": (
            (n_checks_missed_passed / n_checks_missed_total)
            if n_checks_missed_total else None
        ),
        "n_missed": len(missed),
        "wall_clock_s": wall_clock_s,
        "llm_busy_pct": llm_busy_pct,
        "total_duration_s": sum(durations) if durations else None,
        "error_types": errors,
        "tasks": tasks,
    }


def load_runs(runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []
    runs = []
    for job_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        try:
            run = load_run(job_dir)
        except OSError:
            continue
        if run:
            runs.append(run)
    runs.sort(key=lambda r: (r.get("started_at") or "", r["run_id"]), reverse=True)
    return runs


# ----------------------------------------------------------------------------
# Cross-run views
# ----------------------------------------------------------------------------


def head_to_head(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, Any]:
    """Which tasks each run solved that the other did not.

    This is the comparison that carries the whole experiment: two harnesses can
    land on the same pass rate while disagreeing on a third of the tasks.
    """
    # Unscored trials are excluded from both sides: a task one run never got a
    # fair attempt at would otherwise land in "only B solved it", which reads
    # as a difference between the harnesses when it is a dropped connection.
    a = {t["task_name"]: t["resolved"] for t in run_a["tasks"] if t.get("fault") != "transport"}
    b = {t["task_name"]: t["resolved"] for t in run_b["tasks"] if t.get("fault") != "transport"}
    shared = sorted(set(a) & set(b))
    only_a = [name for name in shared if a[name] and not b[name]]
    only_b = [name for name in shared if b[name] and not a[name]]
    both = [name for name in shared if a[name] and b[name]]
    neither = [name for name in shared if not a[name] and not b[name]]
    return {
        "a": run_a["run_id"],
        "b": run_b["run_id"],
        "a_label": run_a["harness_label"],
        "b_label": run_b["harness_label"],
        "n_shared": len(shared),
        "only_a": only_a,
        "only_b": only_b,
        "both": both,
        "neither": neither,
        "agreement": (len(both) + len(neither)) / len(shared) if shared else None,
    }


def build_index(runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    runs = load_runs(runs_dir)

    task_names: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for task in run["tasks"]:
            if task["task_name"] not in seen:
                seen.add(task["task_name"])
                task_names.append(task["task_name"])
    task_names.sort()

    models = {}
    for run in runs:
        key = run["model_fingerprint"] or run["model_label"]
        entry = models.setdefault(
            key,
            {
                "fingerprint": run["model_fingerprint"],
                "label": run["model_label"],
                "quant": run["model_quant"],
                "params": run["model_params"],
                "n_ctx": run["model_n_ctx"],
                # The window the harnesses were actually given, which is not
                # always the one the server reported: llama.cpp answers with the
                # loaded value and Ollama answers with nothing, so n_ctx alone
                # reads as "unknown" for a run that had a perfectly definite
                # window. Runs are newest first, and a run predating this field
                # carries None, so take the newest that has one.
                "context_window": None,
                "runs": [],
            },
        )
        entry["runs"].append(run["run_id"])
        if not entry["context_window"] and run.get("context_window"):
            entry["context_window"] = run["context_window"]

    # Pair every two complete runs that share a model, newest first.
    comparisons = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_model.setdefault(run["model_fingerprint"] or run["model_label"], []).append(run)
    for model_runs in by_model.values():
        for i, run_a in enumerate(model_runs):
            for run_b in model_runs[i + 1 :]:
                # Never pair a subset run against a full run: they share task
                # names, so head_to_head would happily "compare" 1 task against
                # 89 and report a disagreement set that means nothing.
                #
                # Nor across different time budgets. A harness given twice the
                # wall clock finishes strictly more tasks, so the "disagreement
                # set" would be measuring the budget, not the harness -- and it
                # would look every bit as authoritative as a real comparison.
                if (
                    run_a["harness"] != run_b["harness"]
                    and run_a["n_done"]
                    and run_b["n_done"]
                    and run_a["is_partial"] == run_b["is_partial"]
                    and run_a["subset"] == run_b["subset"]
                    and run_a["agent_timeout_multiplier"]
                    == run_b["agent_timeout_multiplier"]
                ):
                    comparisons.append(head_to_head(run_a, run_b))

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "runs": runs,
        "task_names": task_names,
        "models": list(models.values()),
        "comparisons": comparisons,
        "summary": {
            "n_runs": len(runs),
            "n_running": sum(1 for r in runs if r["status"] == "running"),
            "n_models": len(models),
            "n_harnesses": len({r["harness"] for r in runs}),
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Dump the full index")
    parser.add_argument("--out", type=Path, default=None, help="Write the index to a file")
    args = parser.parse_args(list(argv) if argv is not None else None)

    runs_dir = args.runs_dir or load().resolved_runs_dir()
    index = build_index(runs_dir)

    if args.out:
        args.out.write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
        return 0
    if args.json:
        print(json.dumps(index, indent=2))
        return 0

    if not index["runs"]:
        print(f"No runs found under {args.runs_dir}")
        return 0

    print(f"{'harness':<16} {'model':<38} {'pass':>10} {'done':>9}  status")
    print("-" * 92)
    for run in index["runs"]:
        rate = f"{run['pass_rate'] * 100:.1f}%"
        ci = f"[{run['ci_low'] * 100:.0f}-{run['ci_high'] * 100:.0f}]"
        print(
            f"{run['harness_label'][:15]:<16} {run['model_label'][:37]:<38} "
            f"{rate:>10} {run['n_done']}/{run['n_total']:<6} {run['status']}  {ci}"
        )
    for comparison in index["comparisons"]:
        print(
            f"\n{comparison['a_label']} vs {comparison['b_label']}: "
            f"{len(comparison['only_a'])} only-A, {len(comparison['only_b'])} only-B, "
            f"{len(comparison['both'])} both, {len(comparison['neither'])} neither"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
