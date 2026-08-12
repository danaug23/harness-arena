"""Wall-clock throughput per run: the only metric that shows pipelining.

Per-trial duration is the wrong measure. Pipelining does not make a trial
faster -- its install still counts inside that trial's clock -- it overlaps that
install with another trial's generation. The payoff appears only as trials per
hour across the whole run, measured from the job's start to its last completed
trial.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from bench.config import load


def _t(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    runs_dir = args.runs_dir or load().resolved_runs_dir()
    if not runs_dir.exists():
        print(f"No runs found under {runs_dir}")
        return 0

    print(
        f"{'harness':<8} {'cfg':<14} {'done':>5} {'elapsed':>9} "
        f"{'min/trial':>10} {'llm busy':>8}  status"
    )
    print("-" * 78)

    for job in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        manifest_path = job / "harness-bench.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        started = _t(manifest.get("started_at"))
        if not started:
            continue

        finishes: list[datetime] = []
        agent_seconds = 0.0
        for trial in job.iterdir():
            result = trial / "result.json"
            if not (trial.is_dir() and result.exists()):
                continue
            try:
                data = json.loads(result.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            when = _t(data.get("finished_at"))
            if when:
                finishes.append(when)
            phase = data.get("agent_execution") or {}
            a, b = _t(phase.get("started_at")), _t(phase.get("finished_at"))
            if a and b:
                agent_seconds += (b - a).total_seconds()

        n = len(finishes)
        if not n:
            continue

        # Anchor on the job's own start, and end at either the last completed
        # trial (finished/stopped run) or now (still running). Timing a live run
        # from an observer's start instead would report a fictional speedup.
        stopped = _t(manifest.get("stopped_at"))
        end = max(finishes) if stopped else datetime.now(UTC)
        elapsed_min = (end - started).total_seconds() / 60
        per_trial = elapsed_min / n

        cfg = (
            f"{manifest.get('agent_timeout_multiplier')}x "
            f"{manifest.get('n_concurrent', 1)}/{manifest.get('n_concurrent_agents') or 1}"
        )
        status = "stopped" if stopped else "running"
        # Share of wall clock the model spent generating. This is what pipelining
        # is actually for, and unlike min/trial it stays honest mid-run: work in
        # flight is uncounted by trial-completion metrics but shows up here as
        # soon as those trials finish.
        util = 100 * (agent_seconds / 60) / elapsed_min if elapsed_min else 0
        print(
            f"{manifest.get('harness', '?'):<8} {cfg:<14} {n:>5} "
            f"{elapsed_min/60:>7.1f}h {per_trial:>9.1f} {util:>7.0f}%  {status}"
        )

    print("\ncfg = timeout multiplier, trials-in-flight / agents-generating")
    print("min/trial: wall clock per COMPLETED trial -- understates a run with")
    print("           work still in flight, so it is only final once a run ends.")
    print("llm busy%: share of elapsed time the model spent generating. This is")
    print("           the pipelining target; it should approach 100%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
