"""Re-run one task into a finished run, without corrupting what it says.

After a harness bug, a throttled package host or an endpoint that dropped, you
usually want one cell corrected and the other twenty-four left alone. Harbor has
no notion of this: a run is a run, and re-running a task produces a *second*
run. The correction therefore has to be grafted -- the new trial directory
copied in beside the original -- and that is where the damage happens.

Doing it by hand has four ways to go wrong, and three of them are silent:

**Forget the marker and the run's clock is destroyed.** Wall clock runs from the
manifest's start to the last trial to finish, so a next-day graft ends the run a
day later. Measured: a 3.7-hour opencode run reported 27.1 hours, and its
LLM-busy share fell from 93% to 13%. Both wrong, and nothing on screen says so.

**Name the directory wrong and the graft loses.** `load_run` keeps the last
attempt per task sorted by directory name, so a suffix that sorts *before* the
original leaves the failed attempt winning and the correction invisible.

**Use different settings and the cell is not comparable to its neighbours.** A
graft at a different timeout multiplier, attempt count or context window is a
different experiment sitting in the middle of one that was held constant.

**Let the catalog drift and the graft measures a different harness.** This is
the newest and the least visible: the pinned version, the output cap and the
reasoning effort all come from the catalog at run time, and the catalog is
edited between runs. A hermes trial grafted after `{reasoning_effort}` was added
to its block is a thinking trial in a run of non-thinking ones.

So this module refuses rather than warns. It runs the task into a scratch
directory first, then compares the scratch run's own manifest against the
original's, field by field, and grafts only if they agree. Every field it checks
is one the manifest already records precisely because a run is not interpretable
without it -- this is that record being used, rather than re-derived.

    harness-arena rerun --run <job> --task <name>

Nothing is copied until the comparison passes, and the failed original is never
deleted: it stays on disk as the evidence for why the graft exists.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.collect import MANIFEST_NAME, RERUN_MARKER, _read_json
from bench.config import load as load_config

#: Suffix given to a grafted trial directory. Two jobs: it must sort *after*
#: whatever Harbor generated, because the last attempt per task name wins, and
#: it has to be recognisable on disk. Harbor's own suffixes are seven random
#: alphanumerics, so a "zz-" prefix sorts after any of them.
GRAFT_PREFIX = "zz-rerun"

#: Manifest fields that must agree for a graft to be comparable to the trials
#: beside it. Each is here because a difference in it makes the grafted cell
#: measure something the rest of the run did not.
#:
#: The model fingerprint is the strictest: it changes when the weights change,
#: which is the one thing this whole rig exists to hold constant. The harness
#: version is next -- the catalog pins it precisely so two runs a week apart
#: measure the same software. The rest describe how hard the harness was
#: allowed to work.
COMPARABILITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("model.fingerprint", "the weights being served"),
    ("harness", "the harness"),
    ("harness_version", "the pinned harness build"),
    ("dataset", "the benchmark"),
    ("context_window", "the context window"),
    ("agent_max_tokens", "the output ceiling"),
    ("agent_timeout_multiplier", "the timeout multiplier"),
    ("n_attempts", "the attempts per task"),
    ("reasoning_effort", "the reasoning effort"),
    ("reasoning_effort_applied", "whether the effort reached the harness"),
)


def _dig(data: dict[str, Any], path: str) -> Any:
    """`model.fingerprint` out of a nested manifest."""
    cursor: Any = data
    for part in path.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return cursor


def find_run(name: str, runs_dir: Path) -> Path:
    """A run directory from a name, a partial name, or a path."""
    candidate = Path(name)
    if candidate.is_dir() and (candidate / MANIFEST_NAME).exists():
        return candidate.resolve()
    exact = runs_dir / name
    if exact.is_dir():
        return exact.resolve()
    matches = sorted(
        p for p in runs_dir.iterdir()
        if p.is_dir() and name in p.name and (p / MANIFEST_NAME).exists()
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise SystemExit(
            f"No run matching {name!r} under {runs_dir}.\n"
            f"  Pass the directory name as it appears in runs/."
        )
    raise SystemExit(
        f"{name!r} matches {len(matches)} runs:\n"
        + "\n".join(f"  {p.name}" for p in matches)
        + "\n  Be more specific."
    )


def trials_for(run_dir: Path, task: str) -> list[Path]:
    """Every attempt at one task already in this run, oldest name first."""
    return sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and p.name.rsplit("__", 1)[0] == task
    )


def describe_trial(trial_dir: Path) -> str:
    """One line about what an existing attempt did, for the confirmation."""
    result = _read_json(trial_dir / "result.json") or {}
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    exc = (result.get("exception_info") or {}).get("exception_type")
    parts = [f"reward={reward}" if reward is not None else "unscored"]
    if exc:
        parts.append(exc)
    if (trial_dir / RERUN_MARKER).exists():
        parts.append("already a graft")
    return f"{trial_dir.name}  ({', '.join(parts)})"


def runner_argv(manifest: dict[str, Any], task: str, scratch: Path) -> list[str]:
    """The `bench.runner` invocation that reproduces this run for one task.

    Settings come from the manifest rather than from the catalog's current
    defaults: the point of a graft is to land in a run that was held constant,
    and the defaults are exactly what may have moved since.
    """
    argv = [
        sys.executable, "-m", "bench.runner",
        "--no-input",
        "--harness", str(manifest.get("harness")),
        "--task", task,
        "--jobs-dir", str(scratch),
    ]
    dataset = manifest.get("dataset")
    if dataset:
        argv += ["--dataset", dataset]
    for flag, key in (
        ("--agent-timeout-multiplier", "agent_timeout_multiplier"),
        ("--n-attempts", "n_attempts"),
        ("--n-concurrent-agents", "n_concurrent_agents"),
        ("--max-retries", "max_retries"),
    ):
        value = manifest.get(key)
        if value:
            argv += [flag, str(value)]
    # Recorded as resolved, so pass it explicitly: leaving it out would let the
    # endpoint be re-probed and possibly answer differently than it did then.
    effort = manifest.get("reasoning_effort")
    if effort and manifest.get("reasoning_effort_applied") is not False:
        argv += ["--reasoning-effort", str(effort)]
    return argv


def comparability_gap(
    original: dict[str, Any], replay: dict[str, Any]
) -> list[str]:
    """Every field on which the replay is not the same experiment."""
    gaps = []
    for path, label in COMPARABILITY_FIELDS:
        was, now = _dig(original, path), _dig(replay, path)
        # A field the original never recorded cannot be compared against. Those
        # manifests predate the field; refusing them would make old runs
        # ungraftable for a reason that is about this rig, not about them.
        if was is None:
            continue
        if was != now:
            gaps.append(f"{label}: was {was!r}, now {now!r}")
    return gaps


def graft(
    scratch_trial: Path, run_dir: Path, task: str, supersedes: list[Path], why: str
) -> Path:
    """Copy a finished trial into a run, marked as the graft it is."""
    target = run_dir / f"{task}__{GRAFT_PREFIX}"
    # A second graft of the same task gets its own directory rather than
    # overwriting: the previous correction is evidence too, and the newest still
    # sorts last.
    if target.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = run_dir / f"{task}__{GRAFT_PREFIX}-{stamp}"
    shutil.copytree(scratch_trial, target)
    # Written last and always: without it the run's clock ends at this trial's
    # finish time, which is the single most damaging way this can go wrong.
    (target / RERUN_MARKER).write_text(
        json.dumps(
            {
                "supersedes": [p.name for p in supersedes],
                "why": why,
                "grafted_at": datetime.now(UTC).isoformat(),
                "source": str(scratch_trial),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness-arena rerun",
        description=(
            "Re-run one task and graft the result into a finished run, keeping "
            "the run's timing and comparability intact."
        ),
    )
    parser.add_argument(
        "--run", required=True,
        help="The finished run to correct: a directory name, a unique part of "
             "one, or a path.",
    )
    parser.add_argument(
        "--task", required=True, action="append", dest="tasks",
        help="Task name to re-run. Repeat for several.",
    )
    parser.add_argument(
        "--why", default="",
        help="Recorded in the graft marker. Say what was fixed.",
    )
    parser.add_argument(
        "--scratch", default=None,
        help="Where to run before grafting (default: a sibling .rerun directory).",
    )
    parser.add_argument(
        "--keep-scratch", action="store_true",
        help="Leave the scratch run on disk instead of deleting it.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Graft even when the replay is not the same experiment. Records "
             "the differences in the marker.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run and graft nothing.")
    parser.add_argument("--yes", action="store_true", help="Do not ask.")
    args = parser.parse_args(argv)

    runs_dir = load_config().resolved_runs_dir()
    run_dir = find_run(args.run, runs_dir)
    manifest = _read_json(run_dir / MANIFEST_NAME)
    if not manifest:
        raise SystemExit(f"{run_dir.name} has no {MANIFEST_NAME}; nothing to match.")

    print(f"\n  run     : {run_dir.name}")
    print(f"  harness : {manifest.get('harness')} {manifest.get('harness_version') or ''}")
    print(f"  model   : {_dig(manifest, 'model.label')} "
          f"({_dig(manifest, 'model.fingerprint')})")

    for task in args.tasks:
        existing = trials_for(run_dir, task)
        if not existing:
            raise SystemExit(
                f"{run_dir.name} has no trial for task {task!r}.\n"
                f"  A graft corrects a cell that exists; it cannot add one the "
                f"run never attempted."
            )
        print(f"\n  {task}")
        for trial in existing:
            print(f"    existing: {describe_trial(trial)}")

    scratch = Path(args.scratch) if args.scratch else run_dir.parent / ".rerun"
    failures = 0
    for task in args.tasks:
        argv_run = runner_argv(manifest, task, scratch)
        printable = " ".join(argv_run)
        print(f"\n  $ {printable}")
        if args.dry_run:
            continue
        if not args.yes:
            answer = input(f"  re-run {task} and graft it? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("  skipped.")
                continue

        before = {p.name for p in scratch.iterdir()} if scratch.is_dir() else set()
        result = subprocess.run(argv_run, check=False)
        produced = sorted(
            p for p in scratch.iterdir()
            if p.is_dir() and p.name not in before
        ) if scratch.is_dir() else []
        if result.returncode != 0 and not produced:
            print(f"  [!] the re-run produced nothing for {task}; nothing grafted.")
            failures += 1
            continue

        job = produced[-1]
        replay = _read_json(job / MANIFEST_NAME) or {}
        trials = [p for p in sorted(job.iterdir())
                  if p.is_dir() and p.name.rsplit("__", 1)[0] == task]
        if not trials:
            print(f"  [!] the re-run wrote no trial for {task}; nothing grafted.")
            failures += 1
            continue

        gaps = comparability_gap(manifest, replay)
        if gaps:
            print(f"\n  [!] this re-run is not the same experiment as "
                  f"{run_dir.name}:")
            for gap in gaps:
                print(f"        {gap}")
            if not args.force:
                print("  Not grafted. A cell measured under different settings "
                      "is worse than a cell that failed, because it looks fine.")
                print("  Re-run with --force to graft it anyway; the marker "
                      "records the differences.")
                failures += 1
                continue
            print("  --force given: grafting anyway.")

        why = args.why or "re-run after a failure"
        target = graft(trials[-1], run_dir, task, trials_for(run_dir, task), why)
        if gaps:
            marker = _read_json(target / RERUN_MARKER) or {}
            marker["comparability_gaps"] = gaps
            (target / RERUN_MARKER).write_text(
                json.dumps(marker, indent=2), encoding="utf-8"
            )
        print(f"  grafted -> {target.name}")
        print("  the superseded attempt is left on disk as evidence.")

    if scratch.is_dir() and not args.keep_scratch and not args.dry_run:
        shutil.rmtree(scratch, ignore_errors=True)

    if not args.dry_run:
        print("\n  The graft counts toward the score, the checks and the tokens, "
              "and is left out of the two timing figures -- it did not happen "
              "inside this run's window.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
