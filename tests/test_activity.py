"""Guard the live feed's view of a concurrently running benchmark.

At ``--n-concurrent 1`` there is only ever one trial in flight, so "the running
trial" was a well-defined thing and the feed served it. Above one it is not:
observed on a live run at ``--n-concurrent 4``, four trials were working and the
panel showed one of them, chosen by which agent log happened to have been
appended to most recently -- which changes every few seconds, so the feed also
swapped between trials on its own while the reader watched.

``find_active_trials`` reports all of them and the caller picks. What is pinned
here is that it finds every trial and only trials, that the order is stable
across calls (an unstable order makes the tab strip reshuffle under the cursor),
that a selection is honoured, and that a selection which has since finished
falls back *and says it fell back* -- silence there is indistinguishable from
the panel mislabelling whose output it is showing.

The expensive half is deliberate too: the tail is read for the selected trial
only. Reading four tails per poll would multiply the cost of the five-second
refresh by the concurrency, on a page whose scan budget has already caused one
visible regression.

    python tests/test_activity.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.activity import (  # noqa: E402
    find_active_trials,
    read_activity,
    trial_key,
)

failures: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<58} got={got!r} want={want!r}")
    if not ok:
        failures.append(label)


def make_run(runs: Path, name: str, trials: dict[str, dict]) -> Path:
    """A runs/ tree shaped like Harbor's, with the trials described by `trials`.

    Each value may set `done` (a result.json exists), `log` (agent output, which
    is what makes a trial "started"), and `mtime` (to order the logs
    deterministically instead of relying on how fast the filesystem is).
    """
    job = runs / name
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "minion", "harness_label": "dmfa-minion",
        "model": {"label": "Qwen3.8 27B"},
    }), encoding="utf-8")

    for trial_name, spec in trials.items():
        tdir = job / trial_name
        tdir.mkdir()
        # Harbor writes config.json when it creates the trial, before setup.
        (tdir / "config.json").write_text("{}", encoding="utf-8")
        if spec.get("done"):
            (tdir / "result.json").write_text("{}", encoding="utf-8")
        if spec.get("log") is not None:
            agent = tdir / "agent"
            agent.mkdir()
            log = agent / "minion.txt"
            log.write_text(spec["log"], encoding="utf-8")
            if spec.get("mtime"):
                import os
                os.utime(log, (spec["mtime"], spec["mtime"]))
    return job


def test_finds_every_running_trial_and_only_trials(scratch: Path) -> None:
    runs = scratch / "runs"
    now = time.time()
    make_run(runs, "job1", {
        "alpha__a": {"log": "one", "mtime": now - 30},
        "beta__b": {"log": "two", "mtime": now - 10},
        # Created but the agent has not started: no agent/ dir at all. This must
        # still appear, because "setting up" is a state the reader needs to see.
        "gamma__c": {},
        # Finished, so not in flight.
        "delta__d": {"log": "four", "done": True},
    })
    # The rig's own state directory lives in runs/ and is not a run. Its logs/
    # child has no result.json, so scanning it offered a trial named "logs" --
    # invisible while only the newest trial was shown, because a directory with
    # no agent log sorts last and never won.
    state = runs / ".harness-arena" / "logs"
    state.mkdir(parents=True)

    found = find_active_trials(runs)
    names = [f["trial_dir"].name for f in found]
    check("every unfinished trial is found", sorted(names),
          ["alpha__a", "beta__b", "gamma__c"])
    check("a finished trial is not in flight", "delta__d" in names, False)
    check("the rig's own state dir is not a run", "logs" in names, False)
    # Most recently written first, so the default selection is the trial
    # actually producing output.
    check("ordered by most recent output", names[0], "beta__b")


def test_order_is_stable_across_calls(scratch: Path) -> None:
    """Two logs sharing an mtime must not swap places between polls.

    The tab strip is rebuilt from this list every five seconds. An order that
    depends on directory iteration moves tabs under the cursor.
    """
    runs = scratch / "runs"
    same = time.time() - 5
    make_run(runs, "job1", {
        "zeta__z": {"log": "x", "mtime": same},
        "alpha__a": {"log": "x", "mtime": same},
        "mid__m": {"log": "x", "mtime": same},
    })
    orders = {tuple(f["trial_dir"].name for f in find_active_trials(runs))
              for _ in range(5)}
    check("tab order is deterministic", len(orders), 1)
    check("...and tie-broken by name", list(orders)[0],
          ("alpha__a", "mid__m", "zeta__z"))


def test_selection_and_fallback(scratch: Path) -> None:
    runs = scratch / "runs"
    now = time.time()
    make_run(runs, "job1", {
        "alpha__a": {"log": "AAA", "mtime": now - 30},
        "beta__b": {"log": "BBB", "mtime": now - 10},
    })

    default = read_activity(runs)
    check("defaults to the newest trial", default["task"], "beta")
    check("lists both trials", len(default["trials"]), 2)
    check("nothing was asked for, so nothing went stale",
          default["selected_finished"], False)

    keys = {t["task"]: t["key"] for t in default["trials"]}
    picked = read_activity(runs, selected=keys["alpha"])
    check("an explicit selection is honoured", picked["task"], "alpha")
    check("...and is not reported as finished", picked["selected_finished"], False)
    check("the tab order does not depend on the selection",
          [t["key"] for t in picked["trials"]],
          [t["key"] for t in default["trials"]])

    # The case the flag exists for: what you pinned has finished, so the feed
    # legitimately shows something else.
    gone = read_activity(runs, selected="job1/nope__x")
    check("a vanished selection falls back", gone["task"], "beta")
    check("...and says that it did", gone["selected_finished"], True)


def test_only_the_selected_trial_pays_for_a_tail(scratch: Path) -> None:
    """Tab metadata is stat-only; entries are read for one trial per poll."""
    runs = scratch / "runs"
    now = time.time()
    make_run(runs, "job1", {
        "alpha__a": {"log": "hello from alpha", "mtime": now - 30},
        "beta__b": {"log": "hello from beta", "mtime": now - 10},
    })
    act = read_activity(runs)
    check("the selected trial carries its tail", bool(act["entries"]), True)
    check("the summaries carry none",
          [len(t["entries"]) for t in act["trials"]], [0, 0])
    # But they do carry what the tab needs to describe itself.
    beta = next(t for t in act["trials"] if t["task"] == "beta")
    check("a summary still reports its size", beta["log_bytes"] > 0, True)
    check("...and its phase", beta["phase"], "agent")
    setting_up = read_activity(runs)
    check("keys are qualified by run", setting_up["key"].startswith("job1/"), True)


def test_a_trial_that_has_not_started_reports_setting_up(scratch: Path) -> None:
    runs = scratch / "runs"
    make_run(runs, "job1", {"alpha__a": {}})
    act = read_activity(runs)
    check("no agent log means setting up", act["phase"], "setting up")
    # None, not zero: a zero claims the model has been working for no time
    # rather than that it has not started.
    check("and no agent clock at all", act["elapsed_s"], None)


def test_two_benchmarks_at_once_do_not_collide(scratch: Path) -> None:
    """Task names repeat across runs; the key has to distinguish them."""
    runs = scratch / "runs"
    now = time.time()
    make_run(runs, "job1", {"alpha__a": {"log": "one", "mtime": now - 20}})
    make_run(runs, "job2", {"alpha__z": {"log": "two", "mtime": now - 10}})
    act = read_activity(runs)
    keys = [t["key"] for t in act["trials"]]
    check("both runs' trials are listed", len(keys), 2)
    check("...under distinct keys", len(set(keys)), 2)
    check("the newer run is selected", act["run_id"], "job2")
    older = next(k for k in keys if k.startswith("job1/"))
    check("and the other is reachable", read_activity(runs, selected=older)["run_id"],
          "job1")


def test_no_runs_at_all(scratch: Path) -> None:
    check("a missing runs dir is not active",
          read_activity(scratch / "nope"), {"active": False})
    empty = scratch / "empty"
    empty.mkdir()
    check("an empty runs dir is not active",
          read_activity(empty), {"active": False})
    check("...and finds no trials", find_active_trials(empty), [])


def test_trial_key_is_run_qualified(scratch: Path) -> None:
    runs = scratch / "runs"
    make_run(runs, "job1", {"alpha__abc": {"log": "x"}})
    found = find_active_trials(runs)[0]
    check("key names run and trial", trial_key(found), "job1/alpha__abc")


if __name__ == "__main__":
    for test in (
        test_finds_every_running_trial_and_only_trials,
        test_order_is_stable_across_calls,
        test_selection_and_fallback,
        test_only_the_selected_trial_pays_for_a_tail,
        test_a_trial_that_has_not_started_reports_setting_up,
        test_two_benchmarks_at_once_do_not_collide,
        test_no_runs_at_all,
        test_trial_key_is_run_qualified,
    ):
        with tempfile.TemporaryDirectory() as scratch:
            test(Path(scratch))
    print("\n" + ("FAILED: " + ", ".join(failures) if failures else "all checks passed"))
    raise SystemExit(1 if failures else 0)
