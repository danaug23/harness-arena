"""How often each harness ran into the output-token ceiling.

The ceiling is this rig's, not the harness's: ``agent_max_tokens_for(window)``
hands every harness that takes ``{max_tokens}`` the same cap, an eighth of the
window. A response that reaches it is cut off mid-sentence, and what happens
next is entirely the harness's business -- one absorbs the truncation and keeps
working, another ends the turn and stops. So "was it clipped" and "did that
cost anything" are different questions, and only the first one is measurable
from the logs.

The question this answers is comparative: **is one harness being clipped more
than its peers on the same model?** If it is, that harness's score is partly a
measure of the cap rather than of the harness, and the run is not the
experiment it looks like. If every harness is clipped at about the same rate,
the cap is uniform and the differences between them are real.

It was worth writing because the answer was counter-intuitive once already. dsh
lost a trial to a ``max-tokens`` turn end, which looked like dsh being unusually
talkative; measured, its median response was 499 tokens against omp's 219 and
opencode's 296, and all three topped out at exactly the cap on roughly one
response in a hundred. The cap was never the difference. What dsh *did* on
truncation was.

Only harnesses that report per-response usage can be read. A harness whose log
gives one total per trial is listed as unreadable rather than silently omitted,
because "not measured" and "measured, no clipping" are answers a reader must be
able to tell apart.

    harness-arena clipping
    harness-arena clipping --runs-dir runs --model 802849a6
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bench.runner import agent_max_tokens_for

#: Responses at or above this fraction of the cap are counted as near it. A
#: response that stops one token short was not clipped, but a population
#: crowding the ceiling is the same warning as one sitting on it.
_NEAR = 0.9


def _json_lines(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def _dsh(trial: Path) -> list[int]:
    """dsh: one usage chunk per step in the JSONL session log.

    Its ``outputTokens`` already includes reasoning, which is the number the
    cap actually applies to.
    """
    out: list[int] = []
    for log in trial.glob("agent/dsh/sessions/**/session.jsonl"):
        for event in _json_lines(log):
            if event.get("type") != "assistant/chunk":
                continue
            chunk = (event.get("data") or {}).get("chunk") or {}
            if chunk.get("type") != "usage":
                continue
            value = (chunk.get("usage") or {}).get("outputTokens")
            if isinstance(value, (int, float)):
                out.append(int(value))
    return out


def _opencode(trial: Path) -> list[int]:
    """opencode: `step_finish` events, with reasoning counted alongside output.

    Reasoning is generated under the same ceiling, so leaving it out would
    report a harness as comfortable while its responses were being cut off.
    """
    out: list[int] = []
    for event in _json_lines(trial / "agent" / "opencode.txt"):
        if str(event.get("type", "")).replace("-", "_") != "step_finish":
            continue
        tokens = (event.get("part") or event).get("tokens") or {}
        value, reasoning = tokens.get("output"), tokens.get("reasoning") or 0
        if isinstance(value, (int, float)):
            out.append(int(value) + int(reasoning))
    return out


def _omp(trial: Path) -> list[int]:
    out: list[int] = []
    for event in _json_lines(trial / "agent" / "omp.txt"):
        if event.get("type") != "message_end":
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        value = (message.get("usage") or {}).get("output")
        if isinstance(value, (int, float)):
            out.append(int(value))
    return out


#: harness id -> per-response reader. A harness absent here reports no
#: per-response usage this module knows how to read; see the module docstring
#: on why that is stated rather than skipped.
READERS = {"dsh": _dsh, "opencode": _opencode, "omp": _omp}


def scan(runs_dir: Path, model: str | None = None) -> dict[str, dict[str, Any]]:
    """Per-response output-token samples per harness, across every run."""
    samples: dict[str, list[int]] = {}
    trials: dict[str, int] = {}
    unreadable: set[str] = set()

    for job in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        # The runs directory is not only run directories: the rig keeps its own
        # `.harness-arena/logs` and `.active-run.json` beside them, and any tool
        # run with its cwd in here leaves state of its own. Reading a name like
        # `.omc` as a harness reported "no per-response usage in its log" for
        # something that was never a harness, which is a confident wrong answer
        # rather than a missing one. A run directory always carries the `__`
        # separator its name is built from.
        if job.name.startswith(".") or "__" not in job.name:
            continue
        harness = job.name.split("__")[0]
        if model and model not in job.name:
            continue
        reader = READERS.get(harness)
        if reader is None:
            unreadable.add(harness)
            continue
        for trial in (p for p in job.iterdir() if p.is_dir()):
            values = reader(trial)
            if values:
                samples.setdefault(harness, []).extend(values)
                trials[harness] = trials.get(harness, 0) + 1

    report: dict[str, dict[str, Any]] = {}
    for harness, values in samples.items():
        values.sort()
        report[harness] = {
            "trials": trials[harness],
            "responses": len(values),
            "median": int(statistics.median(values)),
            "p90": values[min(int(len(values) * 0.9), len(values) - 1)],
            "max": values[-1],
            "values": values,
        }
    for harness in unreadable - set(report):
        report[harness] = {"unreadable": True}
    return report


def render(report: dict[str, dict[str, Any]], cap: int) -> str:
    lines = [
        f"Output-token ceiling: {cap:,} per response "
        f"(agent_max_tokens_for, an eighth of the window).",
        "",
        f"{'harness':<12}{'trials':>7}{'resps':>7}{'median':>8}{'p90':>8}"
        f"{'max':>8}{'at cap':>8}{'near':>7}",
        "-" * 65,
    ]
    for harness, row in sorted(report.items()):
        if row.get("unreadable"):
            lines.append(f"{harness:<12}{'-- no per-response usage in its log --':>53}")
            continue
        values = row["values"]
        at_cap = sum(1 for v in values if v >= cap)
        near = sum(1 for v in values if v >= cap * _NEAR)
        lines.append(
            f"{harness:<12}{row['trials']:>7}{row['responses']:>7}"
            f"{row['median']:>8,}{row['p90']:>8,}{row['max']:>8,}"
            f"{at_cap:>8}{near:>7}"
        )

    readable = {h: r for h, r in report.items() if not r.get("unreadable")}
    if len(readable) > 1:
        rates = {
            h: sum(1 for v in r["values"] if v >= cap) / len(r["values"])
            for h, r in readable.items()
        }
        worst, best = max(rates, key=rates.get), min(rates, key=rates.get)
        lines += ["", f"most clipped: {worst} at {rates[worst]:.1%} of responses; "
                      f"least: {best} at {rates[best]:.1%}."]
        # A gap this wide means the cap is part of what the scores measure.
        if rates[worst] > 0.05 and rates[worst] > 3 * max(rates[best], 1e-9):
            lines.append(
                f"  {worst} is clipped disproportionately. Its score is partly a "
                f"measure of the ceiling rather than of the harness."
            )
        else:
            lines.append("  No harness stands out; the ceiling is being applied evenly.")
    lines += [
        "",
        "Clipping is not failure on its own -- what a harness does with a cut-off",
        "response is its own behaviour, and measuring that is the point of the run.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness-arena clipping",
        description="How often each harness hit the per-response output ceiling.",
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="Only runs whose directory name contains this (e.g. a fingerprint). "
        "Harnesses are only comparable on the same weights.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Window the cap is derived from. Default: the configured endpoint's.",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args(argv)

    from bench.config import load

    runs_dir = args.runs_dir or load().resolved_runs_dir()
    if not runs_dir.is_dir():
        print(f"No runs directory at {runs_dir}.")
        return 1

    window = args.context_window
    if window is None:
        window = int(getattr(load().endpoint, "context_window", 0) or 0)
    if window <= 0:
        print("No context window known. Pass --context-window.")
        return 1

    cap = agent_max_tokens_for(window)
    report = scan(runs_dir, args.model)
    if not report:
        print(f"No runs found in {runs_dir}.")
        return 0

    if args.json:
        print(json.dumps(
            {h: {k: v for k, v in r.items() if k != "values"} | {"cap": cap}
             for h, r in report.items()},
            indent=2,
        ))
    else:
        print(render(report, cap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
