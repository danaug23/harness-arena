"""Generate a deterministic, difficulty-stratified task subset.

A subset is only useful if every harness runs *exactly* the same tasks, and if
the mix of difficulties matches the full dataset -- otherwise the subset's pass
rate is not a shrunken version of the real one, it is a different measurement.

Selection is stride sampling within each difficulty band over alphabetically
sorted task names: reproducible, no RNG seed to remember, and not biased toward
whatever sorts first (taking the first N alphabetically would over-sample
whichever topics happen to start with 'a').

Usage (needs a checkout of the dataset repo):
    git clone --depth 1 https://github.com/laude-institute/terminal-bench-2
    python -m bench.make_subset --dataset-dir terminal-bench-2 --size 25
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from bench import USER_SUBSET_DIR

# New subsets are written where you can keep them: the checkout when there
# is one, otherwise your workspace rather than inside site-packages.
SUBSET_DIR = USER_SUBSET_DIR
BANDS = ("easy", "medium", "hard")


def read_difficulties(dataset_dir: Path) -> dict[str, str]:
    """Map task name -> difficulty, from each task's task.toml."""
    found: dict[str, str] = {}
    for toml_path in sorted(dataset_dir.glob("*/task.toml")):
        text = toml_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'^\s*difficulty\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if match:
            found[toml_path.parent.name] = match.group(1).strip().lower()
    return found


def stride_pick(names: list[str], count: int) -> list[str]:
    """Take `count` evenly spaced entries from `names`, endpoints included."""
    if count <= 0 or not names:
        return []
    if count >= len(names):
        return list(names)
    if count == 1:
        return [names[len(names) // 2]]
    step = (len(names) - 1) / (count - 1)
    return [names[round(i * step)] for i in range(count)]


def build(difficulties: dict[str, str], size: int) -> tuple[list[str], dict[str, tuple[int, int]]]:
    by_band: dict[str, list[str]] = defaultdict(list)
    for name, band in difficulties.items():
        by_band[band if band in BANDS else "medium"].append(name)
    for names in by_band.values():
        names.sort()

    total = sum(len(v) for v in by_band.values())

    # Largest-remainder apportionment, so the strata sum to exactly `size`
    # instead of drifting by a task or two through independent rounding.
    exact = {b: len(by_band[b]) * size / total for b in BANDS if by_band[b]}
    quota = {b: int(v) for b, v in exact.items()}
    # Every represented band keeps at least one task; a subset with zero easy
    # tasks is not stratified, it is just a hard-task benchmark.
    for band in quota:
        quota[band] = max(1, quota[band])
    while sum(quota.values()) > size:
        band = max(quota, key=lambda b: (quota[b] - exact[b], quota[b]))
        if quota[band] > 1:
            quota[band] -= 1
        else:
            break
    while sum(quota.values()) < size:
        band = max(quota, key=lambda b: exact[b] - quota[b])
        quota[band] += 1

    picked: list[str] = []
    report: dict[str, tuple[int, int]] = {}
    for band in BANDS:
        if band not in quota:
            continue
        chosen = stride_pick(by_band[band], quota[band])
        picked.extend(chosen)
        report[band] = (len(chosen), len(by_band[band]))
    return sorted(picked), report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=25)
    parser.add_argument("--name", default=None, help="Subset name (default: stratified-<size>)")
    parser.add_argument("--commit", default="", help="Dataset commit, recorded for provenance")
    args = parser.parse_args(argv)

    difficulties = read_difficulties(args.dataset_dir)
    if not difficulties:
        parser.error(f"No */task.toml with a difficulty found under {args.dataset_dir}")

    picked, report = build(difficulties, args.size)
    name = args.name or f"stratified-{len(picked)}"

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBSET_DIR / f"{name}.txt"
    header = [
        f"# {name}: {len(picked)} of {len(difficulties)} Terminal-Bench 2 tasks",
        "# Difficulty-stratified, stride-sampled over sorted names. Deterministic:",
        "# regenerating from the same dataset yields the same list.",
        f"# dataset commit: {args.commit or 'unrecorded'}",
        "#",
    ]
    for band in BANDS:
        if band in report:
            take, have = report[band]
            share = 100 * have / len(difficulties)
            header.append(f"#   {band:<7} {take:>3} of {have:>3}  ({share:.1f}% of the full set)")
    out.write_text("\n".join(header) + "\n" + "\n".join(picked) + "\n", encoding="utf-8")

    print("\n".join(header))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
