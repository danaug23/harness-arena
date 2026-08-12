"""Pre-pull every task image before a run starts.

Harbor pulls a task's image inside that trial's environment-start budget. Task
images are not small -- mteb-retrieve is 21.6 GB -- and a cold pull can exceed
the 600s default, at which point the trial dies with EnvironmentStartTimeoutError
and the task is lost rather than scored. That is a measurement hole, not just
slowness: it happened on the first stratified-25 run.

Pulling up front also moves the cost out of the benchmark entirely. Images are
cached by Docker, so this is paid once per machine, not once per run.

    python -m bench.prepull --subset stratified-25
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from bench import ROOT

# Terminal-Bench 2 publishes one prebuilt image per task under this namespace.
# Read off the images Harbor itself pulled, e.g. alexgshaw/gpt2-codegolf:20251031.
DEFAULT_REPO = "alexgshaw"
DEFAULT_TAG = "20251031"
SUBSET_DIR = ROOT / "bench" / "subsets"


def load_tasks(subset: str | None, tasks: list[str] | None) -> list[str]:
    if tasks:
        return tasks
    if not subset:
        raise SystemExit("Pass --subset NAME or --task NAME.")
    path = SUBSET_DIR / f"{subset}.txt"
    if not path.exists():
        raise SystemExit(f"No subset at {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def local_images() -> set[str]:
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def pull(image: str) -> tuple[str, bool, float, str]:
    started = time.monotonic()
    result = subprocess.run(
        ["docker", "pull", image],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.monotonic() - started
    if result.returncode == 0:
        return image, True, elapsed, ""
    tail = (result.stderr or result.stdout or "").strip().splitlines()
    return image, False, elapsed, tail[-1] if tail else "unknown error"


def image_size(image: str) -> str:
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Size}}", image],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "?"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="Parallel pulls. Kept low by default: these are multi-GB layers and "
        "saturating the link makes every pull slower, not faster.",
    )
    parser.add_argument("--force", action="store_true", help="Re-pull cached images")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.subset, args.task)
    have = local_images()
    wanted = [(t, f"{args.repo}/{t}:{args.tag}") for t in tasks]
    todo = [(t, i) for t, i in wanted if args.force or i not in have]

    print(f"{len(wanted)} task images; {len(wanted) - len(todo)} already cached.")
    if not todo:
        print("Nothing to pull.")
        return 0

    print(f"Pulling {len(todo)} with {args.jobs} workers...\n")
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for image, ok, elapsed, err in pool.map(lambda p: pull(p[1]), todo):
            name = image.split("/", 1)[-1]
            if ok:
                print(f"  ok    {name:<48} {elapsed:6.1f}s  {image_size(image)}")
            else:
                print(f"  FAIL  {name:<48} {elapsed:6.1f}s  {err}", file=sys.stderr)
                failures.append((image, err))

    print(f"\n{len(todo) - len(failures)}/{len(todo)} pulled.")
    if failures:
        print("\nFailed (these tasks will pull during the run, or error out):")
        for image, err in failures:
            print(f"  {image}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
