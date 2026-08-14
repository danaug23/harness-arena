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

from bench import subset_path

# Terminal-Bench 2 publishes one prebuilt image per task under this namespace.
# Read off the images Harbor itself pulled, e.g. alexgshaw/gpt2-codegolf:20251031.
DEFAULT_REPO = "alexgshaw"
DEFAULT_TAG = "20251031"



def dataset_spec(dataset: str | None) -> dict:
    """The catalog entry for *dataset*, or the default one.

    Pre-pull needs to know how a dataset ships its environments, and only the
    catalog knows. Reading it here keeps the knowledge in one editable file
    instead of hard-coding a second dataset's layout into this module.
    """
    from bench import registry as registry_mod

    catalog = registry_mod.load()
    wanted = dataset or (catalog.get("defaults") or {}).get("dataset")
    for entry in catalog.get("datasets") or []:
        if isinstance(entry, dict) and entry.get("id") == wanted:
            return entry
    return {"id": wanted}


def images_for(spec: dict, tasks: list[str]) -> tuple[list[tuple[str, str]], str]:
    """(label, image) pairs to fetch for this dataset, plus what they are.

    Two shapes, because datasets ship environments two ways -- see the
    `datasets:` notes in registry.yaml. Returning the pairs rather than pulling
    here keeps the decision testable without Docker.
    """
    repo, tag = spec.get("image_repo"), spec.get("image_tag")
    if repo and tag:
        return [(t, f"{repo}/{t}:{tag}") for t in tasks], "task images"
    bases = [str(b) for b in (spec.get("base_images") or []) if b]
    if bases:
        # One shared layer per dataset, not one per task: the tasks build their
        # own environments FROM these, so the task list is irrelevant here.
        return [(b, b) for b in bases], "base images"
    return [], ""


def load_tasks(subset: str | None, tasks: list[str] | None) -> list[str]:
    if tasks:
        return tasks
    if not subset:
        raise SystemExit("Pass --subset NAME or --task NAME.")
    path = subset_path(subset)
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
    parser.add_argument(
        "--dataset",
        default=None,
        help="Which benchmark to cache for. Defaults to the catalog's "
        "defaults.dataset. How the images are found depends on the catalog "
        "entry; see the `datasets:` notes in registry.yaml.",
    )
    parser.add_argument("--repo", default=None,
                        help="Override the catalog's image_repo.")
    parser.add_argument("--tag", default=None,
                        help="Override the catalog's image_tag.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="Parallel pulls. Kept low by default: these are multi-GB layers and "
        "saturating the link makes every pull slower, not faster.",
    )
    parser.add_argument("--force", action="store_true", help="Re-pull cached images")
    args = parser.parse_args(argv)

    spec = dataset_spec(args.dataset)
    if args.repo:
        spec = {**spec, "image_repo": args.repo}
    if args.tag:
        spec = {**spec, "image_tag": args.tag}

    # Only a per-task dataset needs a task list. Asking for one when the images
    # are shared base layers would refuse a perfectly valid pre-pull.
    per_task = bool(spec.get("image_repo") and spec.get("image_tag"))
    tasks = load_tasks(args.subset, args.task) if per_task else []

    wanted, kind = images_for(spec, tasks)
    if not wanted:
        print(
            f"Nothing to pre-pull for {spec.get('id')!r}: the catalog has no "
            f"image_repo/image_tag and no base_images for it. "
            f"Its environments are presumably built during the run. Add the "
            f"fields to `datasets:` in registry.yaml if that is wrong -- "
            f"guessing them here would fetch the wrong images.",
            file=sys.stderr,
        )
        return 1

    have = local_images()
    todo = [(t, i) for t, i in wanted if args.force or i not in have]

    print(f"{spec.get('id')}: {len(wanted)} {kind}; "
          f"{len(wanted) - len(todo)} already cached.")
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
