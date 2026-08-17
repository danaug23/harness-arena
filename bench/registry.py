"""Read and edit the harness catalog.

``harnesses/registry.yaml`` is the extension point of the whole rig and is the
one config file that *is* committed. That combination is why edits go through
here rather than straight to disk: the UI can write this file, and a credential
pasted into it would be committed and published.

So ``upsert_harness`` refuses anything key-shaped in a literal value. The
supported way to give a harness a credential is the ``{api_key}`` placeholder,
which is resolved at run time and scrubbed back out of manifests and logs.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from bench import (
    IS_CHECKOUT,
    PACKAGED_REGISTRY_PATH,
    USER_REGISTRY_PATH,
    __version__,
    registry_path,
)
from bench.config import looks_like_key

#: Harness ids become directory-name components, so keep them boring.
#: The catalog's preamble, kept here rather than only in the file because
#: `save()` rewrites registry.yaml from parsed YAML -- which drops every
#: comment in it. Anything worth telling the next reader has to live in this
#: constant to survive an edit made from the dashboard. harnesses/registry.yaml
#: begins with exactly this text; tests/test_local_agents.py checks that they
#: have not drifted, because a mismatch means every UI edit rewrites the header
#: and shows up as spurious churn in the diff.
HEADER = """\
# Harness catalog -- the extension point of this rig.
#
# NEVER put a credential in this file: it is committed. Use {api_key},
# which is resolved at run time and scrubbed out of manifests and logs.
#
# Placeholders: {model_id} {base_url} {base_url_root} {host} {n_ctx}
#               {max_tokens} {label} {api_key}
#
# `version:` pins the harness build a run installs, and every harness here is
# pinned on purpose. Unpinned, an install resolves whatever upstream's default
# branch holds at the moment each trial starts, which fails two ways: two runs
# a week apart measure different harnesses under one name, and a push *during*
# a run changes the harness between trials of that run. On 2026-08-13
# NousResearch/hermes-agent@6a198f8a1 made a failed `npm install` fatal 2h42m
# into an 89-task run -- the 31 trials before it installed cleanly, 28 of the
# 33 after it died. One run, two harnesses, one number.
#
# The ref is per-installer: git tag (hermes, omp), commit sha (minion), npm
# version (claude-code, codex, dsh), release version (opencode). Bumping one is
# deliberate -- re-run that harness before comparing across the change.
#
# `datasets:` is the benchmark catalog the dashboard offers in its dropdown.
# `id` is passed to `harbor run --dataset` verbatim, so it must be a dataset
# Harbor can resolve -- see hub.harborframework.com/datasets. `tasks` is
# display only. `slug` is the short name that becomes a segment of every run
# directory, so it is bounded at 12 characters: a run directory already carries
# a harness, a model and a timestamp, and Windows still enforces a 260-character
# path limit on the trial directories written underneath it. Omitting it derives
# one from the id, which works but is longer, less recognisable, and a truncation
# -- two ids sharing their first characters derive the same slug, and a catalog
# whose run directories would stop telling two benchmarks apart is refused on
# save whether the collision was written down or derived.
#
# The remaining fields are what pre-pull needs, and which one a dataset uses
# depends on how it ships its environments:
#
#   image_repo/image_tag  the dataset publishes one prebuilt image per task, so
#                         pre-pull fetches <repo>/<task>:<tag> for every task.
#                         Terminal-Bench 2 works this way.
#   base_images           the dataset builds each environment locally from a
#                         Dockerfile, so there is no per-task image to fetch.
#                         What is worth caching is the base layer those
#                         Dockerfiles start FROM. All 225 aider-polyglot tasks
#                         share buildpack-deps:jammy, so one pull covers the
#                         whole dataset.
#
# A dataset with neither can still be benchmarked; it just cannot be pre-pulled,
# and guessing would fetch the wrong images or none at all.
#
# `host_env` is for the benchmarks whose *own* machinery calls a model, rather
# than only the agent under test. Harbor reads a task's [environment].env and
# [verifier].env from its own environment and refuses to start when a required
# one is unset, so without this the run dies before its first trial. The values
# take the same {placeholders} as a harness block and never reach a manifest
# unredacted.
#
# tau3-bench is the case that exists today: its environment simulates the user
# and its verifier judges assertions in natural language, so both need an
# endpoint of their own. Its task.toml defaults both models to gpt-5.2, which
# means pointing only the URL at a local server asks that server for gpt-5.2 --
# so the model names are set too. A run whose user simulator and judge are the
# same small local model is NOT comparable to a published tau3-bench score,
# where both are frontier models; the manifest records what they were.
#
# Comments below this line do not survive an edit made from the dashboard:
# that path rewrites the file from parsed YAML and re-emits only this header.

"""

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

#: A dataset slug becomes one `__`-separated segment of every run directory
#: name, so it is bounded for the same reason harness ids are. 12 characters is
#: what is left once a harness id, a 57-character model slug, a scope and a
#: timestamp have taken their share and the trial directories Harbor writes
#: underneath still have to fit inside Windows' 260-character path limit.
DATASET_SLUG_MAX = 12
_DATASET_SLUG = re.compile(rf"^[a-z0-9][a-z0-9-]{{0,{DATASET_SLUG_MAX - 1}}}$")

#: Substitutions the runner fills in. Anything else in braces is a typo that
#: would otherwise fail at run time with a bare KeyError.
KNOWN_PLACEHOLDERS = {
    "model_id",
    "base_url",
    # base_url without a trailing "/v1", for harnesses whose SDK appends its
    # own version segment -- Anthropic's does, OpenAI's does not. See
    # bench.runner.base_url_root.
    "base_url_root",
    "host",
    "n_ctx",
    "max_tokens",
    # For harnesses that send a reasoning effort on every request. Resolved per
    # endpoint rather than fixed, because a server can refuse an effort
    # outright -- see bench.runner.effective_reasoning_effort.
    "reasoning_effort",
    "label",
    "api_key",
}

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


class RegistryError(ValueError):
    """A harness definition that would not work, or must not be written."""


def load(path: Path | str | None = None) -> dict[str, Any]:
    target = Path(path) if path else registry_path()
    with target.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RegistryError(f"{target} must contain a YAML mapping.")
    data.setdefault("harnesses", {})
    data.setdefault("defaults", {})
    return data


def datasets(data: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The catalog's dataset entries, skipping anything malformed."""
    catalog = data if data is not None else load()
    return [e for e in (catalog.get("datasets") or []) if isinstance(e, dict)]


def dataset_entry(dataset: str | None, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """The catalog entry for *dataset*, falling back to the default one.

    Returns a bare ``{"id": ...}`` for a dataset that is not catalogued, which
    is a supported state: ``--dataset`` accepts anything Harbor can resolve, and
    the catalog only has to know about the ones the dashboard offers.
    """
    catalog = data if data is not None else load()
    wanted = dataset or (catalog.get("defaults") or {}).get("dataset")
    for entry in datasets(catalog):
        if entry.get("id") == wanted:
            return entry
    return {"id": wanted}


def derive_dataset_slug(dataset: str) -> str:
    """A short, filesystem-safe name for a dataset that has no catalogued slug.

    Harbor dataset ids are ``org/name`` or ``name@version``, and the org is the
    least distinguishing part of one (``swe-bench/swe-bench-verified``), so the
    name is what survives. The version is dropped rather than encoded: it would
    cost most of the budget, and the manifest records the id in full.
    """
    name = str(dataset or "").split("@", 1)[0].rstrip("/")
    name = name.rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:DATASET_SLUG_MAX].rstrip("-") or "dataset"


def slug_of(entry: dict[str, Any]) -> str:
    """The slug this catalog entry actually contributes to a run directory name.

    Catalogued slugs win because they are chosen to be recognisable; a slug that
    would not survive a round trip through a directory name is ignored rather
    than written, since the name is the one place this fact is visible without
    opening a manifest. One function so that what `validate_dataset_slugs`
    checks and what `job_name` writes cannot drift apart.
    """
    declared = entry.get("slug")
    if isinstance(declared, str) and _DATASET_SLUG.match(declared):
        return declared
    return derive_dataset_slug(str(entry.get("id") or ""))


def dataset_slug(dataset: str | None, data: dict[str, Any] | None = None) -> str:
    """The run-directory segment naming which benchmark a run measured."""
    return slug_of(dataset_entry(dataset, data))


def validate_dataset_slugs(data: dict[str, Any]) -> None:
    """Refuse a catalog whose slugs would collide or would not fit in a path.

    Two datasets sharing a slug is the failure worth catching here: the run
    directories stop distinguishing them, which is exactly the mislabelling the
    slug exists to prevent, and nothing downstream would report it.

    Collisions are checked on the *effective* slug -- what `slug_of` returns --
    rather than on the declared one. An entry without a `slug:` still gets one,
    derived by truncating its id, and a truncation collides more readily than a
    chosen name rather than less: `terminal-bench@2.0` and
    `terminal-bench-pro/terminal-bench-pro` both derive `terminal-ben`.
    Checking only what was written down would leave the mislabelling this exists
    to prevent reachable through the one door nobody is watching.
    """
    seen: dict[str, tuple[str, bool]] = {}
    for entry in datasets(data):
        declared = entry.get("slug")
        if declared is not None and (
            not isinstance(declared, str) or not _DATASET_SLUG.match(declared)
        ):
            raise RegistryError(
                f"Dataset {entry.get('id')!r} has slug {declared!r}. Use "
                f"lowercase letters, digits and '-', starting with a letter or "
                f"digit, at most {DATASET_SLUG_MAX} characters -- it becomes a "
                f"segment of every run directory name."
            )
        effective = slug_of(entry)
        if effective in seen:
            other_id, other_declared = seen[effective]
            # The fix differs depending on where the slug came from, so say
            # which: a declared collision is a typo, a derived one is two ids
            # that happen to share their first characters and needs a `slug:`.
            remedy = (
                "Change one of them."
                if declared is not None and other_declared
                else "Give at least one an explicit `slug:` -- this one was "
                     "derived from the id."
            )
            raise RegistryError(
                f"Datasets {other_id!r} and {entry.get('id')!r} both use slug "
                f"{effective!r}. Run directories would stop telling them apart. "
                f"{remedy}"
            )
        seen[effective] = (str(entry.get("id")), declared is not None)


def save(data: dict[str, Any], path=None) -> None:
    """Write the catalog back, keeping one generation of backup.

    The UI can delete a harness, and the adapter notes in this file are the
    accumulated result of debugging each upstream's quirks -- worth one cheap
    undo.
    """
    # Checked on the way out rather than on the way in: every edit path funnels
    # through here, and a colliding slug is only a problem once it is on disk.
    validate_dataset_slugs(data)
    # Writes land on the copy you own. Installed from a wheel that is not the
    # packaged catalog, which lives in site-packages and is not yours to edit.
    target = Path(path) if path else USER_REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.copy2(target, target.with_suffix(".yaml.bak"))

    # Stamp which release this copy was forked from, so `catalog_drift` can
    # report the gap precisely instead of inferring it. Installed only: in a
    # checkout this file is committed, and an extra key here would be churn in
    # every diff. See `catalog_drift` for why the stamp matters.
    if not IS_CHECKOUT:
        data = {"snapshot_of": __version__, **data}

    body = yaml.dump(data, default_flow_style=False, sort_keys=False, width=88)
    target.write_text(HEADER + body, encoding="utf-8")


def catalog_drift() -> dict[str, Any]:
    """What a package upgrade is holding that your own catalog copy never got.

    An installed copy reads the packaged catalog until the first edit, at which
    point `save()` writes a full copy under `.harness-arena/` and
    `registry_path()` prefers that copy forever. From then on `pip install -U`
    updates the code and the *packaged* catalog, and nothing updates yours.

    That is silent, and it is not cosmetic. The catalog carries the harness
    `version:` pins, so an upgrade that re-pins a harness leaves you installing
    the old build under the new release's name -- which is precisely the "two
    runs a week apart measure different harnesses under one name" failure the
    pins exist to prevent. New `datasets:` entries go missing the same way.

    Nothing is merged automatically. A version pin you changed on purpose and
    one you simply never received are indistinguishable in the file, so guessing
    would either discard your edit or silently keep a stale harness. Reporting
    the difference lets you decide; `snapshot_of` says which release yours came
    from. Empty in a checkout, where there is only ever one catalog.
    """
    report: dict[str, Any] = {
        "applies": False,
        "user_path": str(USER_REGISTRY_PATH),
        "packaged_path": str(PACKAGED_REGISTRY_PATH),
        "snapshot_of": None,
        "package_version": __version__,
        "new_datasets": [],
        "new_harnesses": [],
        "version_changes": [],
        "stale": False,
    }
    if IS_CHECKOUT or not USER_REGISTRY_PATH.exists():
        return report
    if USER_REGISTRY_PATH.resolve() == PACKAGED_REGISTRY_PATH.resolve():
        return report

    try:
        mine = load(USER_REGISTRY_PATH)
        theirs = load(PACKAGED_REGISTRY_PATH)
    except (OSError, RegistryError, yaml.YAMLError):
        # A catalog too broken to read is a louder problem than drift, and it is
        # already reported by whatever tried to use it.
        return report

    report["applies"] = True
    report["snapshot_of"] = mine.get("snapshot_of")

    my_datasets = {str(d.get("id")) for d in datasets(mine)}
    report["new_datasets"] = [
        str(d.get("id")) for d in datasets(theirs) if str(d.get("id")) not in my_datasets
    ]

    my_harnesses = mine.get("harnesses") or {}
    their_harnesses = theirs.get("harnesses") or {}
    report["new_harnesses"] = [k for k in their_harnesses if k not in my_harnesses]
    for key, spec in their_harnesses.items():
        if key not in my_harnesses or not isinstance(spec, dict):
            continue
        theirs_version = spec.get("version")
        mine_version = (my_harnesses.get(key) or {}).get("version")
        if theirs_version != mine_version:
            report["version_changes"].append(
                {"harness": key, "yours": mine_version, "packaged": theirs_version}
            )

    report["stale"] = bool(
        report["new_datasets"] or report["new_harnesses"] or report["version_changes"]
    )
    return report


def _check_values(spec: Any, where: str = "") -> None:
    """Reject credentials and unknown placeholders anywhere in a harness block."""
    if isinstance(spec, dict):
        for key, value in spec.items():
            _check_values(value, f"{where}.{key}" if where else str(key))
        return
    if isinstance(spec, list):
        for index, value in enumerate(spec):
            _check_values(value, f"{where}[{index}]")
        return
    if not isinstance(spec, str):
        return

    if looks_like_key(spec):
        raise RegistryError(
            f"{where or 'value'} looks like an API key, and this file is "
            f"committed to source control.\n"
            f"Use the {{api_key}} placeholder instead -- it is resolved at run "
            f"time from your environment or gitignored config.yaml."
        )
    for name in _PLACEHOLDER.findall(spec):
        if name not in KNOWN_PLACEHOLDERS:
            known = ", ".join(sorted(KNOWN_PLACEHOLDERS))
            raise RegistryError(
                f"{where or 'value'} uses unknown placeholder {{{name}}}. "
                f"Available: {known}."
            )


def validate_harness(harness_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    if not _ID.match(harness_id or ""):
        raise RegistryError(
            f"Invalid harness id {harness_id!r}. Use lowercase letters, digits, "
            f"'-' and '_', starting with a letter or digit."
        )
    if not isinstance(spec, dict):
        raise RegistryError("A harness definition must be a mapping.")

    cleaned = {k: v for k, v in spec.items() if v not in (None, "", {}, [])}

    if not cleaned.get("agent"):
        raise RegistryError(
            "`agent` is required: a Harbor built-in agent name, or "
            "'module.path:ClassName' for an adapter in harnesses/."
        )
    cleaned.setdefault("label", harness_id)
    cleaned.setdefault("model_ref", "local/{model_id}")

    allowed = {
        "label", "vendor", "repo", "agent", "model_ref",
        "agent_kwargs", "agent_env", "host_env", "version",
        "min_context_window",
    }
    unknown = set(cleaned) - allowed
    if unknown:
        raise RegistryError(
            f"Unknown field(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )
    for field in ("agent_kwargs", "agent_env", "host_env"):
        if field in cleaned and not isinstance(cleaned[field], dict):
            raise RegistryError(f"`{field}` must be a mapping of key to value.")

    _check_values(cleaned, harness_id)
    return cleaned


def upsert_harness(harness_id: str, spec: dict[str, Any], path=None) -> dict[str, Any]:
    data = load(path)
    data["harnesses"][harness_id] = validate_harness(harness_id, spec)
    save(data, path)
    return data["harnesses"][harness_id]


def delete_harness(harness_id: str, path=None) -> None:
    data = load(path)
    if harness_id not in data.get("harnesses", {}):
        raise RegistryError(f"No harness named {harness_id!r}.")
    if len(data["harnesses"]) == 1:
        raise RegistryError(
            "That is the only harness defined. A catalog with none in it makes "
            "every run a no-op; add a replacement first."
        )
    del data["harnesses"][harness_id]
    save(data, path)


#: Defaults the UI is allowed to change, with the bounds each must stay inside.
#: Everything here alters what a run *measures*, so the bounds exist to stop the
#: UI writing a value that silently invalidates results.
EDITABLE_DEFAULTS: dict[str, tuple[type, float, float]] = {
    "n_concurrent": (int, 1, 32),
    "n_concurrent_agents": (int, 1, 32),
    "n_attempts": (int, 1, 10),
    "agent_timeout_multiplier": (float, 0.1, 100.0),
    "environment_build_timeout_multiplier": (float, 1.0, 50.0),
}


def update_defaults(updates: dict[str, Any], path=None) -> dict[str, Any]:
    data = load(path)
    defaults = data.setdefault("defaults", {})
    for key, raw in updates.items():
        if key == "dataset":
            if not isinstance(raw, str) or not raw.strip():
                raise RegistryError("`dataset` must be a non-empty string.")
            defaults["dataset"] = raw.strip()
            continue
        if key not in EDITABLE_DEFAULTS:
            raise RegistryError(f"`{key}` is not an editable default.")
        caster, low, high = EDITABLE_DEFAULTS[key]
        try:
            value = caster(raw)
        except (TypeError, ValueError):
            raise RegistryError(f"`{key}` must be {caster.__name__}.") from None
        if not low <= value <= high:
            raise RegistryError(f"`{key}` must be between {low} and {high}.")
        defaults[key] = value

    if defaults.get("n_concurrent_agents", 1) > defaults.get("n_concurrent", 1):
        # Harbor rejects this outright, and it is an easy slider mistake to make.
        raise RegistryError(
            "n_concurrent_agents cannot exceed n_concurrent -- an agent phase "
            "runs inside a trial, so there can never be more of them than there "
            "are trials in flight."
        )
    save(data, path)
    return defaults
