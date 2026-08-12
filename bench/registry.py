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

from bench import REGISTRY_PATH
from bench.config import looks_like_key

#: Harness ids become directory-name components, so keep them boring.
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

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
    target = Path(path) if path else REGISTRY_PATH
    with target.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RegistryError(f"{target} must contain a YAML mapping.")
    data.setdefault("harnesses", {})
    data.setdefault("defaults", {})
    return data


def save(data: dict[str, Any], path=None) -> None:
    """Write the catalog back, keeping one generation of backup.

    The UI can delete a harness, and the adapter notes in this file are the
    accumulated result of debugging each upstream's quirks -- worth one cheap
    undo.
    """
    target = Path(path) if path else REGISTRY_PATH
    if target.exists():
        shutil.copy2(target, target.with_suffix(".yaml.bak"))
    header = (
        "# Harness catalog -- the extension point of this rig.\n"
        "#\n"
        "# NEVER put a credential in this file: it is committed. Use {api_key},\n"
        "# which is resolved at run time and scrubbed out of manifests and logs.\n"
        "#\n"
        "# Placeholders: {model_id} {base_url} {base_url_root} {host} {n_ctx}\n"
        "#               {max_tokens} {label} {api_key}\n\n"
    )
    body = yaml.dump(data, default_flow_style=False, sort_keys=False, width=88)
    target.write_text(header + body, encoding="utf-8")


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
