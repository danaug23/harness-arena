"""Every machine-specific setting, in one place and out of version control.

This module exists because the rig used to hardcode one person's LAN address and
one person's conda path. Anything that differs between two people running this
repo belongs here, and nothing here is committed.

Precedence, lowest to highest::

    built-in defaults  ->  config.yaml  ->  HARNESS_ARENA_* env vars  ->  CLI flags

``config.yaml`` sits next to this repo's root and is gitignored;
``config.example.yaml`` is the committed template. ``HARNESS_ARENA_CONFIG``
overrides the location, which is what lets one checkout drive several endpoints.

Secrets
-------
An API key is the one setting that must never reach disk in this repo, a run
manifest, or a terminal. The supported path is *indirection*: config.yaml names
an environment variable (``api_key_env``), and the key itself lives in your
shell, your secret manager, or a ``.env`` file that is gitignored.

Storing the literal key in config.yaml is supported, because the setup UI has to
be able to write one somewhere and asking people to hand-edit a shell profile is
how you get keys pasted into registry.yaml instead. It is not the default, it
warns, and config.yaml is gitignored -- but treat that file as a credential once
you do it.

``Config.redacted()`` is what anything that logs, serializes, or renders must
call. There is no code path that writes a resolved key anywhere.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from bench import WORKSPACE

CONFIG_NAME = "config.yaml"
EXAMPLE_NAME = "config.example.yaml"

#: Marker substituted for a key in anything a human or a file might see.
#: Used only when the server will not say and nothing is configured. Chosen
#: low on purpose: overshooting the real window is the dangerous direction --
#: the server truncates silently and the run scores it as a reasoning failure
#: rather than a configuration error. Ollama's own modern default is the same
#: number, so a default Ollama install is not overshot by this.
DEFAULT_CONTEXT_WINDOW = 4096

REDACTED = "***redacted***"


class ConfigError(RuntimeError):
    """Configuration is missing or self-contradictory."""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    """How to talk to one flavor of OpenAI-compatible endpoint.

    Both providers speak the same wire protocol; they differ in what they can
    tell you about themselves and in whether concurrency is yours to spend.
    """

    id: str
    label: str
    #: Prefilled when the user picks this provider in the setup flow. Empty means
    #: "no sensible default, the user must supply one".
    default_base_url: str
    #: Environment variable consulted when config.yaml does not name one.
    default_api_key_env: str
    #: Whether a key is required for the endpoint to answer at all.
    requires_api_key: bool
    #: llama-server and friends expose /props, which reports the loaded weights
    #: and the slot count. Hosted aggregators do not, so the model must be named
    #: explicitly and identity has to be derived from the id instead.
    supports_props: bool
    #: Default cap on concurrent agent phases. A self-hosted single-slot server
    #: must stay at 1 or requests queue and latency stops meaning anything; a
    #: hosted API bills per token and has no local slot to contend for.
    default_agent_concurrency: int
    #: Whether the served model can be discovered rather than configured. A local
    #: server has exactly one model loaded; OpenRouter lists hundreds.
    model_is_discoverable: bool


PROVIDERS: dict[str, Provider] = {
    "openai-compatible": Provider(
        id="openai-compatible",
        label="OpenAI-compatible endpoint",
        default_base_url="http://localhost:8080/v1",
        default_api_key_env="OPENAI_API_KEY",
        requires_api_key=False,
        supports_props=True,
        default_agent_concurrency=1,
        model_is_discoverable=True,
    ),
    "openrouter": Provider(
        id="openrouter",
        label="OpenRouter",
        default_base_url="https://openrouter.ai/api/v1",
        default_api_key_env="OPENROUTER_API_KEY",
        requires_api_key=True,
        supports_props=False,
        default_agent_concurrency=4,
        model_is_discoverable=False,
    ),
}

DEFAULT_PROVIDER = "openai-compatible"


def provider_for(provider_id: str) -> Provider:
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise ConfigError(
            f"Unknown provider {provider_id!r}. Known providers: {known}."
        ) from None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EndpointConfig:
    """Which model server to benchmark against."""

    provider: str = DEFAULT_PROVIDER
    base_url: str = ""
    #: Name of the environment variable holding the API key. The supported way
    #: to supply a credential.
    api_key_env: str = ""
    #: A literal key. Written only by the setup UI; see the module docstring.
    api_key: str = ""
    #: Required for providers that cannot report which model is loaded.
    model: str = ""
    #: Overrides the auto-derived display label for this endpoint's model.
    label: str = ""
    #: The context window handed to every harness. 0 means "ask the server".
    #:
    #: Needed because not every server will say. llama.cpp reports the loaded
    #: window on /props; Ollama reports only the model's architectural maximum,
    #: which is not what the server was configured to use, and nothing in its
    #: API exposes the effective value. Rather than let each harness guess
    #: differently, set it once here.
    context_window: int = 0
    #: Reasoning effort handed to harnesses that send one. Empty means "ask the
    #: endpoint", which is almost always right: a server that cannot think
    #: refuses a real effort with a 400 and takes every trial down with it, so
    #: the value is measured rather than assumed. Set it only to pin a
    #: comparison to one specific effort. See bench.runner.
    reasoning_effort: str = ""

    def resolved_provider(self) -> Provider:
        return provider_for(self.provider)

    def resolved_base_url(self) -> str:
        return (self.base_url or self.resolved_provider().default_base_url).rstrip("/")

    def resolve_api_key(self) -> str:
        """The key to send, or "" if there is none.

        Checked in order: the literal key, the named environment variable, then
        the provider's conventional variable. The last one is what makes
        ``OPENROUTER_API_KEY`` in your shell work with no config at all.
        """
        if self.api_key:
            return self.api_key
        provider = self.resolved_provider()
        for name in (self.api_key_env, provider.default_api_key_env):
            if name and os.environ.get(name):
                return os.environ[name]
        return ""

    def validate(self) -> None:
        provider = self.resolved_provider()
        if not self.resolved_base_url():
            raise ConfigError(
                f"No base_url for provider {provider.id!r}. Set endpoint.base_url "
                f"in {CONFIG_NAME}."
            )
        if provider.requires_api_key and not self.resolve_api_key():
            names = " or ".join(
                dict.fromkeys(
                    n for n in (self.api_key_env, provider.default_api_key_env) if n
                )
            )
            raise ConfigError(
                f"{provider.label} requires an API key and none was found.\n"
                f"  Set {names} in your environment, or run the setup flow.\n"
                f"  Do not put the key in harnesses/registry.yaml -- that file is "
                f"committed."
            )
        if not provider.model_is_discoverable and not self.model:
            raise ConfigError(
                f"{provider.label} cannot report which model is loaded, so it has "
                f"to be named. Set endpoint.model in {CONFIG_NAME} "
                f"(e.g. 'qwen/qwen3-coder')."
            )


@dataclass
class DashboardConfig:
    """Where the UI listens.

    The default binds loopback deliberately. The dashboard reads run output and
    -- once the control plane lands -- starts and stops processes, so exposing it
    on a routable interface hands those capabilities to the network.
    """

    host: str = "127.0.0.1"
    port: int = 8420
    open_browser: bool = True


@dataclass
class Config:
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    #: Where Harbor job directories are written. Relative paths resolve against
    #: the repo root, so the default stays inside the checkout and gitignored.
    runs_dir: str = "runs"
    #: Recorded in each run manifest so a shared result set can say which machine
    #: produced it. Off by default: the old code wrote the real hostname into
    #: every manifest, and manifests are exactly what gets exported and shared.
    record_hostname: bool = False

    # -- paths ------------------------------------------------------------

    def resolved_runs_dir(self) -> Path:
        path = Path(self.runs_dir).expanduser()
        return path if path.is_absolute() else (WORKSPACE / path)

    # -- serialization ----------------------------------------------------

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if redact and data["endpoint"].get("api_key"):
            data["endpoint"]["api_key"] = REDACTED
        return data

    def redacted(self) -> dict[str, Any]:
        """Safe to log, serialize into a manifest, or hand to the browser."""
        return self.to_dict(redact=True)

    def save(self, path: Path | None = None) -> Path:
        """Write config.yaml. Used by the setup UI; safe to call by hand.

        Writes the *unredacted* key, because a config that cannot round-trip its
        own credential is a config that silently stops working on reload. The
        file is gitignored and this is the only writer.
        """
        target = path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(redact=False)
        text = yaml.dump(payload, default_flow_style=False, sort_keys=False)
        header = (
            "# harness-arena local configuration -- NOT committed (see .gitignore).\n"
            "# Treat this file as a credential if endpoint.api_key is set.\n"
            f"# Template: {EXAMPLE_NAME}\n\n"
        )
        target.write_text(header + text, encoding="utf-8")
        # A file that may hold a key should not be world-readable. No-op on
        # Windows, where the call succeeds but ACLs govern access.
        try:
            target.chmod(0o600)
        except OSError:
            pass
        return target


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def config_path() -> Path:
    override = os.environ.get("HARNESS_ARENA_CONFIG")
    return Path(override).expanduser() if override else (WORKSPACE / CONFIG_NAME)


def example_path() -> Path:
    return WORKSPACE / EXAMPLE_NAME


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


#: Environment overrides, mapped to their dotted config path. These exist so a
#: container or CI job can point the rig at an endpoint without writing a file.
ENV_OVERRIDES: dict[str, str] = {
    "HARNESS_ARENA_PROVIDER": "endpoint.provider",
    "HARNESS_ARENA_BASE_URL": "endpoint.base_url",
    "HARNESS_ARENA_API_KEY_ENV": "endpoint.api_key_env",
    "HARNESS_ARENA_MODEL": "endpoint.model",
    "HARNESS_ARENA_LABEL": "endpoint.label",
    "HARNESS_ARENA_RUNS_DIR": "runs_dir",
    "HARNESS_ARENA_HOST": "dashboard.host",
    "HARNESS_ARENA_PORT": "dashboard.port",
}

# Deliberately absent from ENV_OVERRIDES: a variable that sets the literal key.
# The key is supplied *by name* through api_key_env so that it is never one
# copy-paste away from a committed file or a `docker run` line in a README.


def _apply_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    head, _, tail = dotted.partition(".")
    if tail:
        section = data.setdefault(head, {})
        if isinstance(section, dict):
            section[tail] = value
    else:
        data[head] = value


def _from_env(data: dict[str, Any]) -> dict[str, Any]:
    for name, dotted in ENV_OVERRIDES.items():
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        _apply_dotted(data, dotted, raw)
    return data


def _build(data: dict[str, Any]) -> Config:
    endpoint_data = dict(data.get("endpoint") or {})
    dashboard_data = dict(data.get("dashboard") or {})

    known_endpoint = {f for f in EndpointConfig.__dataclass_fields__}
    known_dashboard = {f for f in DashboardConfig.__dataclass_fields__}
    unknown = (set(endpoint_data) - known_endpoint) | (
        set(dashboard_data) - known_dashboard
    )
    if unknown:
        # A typo'd key that is silently ignored is how someone spends an hour
        # wondering why their base_url did nothing.
        raise ConfigError(
            f"Unrecognized setting(s) in {CONFIG_NAME}: {', '.join(sorted(unknown))}"
        )

    endpoint = EndpointConfig(**{k: v for k, v in endpoint_data.items()})
    endpoint.provider = str(endpoint.provider or DEFAULT_PROVIDER)

    dashboard = DashboardConfig(**{k: v for k, v in dashboard_data.items()})
    dashboard.port = int(dashboard.port)
    if isinstance(dashboard.open_browser, str):
        dashboard.open_browser = _coerce_bool(dashboard.open_browser)

    record_hostname = data.get("record_hostname", False)
    if isinstance(record_hostname, str):
        record_hostname = _coerce_bool(record_hostname)

    return Config(
        endpoint=endpoint,
        dashboard=dashboard,
        runs_dir=str(data.get("runs_dir") or "runs"),
        record_hostname=bool(record_hostname),
    )


def load(path: Path | None = None) -> Config:
    """Read config.yaml (if present), then apply environment overrides.

    A missing file is not an error: the built-in defaults plus environment
    variables are a complete configuration for a local endpoint on the default
    port, which is what makes ``git clone && harness-arena probe`` work.
    """
    target = path or config_path()
    data: dict[str, Any] = {}
    if target.exists():
        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{target} is not valid YAML: {exc}") from exc
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigError(f"{target} must contain a YAML mapping.")
            data = loaded
    return _build(_from_env(data))


def load_validated(path: Path | None = None) -> Config:
    config = load(path)
    config.endpoint.validate()
    return config


def with_overrides(config: Config, **endpoint_overrides: Any) -> Config:
    """Apply CLI flags, the highest-precedence layer.

    ``None`` and ``""`` mean "not supplied" -- an unset argparse flag must not
    erase a configured value.
    """
    supplied = {k: v for k, v in endpoint_overrides.items() if v not in (None, "")}
    if not supplied:
        return config
    return replace(config, endpoint=replace(config.endpoint, **supplied))


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

_KEYISH = re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{8,})\b")


def looks_like_key(text: str) -> bool:
    """Whether a string carries something shaped like a credential.

    Used to refuse writes into committed files. Deliberately shape-based rather
    than exact-match: it has to catch a key belonging to someone else, pasted
    into a form, that this process has never seen.
    """
    return bool(_KEYISH.search(text or ""))


#: CSI sequences (color, cursor moves), OSC sequences (window titles), and the
#: two-character escapes. Agents color their output for a terminal; the live
#: feed is HTML, where an unhandled escape renders as literal "[0m[2m" noise
#: that buries the text it was decorating.
_ANSI = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"        # CSI  e.g. ESC[2m, ESC[?25l
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC  e.g. ESC]0;title BEL
    r"|\x1b[@-Z\\-_]"                 # single-character escapes
)


def strip_ansi(text: str) -> str:
    """Remove terminal control sequences from captured agent output.

    Not every harness offers a way to turn color off -- minion, for one, emits
    it unconditionally -- so the reader has to cope rather than the writer.

    Carriage returns get the same treatment for the same reason: a spinner
    rewrites one line by returning to column zero, and without collapsing that
    the feed shows every frame of the animation instead of the line's final
    state.
    """
    cleaned = _ANSI.sub("", text or "")
    cleaned = cleaned.replace("\r\n", "\n")
    if "\r" in cleaned:
        cleaned = "\n".join(line.rsplit("\r", 1)[-1] for line in cleaned.split("\n"))
    return cleaned


def scrub(text: str, config: Config | None = None) -> str:
    """Remove anything key-shaped from text bound for a log or the browser.

    Belt and braces over ``redacted()``: harness output is captured verbatim and
    a harness that echoes its own configuration would otherwise print the key
    into the live feed.
    """
    if config is not None:
        key = config.endpoint.resolve_api_key()
        if key and len(key) >= 8:
            text = text.replace(key, REDACTED)
    return _KEYISH.sub(REDACTED, text)


def describe(config: Config) -> str:
    endpoint = config.endpoint
    provider = endpoint.resolved_provider()
    key = endpoint.resolve_api_key()
    if key:
        source = "config.yaml" if endpoint.api_key else "environment"
        key_note = f"set ({source})"
    else:
        key_note = "none" + ("" if not provider.requires_api_key else "  [!] required")
    lines = [
        f"  provider     {provider.label}",
        f"  endpoint     {endpoint.resolved_base_url()}",
        f"  api key      {key_note}",
    ]
    if endpoint.model:
        lines.append(f"  model        {endpoint.model}")
    lines.append(f"  runs dir     {config.resolved_runs_dir()}")
    return "\n".join(lines)
