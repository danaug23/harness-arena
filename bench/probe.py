"""Identify the model currently behind the configured endpoint.

The experiment only means anything if the model is held constant *and named
exactly*, so this module's job is to make a run impossible to mislabel.

How that is done depends on what the endpoint can tell you about itself:

**Self-hosted (llama-server, vLLM, LM Studio, ...)** advertise a generic alias
like ``my-model`` and nothing else, but expose ``/props``, which reports the
actual weights. Reload the same alias with a different quant and the alias does
not change -- so we fingerprint the *reported weights* (id, parameter count,
file size, ftype, trained context, path) and key the label cache on that. A new
quant produces a new fingerprint and therefore asks for a new label.

**Hosted aggregators (OpenRouter)** have no ``/props`` and serve hundreds of
models, so there is nothing to discover: the model is named in config and the
fingerprint is derived from that name. The weights behind a hosted model id can
change without notice, which is worth knowing when you compare across weeks --
a hosted run is a measurement of an endpoint, not of a file you hold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bench import MODELS_CACHE_PATH
from bench.config import (
    PROVIDERS,
    Config,
    ConfigError,
    EndpointConfig,
    load,
    with_overrides,
)


@dataclass
class ModelIdentity:
    """Everything we know about the model behind an endpoint."""

    served_id: str
    fingerprint: str
    label: str
    base_url: str
    host: str
    n_ctx: int
    provider: str = "openai-compatible"
    #: Whether the endpoint accepts a real ``reasoning.effort``. None means the
    #: question was never answered -- see supports_reasoning_effort.
    supports_reasoning: bool | None = None
    n_ctx_train: int | None = None
    n_params: int | None = None
    size_bytes: int | None = None
    ftype: str | None = None
    model_path: str | None = None
    build_info: str | None = None
    total_slots: int | None = None
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Filesystem-safe short name used in run directory names."""
        source = self.label or self.served_id or "model"
        base = re.sub(r"[^a-zA-Z0-9]+", "-", source).strip("-").lower()
        # A label of only punctuation would otherwise produce a run directory
        # that leads with a dash, which reads as a flag to half of CLI tooling.
        return f"{base[:48] or 'model'}-{self.fingerprint[:8]}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["slug"] = self.slug
        return data


def _get_json(
    url: str, timeout: float = 15.0, api_key: str = ""
) -> dict[str, Any] | None:
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    # OpenRouter attributes requests by these headers and rejects some clients
    # without a User-Agent. Harmless everywhere else.
    request.add_header("User-Agent", "harness-arena")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


def _human_params(n_params: int | None) -> str:
    if not n_params:
        return "?"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if n_params >= scale:
            return f"{n_params / scale:.1f}{suffix}"
    return str(n_params)


def _human_bytes(size: int | None) -> str:
    if not size:
        return "?"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _suggest_label(served_id: str, model_path: str | None, ftype: str | None) -> str:
    """Best-effort display name, derived from the weights filename when available.

    A self-hosted server reports the full weights path, which is almost always
    more descriptive than the alias -- e.g. ``.../Qwen3-Coder-30B-Q4_K_M.gguf``
    beats ``my-model``.
    """
    stem = ""
    if model_path:
        stem = Path(model_path.replace("\\", "/")).name
        stem = re.sub(r"\.gguf$", "", stem, flags=re.IGNORECASE)
        # Drop multi-part suffixes like -00001-of-00009.
        stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
        stem = stem.replace("_", " ").replace("-", " ").strip()
    if not stem:
        stem = served_id
    if ftype and ftype.split()[0].lower() not in stem.lower():
        stem = f"{stem} ({ftype})"
    return stem


def _host_of(url: str) -> str:
    return re.sub(r"^https?://", "", url).split("/")[0].split(":")[0]


def _probe_self_hosted(
    endpoint: EndpointConfig, timeout: float
) -> ModelIdentity:
    root = endpoint.resolved_base_url()
    api_key = endpoint.resolve_api_key()
    api_root = root[: -len("/v1")] if root.endswith("/v1") else root

    models = _get_json(f"{root}/models", timeout, api_key) or _get_json(
        f"{api_root}/v1/models", timeout, api_key
    )
    if not models:
        raise RuntimeError(
            f"No response from {root}/models -- is a model server running at "
            f"{root}?\n"
            f"  Check the endpoint with:  curl {root}/models"
        )

    entries = models.get("data") or []
    if not entries:
        raise RuntimeError(f"{root}/models returned no models.")

    # A server can host more than one model; honor an explicit choice if the
    # config names one, otherwise take the only (or first) one it lists.
    entry = entries[0]
    if endpoint.model:
        match = next((e for e in entries if e.get("id") == endpoint.model), None)
        if match is None:
            available = ", ".join(str(e.get("id")) for e in entries[:10])
            raise RuntimeError(
                f"{root} does not serve {endpoint.model!r}. Available: {available}"
            )
        entry = match
    meta = entry.get("meta") or {}

    props = _get_json(f"{api_root}/props", timeout, api_key) or {}
    default_settings = props.get("default_generation_settings") or {}

    served_id = str(entry.get("id") or props.get("model_alias") or "unknown")
    n_ctx = int(default_settings.get("n_ctx") or meta.get("n_ctx") or 0)
    n_ctx_train = meta.get("n_ctx_train")
    n_params = meta.get("n_params")
    size_bytes = meta.get("size")
    ftype = meta.get("ftype") or props.get("model_ftype")

    # Fingerprint the weights, not the alias. Anything that changes when you
    # load a different file changes this; restarting the same file does not.
    material = "|".join(
        str(part)
        for part in (
            served_id,
            n_params,
            size_bytes,
            ftype,
            n_ctx_train,
            meta.get("n_embd"),
            meta.get("n_vocab"),
            props.get("model_path"),
        )
    )
    fingerprint = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    return ModelIdentity(
        served_id=served_id,
        fingerprint=fingerprint,
        label="",  # resolved separately
        base_url=root,
        host=_host_of(root),
        provider=endpoint.provider,
        n_ctx=n_ctx or int(n_ctx_train or 0),
        n_ctx_train=n_ctx_train,
        n_params=n_params,
        size_bytes=size_bytes,
        ftype=ftype,
        model_path=props.get("model_path"),
        build_info=props.get("build_info"),
        total_slots=props.get("total_slots"),
        raw={"meta": meta, "props_alias": props.get("model_alias")},
    )


def _probe_hosted(endpoint: EndpointConfig, timeout: float) -> ModelIdentity:
    """Confirm a hosted provider actually serves the configured model id.

    There is no discovery here -- the point is to fail now, with the list of
    near-misses, rather than after Harbor has built a container and the first
    trial dies on a 404 that reads like a network problem.
    """
    root = endpoint.resolved_base_url()
    api_key = endpoint.resolve_api_key()
    wanted = endpoint.model

    catalog = _get_json(f"{root}/models", timeout, api_key)
    if not catalog:
        raise RuntimeError(
            f"No response from {root}/models. Check the endpoint and that the "
            f"API key is valid."
        )

    entries = [e for e in (catalog.get("data") or []) if isinstance(e, dict)]
    entry = next((e for e in entries if e.get("id") == wanted), None)
    if entry is None:
        # Suggest what they probably meant rather than dumping 300 model ids.
        stem = wanted.split("/")[-1].lower()
        near = [
            str(e.get("id"))
            for e in entries
            if stem[:6] and stem[:6] in str(e.get("id", "")).lower()
        ][:8]
        hint = f"\n  Did you mean: {', '.join(near)}" if near else ""
        raise RuntimeError(
            f"{root} does not serve {wanted!r} ({len(entries)} models available)."
            f"{hint}"
        )

    top = entry.get("top_provider") or {}
    n_ctx = int(top.get("context_length") or entry.get("context_length") or 0)

    # The weights behind a hosted id are not ours to inspect and can change
    # without notice, so the id is all the identity there is. Including the
    # provider keeps a hosted run from colliding with a local run of a model
    # that happens to share a name.
    material = f"{endpoint.provider}|{wanted}"
    fingerprint = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    return ModelIdentity(
        served_id=wanted,
        fingerprint=fingerprint,
        label="",
        base_url=root,
        host=_host_of(root),
        provider=endpoint.provider,
        n_ctx=n_ctx,
        n_ctx_train=entry.get("context_length"),
        ftype=None,
        model_path=None,
        build_info=str(entry.get("name") or ""),
        # Hosted APIs have no local slot to contend for; concurrency is a rate
        # limit, not a queue, so there is no slot count to report.
        total_slots=None,
        raw={"name": entry.get("name"), "pricing": entry.get("pricing")},
    )


def list_models(endpoint: EndpointConfig, timeout: float = 15.0) -> dict[str, Any]:
    """What the endpoint can serve, *without* requiring a model to be chosen.

    Deliberately separate from ``probe()``, which refuses to run until a hosted
    provider has been told which model to use. That would be circular here: you
    cannot choose from a list you cannot fetch. So this validates only what it
    needs -- a URL, and a key if the provider demands one.

    Returns the ids plus a ``default``: for a self-hosted server that is the one
    model actually loaded, which is what should be pre-selected.
    """
    root = endpoint.resolved_base_url()
    if not root:
        raise RuntimeError("No endpoint URL is set.")

    provider = endpoint.resolved_provider()
    api_key = endpoint.resolve_api_key()
    if provider.requires_api_key and not api_key:
        raise RuntimeError(
            f"{provider.label} needs an API key before it will list models."
        )

    catalog = _get_json(f"{root}/models", timeout, api_key)
    if not catalog:
        raise RuntimeError(
            f"No response from {root}/models -- is a model server running there?"
        )

    models: list[dict[str, Any]] = []
    for entry in catalog.get("data") or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        top = entry.get("top_provider") or {}
        models.append(
            {
                "id": str(entry["id"]),
                "name": str(entry.get("name") or entry["id"]),
                "context_length": top.get("context_length")
                or entry.get("context_length")
                or (entry.get("meta") or {}).get("n_ctx_train"),
            }
        )
    if not models:
        raise RuntimeError(f"{root}/models returned no models.")

    # A self-hosted server has exactly one model loaded, so the list *is* the
    # answer. A hosted aggregator lists hundreds and has no notion of "yours",
    # so there is nothing sensible to preselect.
    default = models[0]["id"] if provider.model_is_discoverable else ""
    if provider.supports_props:
        api_root = root[: -len("/v1")] if root.endswith("/v1") else root
        alias = (_get_json(f"{api_root}/props", timeout, api_key) or {}).get(
            "model_alias"
        )
        if alias and any(m["id"] == alias for m in models):
            default = alias

    models.sort(key=lambda m: m["id"].lower())
    return {
        "models": models,
        "default": default,
        "discoverable": provider.model_is_discoverable,
    }


def suggest_label(identity: ModelIdentity) -> str:
    """The display name derived from the weights, ignoring any override."""
    return _suggest_label(identity.served_id, identity.model_path, identity.ftype)


def cached_label(fingerprint: str) -> str:
    """A name previously pinned to these weights, if any."""
    return (_load_cache(MODELS_CACHE_PATH).get(fingerprint) or {}).get("label", "")


def effective_label(endpoint: EndpointConfig, identity: ModelIdentity) -> str:
    """The name a run started right now would actually be recorded under.

    Same precedence ``resolve()`` uses: an explicit override wins, then a name
    pinned to these weights, then the name derived from the weights filename.
    Worth surfacing, because "leave it empty" does not mean "derive it" once
    something has been pinned -- and the difference only shows up later, in the
    labels on your results.
    """
    return endpoint.label or cached_label(identity.fingerprint) or suggest_label(identity)


def remember_label(endpoint: EndpointConfig, label: str) -> str | None:
    """Pin a display label to the weights currently served.

    The label cache is keyed by weights fingerprint, so a name chosen in the UI
    survives a server restart and reappears for the same weights later. Failure
    is not worth surfacing -- the label is already saved in config either way.
    """
    if not label:
        return None
    try:
        identity = probe(endpoint)
    except (ConfigError, RuntimeError):
        return None
    cache = _load_cache(MODELS_CACHE_PATH)
    entry = dict(cache.get(identity.fingerprint) or {})
    entry.update(
        {
            "label": label,
            "served_id": identity.served_id,
            "model_path": identity.model_path,
            "ftype": identity.ftype,
            "n_params": identity.n_params,
        }
    )
    entry.setdefault("notes", "")
    cache[identity.fingerprint] = entry
    _save_cache(MODELS_CACHE_PATH, cache)
    return identity.fingerprint


def probe(endpoint: EndpointConfig, timeout: float = 15.0) -> ModelIdentity:
    """Query the endpoint and build a ModelIdentity (label not yet resolved)."""
    endpoint.validate()
    provider = endpoint.resolved_provider()
    if provider.supports_props:
        return _probe_self_hosted(endpoint, timeout)
    return _probe_hosted(endpoint, timeout)


def _post_responses(
    endpoint: EndpointConfig, url: str, body: dict[str, Any], timeout: float
) -> int:
    """POST to a /responses URL and return the status, or 0 if it never landed."""
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("User-Agent", "harness-arena")
    api_key = endpoint.resolve_api_key()
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status
    except urllib.error.HTTPError as exc:
        exc.read()
        return exc.code
    except (urllib.error.URLError, OSError):
        return 0


#: The effort the probe asks for. Deliberately the same value a run sends once
#: the answer comes back True: probing one effort and shipping another would
#: leave the only case that matters untested. It cannot be imported from
#: bench.runner, which imports this module, so a test keeps the two in step.
PROBE_EFFORT = "high"


def supports_reasoning_effort(
    endpoint: EndpointConfig,
    timeout: float = 90.0,
    *,
    served_id: str = "",
) -> bool | None:
    """Whether this endpoint accepts a real ``reasoning.effort`` on /responses.

    Codex is the harness that cares, because it always sends a ``reasoning``
    object and the effort inside it is not negotiable from the outside -- but
    the answer is a property of the *server*, not of Codex, so it is measured
    here with the rest of what the endpoint will tell us about itself.

    Servers genuinely disagree, and the disagreement is invisible until a run
    is already burning: llama.cpp accepts an effort against any model, while
    Ollama refuses one for a model that cannot think ("does not support
    thinking", HTTP 400) and every trial then dies at its first request with a
    bare non-zero exit code. Measured against ollama 0.32.5 and codex-cli
    0.147.0, not read off the docs.

    Returns True if an effort is accepted, False if the server specifically
    refuses it, and None when the question could not be answered -- no
    /responses route, an unreachable endpoint, or a rejection that turns out
    not to be about reasoning at all. None is not a failure: it means "leave
    the harness default alone", so an endpoint this probe cannot interrogate
    behaves exactly as it did before the probe existed.

    Nothing here raises. ``resolve()`` calls this on the run path, where an
    exception would abort the very runs the None fallback exists to let
    through untouched.
    """
    model = served_id or endpoint.model
    if not model:
        # Naming the model is part of asking the question, not part of the
        # answer -- and an endpoint that cannot be asked for its model id
        # cannot be asked about reasoning either.
        try:
            model = _served_id_for_speed(endpoint)
        except (ConfigError, RuntimeError):
            return None

    # 16 is the smallest ceiling the Responses API accepts, so this asks for as
    # little generation as the wire allows.
    body: dict[str, Any] = {"model": model, "input": "hi", "max_output_tokens": 16}
    asking = {**body, "reasoning": {"effort": PROBE_EFFORT}}

    root = endpoint.resolved_base_url()
    url = f"{root}/responses"
    status = _post_responses(endpoint, url, asking, timeout)
    if status == 404 and not root.endswith("/v1"):
        # base_url may be written with or without the /v1 suffix -- the model
        # lookup above accepts both, so this has to as well. Ollama serves the
        # OpenAI routes only under /v1, and a 404 here would otherwise read as
        # "no answer" and hand back the very default that breaks it. Tried only
        # on a 404, so the ordinary case still costs one request.
        url = f"{root}/v1/responses"
        status = _post_responses(endpoint, url, asking, timeout)
    if status == 200:
        return True
    if status == 0:
        # Nothing landed: unreachable, or too slow to answer. Not a fact about
        # reasoning.
        return None

    # Any other rejection only means something once the same request *without*
    # the reasoning object is known to succeed. Otherwise a wrong model id or a
    # missing credential would read as "this server hates reasoning" and
    # quietly downgrade every future run against it.
    #
    # The control request, not the status code, is what makes that safe -- so
    # every rejection is put through it rather than a chosen few. Servers do
    # not agree on how to spell "I will not do that": 400 and 422 are both
    # common, and a server that answers 500 to an unsupported parameter would
    # otherwise fall through to the fallback and take a whole run with it.
    #
    # Sent to the same URL the question was asked at, or the two answers would
    # not be about the same route.
    return False if _post_responses(endpoint, url, body, timeout) == 200 else None


def measure_speed(
    endpoint: EndpointConfig, *, max_tokens: int = 256, timeout: float = 600.0
) -> dict[str, Any]:
    """Measure uncontended output tokens/sec, and recommend a timeout multiplier.

    The single most consequential setting in this rig is
    ``agent_timeout_multiplier``, and it is pure arithmetic given your
    generation speed: Terminal-Bench budgets a task at 900-1800s assuming
    frontier-API throughput, while a trajectory spends 30k-150k output tokens.
    Set it too low and tasks time out half-finished, which scores identically to
    being wrong -- so the benchmark measures your hardware instead of the
    harness. Guessing that number is how people get a meaningless run; measuring
    it takes about a minute.
    """
    root = endpoint.resolved_base_url()
    api_key = endpoint.resolve_api_key()

    payload = json.dumps(
        {
            "model": endpoint.model or _served_id_for_speed(endpoint),
            "messages": [
                {
                    "role": "user",
                    "content": "Count from 1 to 120, one number per line. "
                    "No commentary.",
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{root}/chat/completions", data=payload, method="POST"
    )
    request.add_header("Content-Type", "application/json")
    request.add_header("User-Agent", "harness-arena")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Speed probe failed against {root}: {exc}") from exc
    elapsed = time.perf_counter() - started

    usage = body.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        raise RuntimeError(
            "The endpoint returned no usage.completion_tokens, so speed cannot "
            "be measured. Set agent_timeout_multiplier by hand."
        )

    tokens_per_s = completion_tokens / elapsed if elapsed > 0 else 0.0

    # Sized against the *upper* end of observed trajectories (~150k output
    # tokens), not the median, with 1.5x headroom for the non-generation parts
    # of a trial.
    #
    # The asymmetry is deliberate. Over-budgeting costs wall clock on tasks the
    # agent was going to abandon anyway; under-budgeting silently converts the
    # benchmark from a capability measurement into a throughput measurement,
    # because a timeout scores exactly like a wrong answer. A mid-trajectory
    # reference recommends 8.0 at ~29 tok/s, and 8.0 has been observed to be too
    # low at that speed -- individual tasks burned the full budget and still
    # timed out.
    reference_tokens = 150_000
    base_budget_s = 900.0
    needed = (reference_tokens / tokens_per_s) * 1.5 if tokens_per_s else 0.0
    raw = needed / base_budget_s
    # Round up to a value someone would actually type, so two people with
    # similar hardware land on the same multiplier and stay comparable.
    ladder = [1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0]
    recommended = next((step for step in ladder if step >= raw), ladder[-1])

    return {
        "tokens_per_s": tokens_per_s,
        "completion_tokens": completion_tokens,
        "elapsed_s": elapsed,
        "recommended_multiplier": recommended,
        "note": (
            "A floor, not a target: this covers generation only. The agent "
            "budget also absorbs prompt processing on every turn and the wall "
            "clock of the commands the agent runs, so doubling this is a "
            "reasonable starting point. Measured with one uncontended request "
            "-- concurrency above the server's slot count reduces it sharply."
        ),
    }


def _served_id_for_speed(endpoint: EndpointConfig) -> str:
    """The model id to name in a completions request when config does not."""
    identity = probe(endpoint)
    return identity.served_id


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def describe(identity: ModelIdentity) -> str:
    lines = [
        f"  endpoint     {identity.base_url}",
        f"  provider     {identity.provider}",
        f"  served id    {identity.served_id}",
    ]
    if identity.model_path:
        lines.append(f"  weights      {identity.model_path}")
    if identity.n_params:
        lines.append(f"  params       {_human_params(identity.n_params)}")
    if identity.size_bytes:
        lines.append(f"  size         {_human_bytes(identity.size_bytes)}")
    if identity.ftype:
        lines.append(f"  quant        {identity.ftype}")
    lines.append(
        f"  context      {identity.n_ctx:,} (trained {identity.n_ctx_train or 0:,})"
    )
    if identity.total_slots is not None:
        lines.append(f"  slots        {identity.total_slots}")
    if identity.build_info:
        lines.append(f"  build        {identity.build_info}")
    lines.append(f"  fingerprint  {identity.fingerprint}")
    return "\n".join(lines)


def resolve(
    endpoint: EndpointConfig,
    *,
    label: str | None = None,
    interactive: bool = True,
    cache_path: Path = MODELS_CACHE_PATH,
) -> ModelIdentity:
    """Probe the endpoint and attach a human label, asking once per fingerprint."""
    identity = probe(endpoint)
    cache = _load_cache(cache_path)
    cached = cache.get(identity.fingerprint)

    chosen = label or endpoint.label
    if chosen:
        identity.label = chosen
        identity.notes = (cached or {}).get("notes", "")
    elif cached:
        identity.label = cached["label"]
        identity.notes = cached.get("notes", "")
    elif interactive and sys.stdin.isatty():
        suggested = _suggest_label(
            identity.served_id, identity.model_path, identity.ftype
        )
        print("\nNew model detected at the endpoint:\n")
        print(describe(identity))
        print()
        answer = input(f"Display label [{suggested}]: ").strip()
        identity.label = answer or suggested
        identity.notes = input("Notes (optional): ").strip()
    else:
        identity.label = _suggest_label(
            identity.served_id, identity.model_path, identity.ftype
        )

    # A capability rather than an identity, but cached beside the label for the
    # same reason: it costs a live request, and a sweep would otherwise re-ask
    # it once per harness.
    #
    # Unlike the label, it is *not* a property of the weights alone. The same
    # GGUF answers differently behind llama.cpp and behind Ollama, so the
    # answer is filed under the base URL it was measured against -- otherwise
    # repointing at a second server would silently reuse the first one's
    # answer, which is the exact mistake this probe exists to stop. A bare
    # boolean written by an earlier version cannot be attributed to any
    # endpoint, so it is discarded rather than guessed at.
    #
    # Only a definite answer is trusted, so an endpoint that was merely
    # unreachable last time is asked again rather than pinned to "unknown"
    # forever.
    base_url = endpoint.resolved_base_url()
    by_endpoint = (cached or {}).get("supports_reasoning")
    by_endpoint = dict(by_endpoint) if isinstance(by_endpoint, dict) else {}
    if isinstance(by_endpoint.get(base_url), bool):
        identity.supports_reasoning = by_endpoint[base_url]
    else:
        identity.supports_reasoning = supports_reasoning_effort(
            endpoint, served_id=identity.served_id
        )
        if isinstance(identity.supports_reasoning, bool):
            by_endpoint[base_url] = identity.supports_reasoning

    entry = {
        "label": identity.label,
        "notes": identity.notes,
        "served_id": identity.served_id,
        "model_path": identity.model_path,
        "ftype": identity.ftype,
        "n_params": identity.n_params,
        "supports_reasoning": by_endpoint,
    }
    if cached != entry:
        cache[identity.fingerprint] = entry
        _save_cache(cache_path, cache)

    return identity


def add_endpoint_args(parser: argparse.ArgumentParser) -> None:
    """Endpoint flags shared by every command that talks to the model."""
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the configured endpoint, e.g. http://localhost:8080/v1",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=sorted(PROVIDERS),
        help="Override the configured provider",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id to benchmark (required for providers that serve many)",
    )
    parser.add_argument("--label", default=None, help="Override the model label")


def config_from_args(args: argparse.Namespace) -> Config:
    """Config with any endpoint flags layered on top."""
    return with_overrides(
        load(),
        provider=getattr(args, "provider", None),
        base_url=getattr(args, "base_url", None),
        model=getattr(args, "model", None),
        label=getattr(args, "label", None),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_endpoint_args(parser)
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument(
        "--no-input", action="store_true", help="Never prompt; auto-derive a label"
    )
    parser.add_argument(
        "--speed",
        action="store_true",
        help="Also measure output tokens/sec and recommend a timeout multiplier",
    )
    args = parser.parse_args(argv)

    config = config_from_args(args)
    identity = resolve(config.endpoint, interactive=not args.no_input)

    speed = measure_speed(config.endpoint) if args.speed else None

    if args.json:
        payload = identity.to_dict()
        if speed:
            payload["speed"] = speed
        print(json.dumps(payload, indent=2))
        return 0

    print(f"\n{identity.label}\n")
    print(describe(identity))
    if identity.notes:
        print(f"  notes        {identity.notes}")
    print(f"  slug         {identity.slug}")

    if speed:
        print(
            f"\n  speed        {speed['tokens_per_s']:.1f} output tok/s "
            f"({speed['completion_tokens']} tokens in {speed['elapsed_s']:.1f}s)"
        )
        print(f"  suggested    agent_timeout_multiplier: {speed['recommended_multiplier']}")
        print(
            f"\n  Set that in harnesses/registry.yaml under `defaults:`.\n"
            f"  {speed['note']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
