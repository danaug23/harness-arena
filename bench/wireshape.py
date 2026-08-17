"""What a harness puts on the wire, and whether this endpoint will accept it.

A harness and a model server can both be working and still be unable to talk to
each other, because they disagree about a request *shape* rather than about a
value. That failure is invisible to everything else this rig checks: the model
loads, ``/v1/models`` answers 200, every other harness runs, and the one that
cannot is the one you were trying to measure.

The case that produced this module, on 2026-08-17:

    Claude Code sends ``messages: [user, system]`` -- a second, non-first
    ``system``-role message carrying its agent and skill listings. llama.cpp's
    ``/v1/messages`` bridge passes the role straight through to the model's
    chat template, and Qwen3.8-27B's template raises::

        {%- if message.role == "system" %}
            {%- if not loop.first %}
                {{- raise_exception('System message must be at the beginning.') }}

    Every trial died at its first request with HTTP 500 after ten retries, ~3
    minutes each, having done no work. The same endpoint, the same llama.cpp
    build and the same Claude Code version had run a full sweep two days
    earlier against different weights, whose template did not raise. Nothing in
    the rig changed; the chat template inside the GGUF did.

Two properties keep this from becoming its own source of broken runs:

**A control request decides whether the question means anything.** Every probe
sends the *same* payload twice, once in the ordinary shape and once in the
shape the harness actually sends. A rejection only counts when the ordinary
shape was accepted first. Without that, a wrong model id or a missing
credential would read as "this server hates Claude Code" and block a run that
would have been fine. This is the rule `probe.supports_reasoning_effort`
already uses, for the same reason.

**Only a recognised rejection blocks.** A refusal whose text is not one this
module knows is reported and *not* acted on. So an endpoint that works today
cannot be blocked tomorrow by a probe reading a novel error message as fatal:
the worst a surprise can do is produce a warning. Everything else is left to
behave exactly as it did before this module existed.

Which harnesses are asked about is read off the catalog rather than listed
here. A harness is Anthropic-shaped if its catalog entry routes an Anthropic
client at the endpoint -- ``ANTHROPIC_BASE_URL`` -- so a future harness that
speaks the Messages API is covered the day it is added, and no harness that
speaks anything else is ever probed for a shape it does not send.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from bench.config import EndpointConfig

# ---------------------------------------------------------------------------
# Addressing
# ---------------------------------------------------------------------------

#: A trailing API-version segment, e.g. the "/v1" in http://192.0.2.10:8002/v1.
#:
#: Which of the two forms a harness wants is not a style choice, it is a
#: property of the client SDK, and the two families disagree:
#:
#:   * Anthropic's client owns the version segment. It posts to
#:     <root>/v1/messages, so handing it a base URL that already ends in /v1
#:     produces /v1/v1/messages. Measured against claude-cli 2.1.226 pointed at
#:     a logging server, not inferred from the docs.
#:   * OpenAI's clients do not. Codex posts to <base_url>/responses, so the
#:     same URL must keep its /v1. Measured the same way, codex-cli 0.147.0.
#:
#: Both are therefore correct and neither can be the single value of
#: {base_url}. Hence a second placeholder rather than a per-harness fixup: the
#: distinction belongs to the SDK, so every Anthropic-shaped harness added
#: later wants the same thing.
#:
#: Lives here rather than in bench.runner because it is a statement about how a
#: client addresses an endpoint, which is this module's whole subject -- and
#: because bench.diagnose needs it without importing the runner. `bench.runner`
#: re-exports it, so `bench.runner.base_url_root` still resolves.
_VERSION_SUFFIX = re.compile(r"/+v\d+/*$")


def base_url_root(base_url: str) -> str:
    """`base_url` with any trailing API-version segment removed.

    Idempotent, and a no-op for a URL that has none -- an endpoint served at a
    bare host:port stays exactly as configured.
    """
    stripped = (base_url or "").strip()
    return _VERSION_SUFFIX.sub("", stripped) or stripped


def messages_url(base_url: str) -> str:
    """Where an Anthropic-shaped client posts, given a configured base URL."""
    return f"{base_url_root(base_url)}/v1/messages"


# ---------------------------------------------------------------------------
# Which harnesses speak which wire
# ---------------------------------------------------------------------------

#: The variable that points an Anthropic client at a non-Anthropic endpoint.
#: A catalog entry setting it is, by construction, routing the Messages API
#: here -- which is the only thing this module needs to know about a harness.
_ANTHROPIC_ROUTING_VAR = "ANTHROPIC_BASE_URL"


def anthropic_shaped(spec: dict[str, Any]) -> bool:
    """Whether this catalog entry points an Anthropic Messages client at us."""
    if not isinstance(spec, dict):
        return False
    for block in ("host_env", "agent_env"):
        env = spec.get(block)
        if isinstance(env, dict) and _ANTHROPIC_ROUTING_VAR in env:
            return True
    return False


# ---------------------------------------------------------------------------
# The shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WireShape:
    """One request shape a harness sends that an endpoint may not accept.

    `control` and `question` differ in exactly the property being tested and in
    nothing else, so a difference in the answers is attributable.
    """

    id: str
    title: str
    #: What the harness does, in the terms of someone reading a failure.
    detail: str
    #: Rejection texts that mean "this endpoint refuses this shape". Each entry
    #: is a tuple of needles that must *all* appear, which is what stops a bare
    #: "error" from matching everything. Only these block a run -- see the
    #: module docstring.
    rejection: tuple[tuple[str, ...], ...]
    fixes: tuple[str, ...]

    def bodies(self, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError  # pragma: no cover - overridden per shape


@dataclass(frozen=True)
class _SystemNotFirst(WireShape):
    """A ``system``-role message that is not the first message.

    Claude Code emits one per session: the top-level `system` array carries the
    CLI's own prompt, and the agent/skill listings arrive as a separate
    system-role turn *after* the first user message. The top-level array is not
    the same thing and is not what fails -- llama.cpp merges its blocks into a
    single leading system message, and three blocks are accepted where one
    trailing system *message* is not.
    """

    def bodies(self, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
        base: dict[str, Any] = {
            "model": model,
            # The smallest generation the wire allows: this asks a question
            # about the prompt being *accepted*, and answering it costs a slot
            # on a machine that is usually about to run a benchmark.
            "max_tokens": 1,
            "system": [{"type": "text", "text": "You are a helpful assistant."}],
        }
        control = {**base, "messages": [{"role": "user", "content": "hi"}]}
        question = {
            **base,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": [{"type": "text", "text": "Reference."}]},
            ],
        }
        return control, question


SYSTEM_NOT_FIRST = _SystemNotFirst(
    id="system-not-first",
    title="A system message that is not the first message",
    detail=(
        "Claude Code sends its agent and skill listings as a system-role "
        "message after the first user turn. Many chat templates accept a "
        "system message only in position zero and abort rendering when they "
        "find one anywhere else, which surfaces as HTTP 500 on every request "
        "the harness makes -- so every trial fails at its first call having "
        "done no work."
    ),
    rejection=(
        ("system message must be at the beginning",),
        ("system message must be first",),
        ("jinja", "system message"),
        ("raise_exception", "system message"),
        ("only one system message",),
        ("system role", "not supported"),
    ),
    fixes=(
        "Patch the server's chat template: harness-arena template-fix",
        "Then restart llama-server with --chat-template-file <the written file> "
        "and re-run `harness-arena doctor` to confirm.",
        "Or leave the Anthropic-shaped harnesses out of this sweep and run the "
        "rest, which are unaffected.",
    ),
)

#: Every shape this rig knows how to ask about, with the harnesses it applies
#: to. Keyed by the predicate rather than by harness id so the catalog stays
#: the source of truth about what a harness is.
SHAPES: tuple[tuple[WireShape, Any], ...] = ((SYSTEM_NOT_FIRST, anthropic_shaped),)


# ---------------------------------------------------------------------------
# Asking the endpoint
# ---------------------------------------------------------------------------

#: The three answers, and what each one licenses.
#:
#:   accepted  the endpoint took the shape -- nothing to do
#:   rejected  it refused, in terms this module recognises -- block the run
#:   unknown   the question could not be answered, or was answered in terms
#:             nobody has seen before -- report, change nothing
ACCEPTED, REJECTED, UNKNOWN = "accepted", "rejected", "unknown"


@dataclass
class Verdict:
    """What one endpoint said about one shape."""

    shape: WireShape
    result: str = UNKNOWN
    #: HTTP status of the question request, or 0 if it never landed.
    status: int = 0
    #: What the server said, trimmed. Empty when it said nothing useful.
    message: str = ""
    #: Why the answer is `unknown`, when it is.
    why: str = ""
    #: Harnesses in the current selection that send this shape.
    harnesses: list[str] = field(default_factory=list)

    @property
    def blocks(self) -> bool:
        return self.result == REJECTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape.id,
            "title": self.shape.title,
            "result": self.result,
            "status": self.status,
            "message": self.message,
            "why": self.why,
            "harnesses": list(self.harnesses),
        }


def _post(
    url: str, body: dict[str, Any], api_key: str, timeout: float
) -> tuple[int, str]:
    """POST JSON and return (status, body-text). Status 0 means it never landed."""
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("User-Agent", "harness-arena")
    # Both spellings: Anthropic servers read x-api-key, and the OpenAI-shaped
    # servers that also expose /v1/messages read the bearer header. Sending
    # both costs nothing and saves a false "unauthorised" reading as a refusal.
    request.add_header("anthropic-version", "2023-06-01")
    if api_key:
        request.add_header("x-api-key", api_key)
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status, ""
    except urllib.error.HTTPError as exc:
        try:
            text = exc.read().decode("utf-8", "replace")
        except OSError:
            text = ""
        return exc.code, text
    except (urllib.error.URLError, OSError, ValueError):
        return 0, ""


def _recognised(shape: WireShape, text: str) -> bool:
    lowered = (text or "").lower()
    return any(all(n in lowered for n in group) for group in shape.rejection)


def _trim(text: str, limit: int = 400) -> str:
    """One line of server error, short enough to print beside a fix."""
    collapsed = " ".join((text or "").split())
    return collapsed[:limit]


def probe_shape(
    endpoint: EndpointConfig,
    shape: WireShape,
    *,
    served_id: str,
    timeout: float = 30.0,
) -> Verdict:
    """Ask one endpoint whether it accepts one shape.

    Never raises. This runs on the path to a run, where an exception would
    abort the very runs the `unknown` fallback exists to let through untouched.
    """
    verdict = Verdict(shape=shape)
    if not served_id:
        verdict.why = "the endpoint could not be asked which model it serves"
        return verdict

    url = messages_url(endpoint.resolved_base_url())
    api_key = endpoint.resolve_api_key()
    control, question = shape.bodies(served_id)

    status, text = _post(url, control, api_key, timeout)
    if status != 200:
        # No Messages route, a wrong model id, a missing credential, an
        # unreachable host. Every one of them makes the question meaningless,
        # and none of them is this shape's fault.
        verdict.why = (
            "the endpoint did not accept an ordinary request either "
            f"(HTTP {status or 'no response'}), so nothing can be concluded "
            "about this shape"
        )
        verdict.status = status
        verdict.message = _trim(text)
        return verdict

    status, text = _post(url, question, api_key, timeout)
    verdict.status = status
    verdict.message = _trim(text)
    if status == 200:
        verdict.result = ACCEPTED
        verdict.message = ""
        return verdict
    if status == 0:
        verdict.why = "the request never landed, so the refusal is not established"
        return verdict
    if _recognised(shape, text):
        verdict.result = REJECTED
        return verdict
    verdict.why = (
        f"the endpoint refused this shape with HTTP {status}, in terms this "
        "rig has not seen before -- reported, but not acted on"
    )
    return verdict


def shapes_for(spec: dict[str, Any]) -> list[WireShape]:
    """Every shape a given catalog entry is known to send."""
    return [shape for shape, applies in SHAPES if applies(spec)]


def check_selection(
    endpoint: EndpointConfig,
    registry: dict[str, Any],
    harness_ids: list[str],
    *,
    served_id: str,
    timeout: float = 30.0,
) -> list[Verdict]:
    """Probe only the shapes the selected harnesses actually send.

    An empty result means nothing in this selection sends a shape worth asking
    about, which is the ordinary case: one probe is skipped entirely rather
    than answered, so a sweep of OpenAI-shaped harnesses costs no requests.
    """
    catalog = registry.get("harnesses") or {}
    wanted: dict[str, list[str]] = {}
    by_id: dict[str, WireShape] = {}
    for harness_id in harness_ids:
        for shape in shapes_for(catalog.get(harness_id) or {}):
            by_id[shape.id] = shape
            wanted.setdefault(shape.id, []).append(harness_id)

    verdicts = []
    for shape_id, senders in wanted.items():
        verdict = probe_shape(
            endpoint, by_id[shape_id], served_id=served_id, timeout=timeout
        )
        verdict.harnesses = senders
        verdicts.append(verdict)
    return verdicts


def blocked_harnesses(verdicts: list[Verdict]) -> dict[str, Verdict]:
    """Harness -> the verdict that stops it. Only recognised refusals appear."""
    blocked: dict[str, Verdict] = {}
    for verdict in verdicts:
        if not verdict.blocks:
            continue
        for harness_id in verdict.harnesses:
            blocked.setdefault(harness_id, verdict)
    return blocked


# ---------------------------------------------------------------------------
# Patching the template that refuses
# ---------------------------------------------------------------------------
#
# The remedy for a template that rejects a shape is to change the template, not
# to change what the harness sends: rewriting the request would mean the
# harness under test is no longer the harness being measured, and a run altered
# that way is not comparable to one that was not.
#
# The patch below is deliberately the smallest edit that removes the refusal.
# It replaces the `raise_exception` with the emit the neighbouring branches
# already use, and touches no other branch -- so for every conversation that
# renders today, the patched template produces byte-identical output. The only
# behaviour that changes is the one that currently aborts.

#: The refusal itself. llama.cpp's `/props` exposes the template the loaded
#: weights carry, which is where this is found.
_RAISE = re.compile(
    r"\{\{-?\s*raise_exception\(\s*"
    r"(['\"])System message must be (?:at the beginning|first)\.?\1"
    r"\s*\)\s*-?\}\}"
)

#: What replaces it: the ChatML system turn the template already emits for a
#: leading system message, rendered in place so ordering is preserved.
_EMIT = "{{- '<|im_start|>system\\n' + content + '<|im_end|>' + '\\n' }}"

#: The variable the surrounding loop is expected to hold the rendered message
#: in. Checked rather than assumed: a template that names it something else
#: would be patched into one that renders an undefined variable, which is a
#: worse failure than the one being fixed.
_CONTENT_BINDING = re.compile(r"\{%-?\s*set\s+content\s*=")


@dataclass
class Patch:
    """The result of trying to patch a chat template."""

    ok: bool
    text: str = ""
    #: One line per edit, for printing.
    changes: list[str] = field(default_factory=list)
    #: Why it was refused, when it was.
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "changes": list(self.changes),
            "why": self.why,
            "chars": len(self.text),
        }


def patch_template(template: str) -> Patch:
    """Remove a chat template's refusal of a non-first system message.

    Refuses rather than guesses. A template that is not ChatML, or whose loop
    does not bind the rendered message to `content`, cannot take this edit --
    and a template patched on a guess would fail at render time on every
    request instead of on the one shape that fails today.
    """
    if not template or not template.strip():
        return Patch(ok=False, why="the endpoint reported no chat template")

    matches = list(_RAISE.finditer(template))
    if not matches:
        return Patch(
            ok=False,
            why=(
                "this template does not refuse a non-first system message, so "
                "there is nothing here to patch -- the endpoint's refusal is "
                "something else"
            ),
        )
    if "<|im_start|>" not in template:
        return Patch(
            ok=False,
            why=(
                "this template is not ChatML, so the replacement turn would "
                "use markers the model was not trained on. Patch it by hand: "
                "emit the message in the template's own system format instead "
                "of raising."
            ),
        )

    changes = []
    for match in matches:
        # The binding has to be in scope where the raise is. Looked for above
        # the match rather than anywhere in the file, so a `set content` in an
        # unrelated macro cannot vouch for a loop that has none.
        before = template[: match.start()]
        if not _CONTENT_BINDING.search(before[-2000:]):
            return Patch(
                ok=False,
                why=(
                    "the refusal is not inside a loop that binds the rendered "
                    "message to `content`, so the replacement would render an "
                    "undefined variable. Patch this template by hand."
                ),
            )
        line = before.count("\n") + 1
        changes.append(f"line {line}: raise_exception(...) -> emit a system turn")

    # A function replacement, so backslashes in _EMIT stay literal.
    patched = _RAISE.sub(lambda _m: _EMIT, template)
    return Patch(ok=True, text=patched, changes=changes)


def fetch_template(
    endpoint: EndpointConfig, timeout: float = 15.0
) -> tuple[str, str]:
    """(template, why-not) from the endpoint's /props. Never raises."""
    root = base_url_root(endpoint.resolved_base_url())
    api_key = endpoint.resolve_api_key()
    for url in (f"{root}/props", f"{endpoint.resolved_base_url()}/props"):
        request = urllib.request.Request(url)
        request.add_header("User-Agent", "harness-arena")
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                props = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if isinstance(props, dict) and isinstance(props.get("chat_template"), str):
            return props["chat_template"], ""
    return "", (
        f"{root}/props did not return a chat template. Only servers that "
        "expose their template can have it patched from here -- llama.cpp "
        "does; most hosted providers do not."
    )
