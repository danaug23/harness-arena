"""Request-shape probing, and the template patch that fixes a refusal.

Two things have to hold, and the second matters more than the first:

  * A refusal this rig recognises stops the harness that would hit it.
  * *Nothing else* is stopped. A probe that blocked a working endpoint would
    be a worse bug than the one it exists to catch, so most of this file is
    about the answers that must not block: an unreachable server, a wrong
    model id, a rejection nobody has seen before, a harness that does not send
    the shape at all.

Nothing here makes a network request. `_post` is the seam -- it is the only
function in bench/wireshape.py that touches a socket, so replacing it puts a
whole endpoint's behaviour under the test's control.

    python tests/test_wireshape.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import diagnose, wireshape  # noqa: E402
from bench.collect import salient_error  # noqa: E402
from bench.runner import base_url_root as runner_base_url_root  # noqa: E402

failures: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<62} got={got!r} want={want!r}")
    if not ok:
        failures.append(label)


def check_true(label: str, got: object) -> None:
    check(label, bool(got), True)


# ---------------------------------------------------------------------------
# The real error text, from the run that produced this module
# ---------------------------------------------------------------------------
#
# llama.cpp b10269 serving Qwen3.8-27B, answering Claude Code 2.1.223. Kept
# verbatim apart from the endpoint's address: a signature that stops matching
# the thing it was written for is worse than no signature at all.

LLAMA_CPP_500 = (
    '{"error":{"code":500,"message":"\\n------------\\nWhile executing '
    "CallExpression at line 106, column 32 in source:\\n...first %}\\u21b5 "
    "{{- raise_exception('System message must be at the beginnin...\\n ^\\n"
    'Error: Jinja Exception: System message must be at the beginning.",'
    '"type":"server_error"}}'
)

# What Harbor recorded for the trial: 5 KB of captured agent stdout with the
# command that failed at the front and the reason at the very end.
HARBOR_EXCEPTION = (
    'Command failed (exit 1): export PATH="$HOME/.local/bin:$PATH"; printf '
    '"%s" "$instruction" | claude --verbose --output-format=stream-json '
    "--permission-mode=bypassPermissions --print\n"
    "stdout: " + ('{"type":"system","subtype":"api_retry"}\n' * 200) +
    '{"subtype":"success","api_error_status":500,"result":"API Error: 500 '
    "\\n------------\\nError: Jinja Exception: System message must be at the "
    'beginning."}'
)


# ---------------------------------------------------------------------------
# Which harnesses are asked, and which are never asked
# ---------------------------------------------------------------------------

print("-- who sends what --")

CLAUDE_LIKE = {
    "agent": "claude-code",
    "host_env": {"ANTHROPIC_BASE_URL": "{base_url_root}"},
    "agent_env": {"ANTHROPIC_API_KEY": "{api_key}"},
}
OPENAI_LIKE = {
    "agent": "harnesses.codex:Codex",
    "agent_env": {"OPENAI_API_KEY": "{api_key}"},
    "agent_kwargs": {"base_url": "{base_url}"},
}

check("an Anthropic-routed harness is recognised",
      wireshape.anthropic_shaped(CLAUDE_LIKE), True)
check("an OpenAI-routed harness is not", wireshape.anthropic_shaped(OPENAI_LIKE),
      False)
check("a harness with no env block is not", wireshape.anthropic_shaped({}), False)
check("garbage is not", wireshape.anthropic_shaped(None), False)
# Read off the catalog, not a list here, so a harness added later is covered
# without anyone remembering to add it.
check("the routing variable is enough on its own",
      wireshape.anthropic_shaped({"agent_env": {"ANTHROPIC_BASE_URL": "x"}}), True)

check("an Anthropic harness has a shape to check",
      [s.id for s in wireshape.shapes_for(CLAUDE_LIKE)], ["system-not-first"])
check("an OpenAI harness has none", wireshape.shapes_for(OPENAI_LIKE), [])

# The shipped catalog must agree, or the check silently covers nothing.
from bench import registry as registry_mod  # noqa: E402

_catalog = registry_mod.load()
_shaped = sorted(
    h for h, spec in (_catalog.get("harnesses") or {}).items()
    if wireshape.anthropic_shaped(spec)
)
check("exactly the Anthropic harnesses in the shipped catalog", _shaped,
      ["claude-code"])


# ---------------------------------------------------------------------------
# Addressing
# ---------------------------------------------------------------------------

print("\n-- addressing --")

check("an Anthropic client posts under /v1/messages",
      wireshape.messages_url("http://example.invalid:8002"),
      "http://example.invalid:8002/v1/messages")
check("...and a configured /v1 is not doubled",
      wireshape.messages_url("http://example.invalid:8002/v1"),
      "http://example.invalid:8002/v1/messages")
# The runner re-exports it; nothing that imported it before may break.
check("bench.runner still exposes base_url_root",
      runner_base_url_root("http://example.invalid/v1"), "http://example.invalid")


# ---------------------------------------------------------------------------
# The probe: a rejection only counts once the control is known good
# ---------------------------------------------------------------------------

print("\n-- probing --")


class FakeEndpoint:
    """Enough EndpointConfig for the probe, with no config file involved."""

    def __init__(self, base_url: str = "http://example.invalid:8002") -> None:
        self._base_url = base_url

    def resolved_base_url(self) -> str:
        return self._base_url

    def resolve_api_key(self) -> str:
        return "local"


def with_endpoint(answers, *, record=None):
    """Replace the socket seam with a scripted server.

    `answers` is called with each request body and returns (status, text).
    """
    original = wireshape._post

    def fake(url, body, api_key, timeout):
        if record is not None:
            record.append(body)
        return answers(body)

    wireshape._post = fake
    try:
        return wireshape.probe_shape(
            FakeEndpoint(), wireshape.SYSTEM_NOT_FIRST, served_id="m"
        )
    finally:
        wireshape._post = original


def _is_question(body):
    """The question is the request whose messages carry a trailing system."""
    return any(m.get("role") == "system" for m in body.get("messages", []))


# The case this was written for.
v = with_endpoint(lambda b: (500, LLAMA_CPP_500) if _is_question(b) else (200, ""))
check("a template refusal is a refusal", v.result, wireshape.REJECTED)
check("...and it blocks", v.blocks, True)
check("...and the status is kept", v.status, 500)

# An endpoint that takes the shape.
v = with_endpoint(lambda b: (200, ""))
check("an endpoint that accepts it says so", v.result, wireshape.ACCEPTED)
check("...and blocks nothing", v.blocks, False)
check("...and quotes no error", v.message, "")

# Everything below must NOT block. These are the answers that would turn this
# check into its own outage.
v = with_endpoint(lambda b: (404, "no such route"))
check("no Messages route is not a refusal", v.result, wireshape.UNKNOWN)
check("...and blocks nothing", v.blocks, False)

v = with_endpoint(lambda b: (401, "unauthorized"))
check("a missing credential is not a refusal", v.result, wireshape.UNKNOWN)

v = with_endpoint(lambda b: (0, ""))
check("an unreachable endpoint is not a refusal", v.result, wireshape.UNKNOWN)

v = with_endpoint(
    lambda b: (400, "model 'm' not found") if _is_question(b) else (200, "")
)
check("an unrecognised rejection is not acted on", v.result, wireshape.UNKNOWN)
check("...and blocks nothing", v.blocks, False)
check_true("...and says why it could not answer", len(v.why) > 20)
check_true("...while still quoting what the server said", "not found" in v.message)

v = with_endpoint(lambda b: (500, LLAMA_CPP_500))
check("a server failing the control request too proves nothing",
      v.result, wireshape.UNKNOWN)

v = with_endpoint(
    lambda b: (0, "") if _is_question(b) else (200, "")
)
check("a question that never landed proves nothing", v.result, wireshape.UNKNOWN)

# Without a model id there is no question to ask, and asking anyway would send
# a request the server cannot route.
_calls: list = []
_original = wireshape._post
wireshape._post = lambda *a, **k: _calls.append(a) or (200, "")
try:
    v = wireshape.probe_shape(FakeEndpoint(), wireshape.SYSTEM_NOT_FIRST, served_id="")
finally:
    wireshape._post = _original
check("no model id means no request at all", len(_calls), 0)
check("...and no verdict", v.result, wireshape.UNKNOWN)

# The two requests must differ in the property under test and nothing else,
# or a difference in the answers is not attributable to it.
_sent: list = []
with_endpoint(lambda b: (200, ""), record=_sent)
check("exactly two requests are sent", len(_sent), 2)
_control, _question = _sent
check("the control has no trailing system message",
      [m["role"] for m in _control["messages"]], ["user"])
check("the question does", [m["role"] for m in _question["messages"]],
      ["user", "system"])
check("they agree on the model", _control["model"], _question["model"])
check("they agree on the system prompt", _control["system"], _question["system"])
check("both ask for as little generation as the wire allows",
      (_control["max_tokens"], _question["max_tokens"]), (1, 1))


# ---------------------------------------------------------------------------
# Selection: nothing is probed on behalf of a harness that does not send it
# ---------------------------------------------------------------------------

print("\n-- selection --")

FAKE_CATALOG = {"harnesses": {"claude-code": CLAUDE_LIKE, "codex": OPENAI_LIKE}}


def selection(harnesses, answers=lambda b: (200, "")):
    calls: list = []
    original = wireshape._post

    def fake(url, body, api_key, timeout):
        calls.append(body)
        return answers(body)

    wireshape._post = fake
    try:
        verdicts = wireshape.check_selection(
            FakeEndpoint(), FAKE_CATALOG, harnesses, served_id="m"
        )
    finally:
        wireshape._post = original
    return verdicts, calls


_v, _calls = selection(["codex"])
check("an OpenAI-only sweep is not probed at all", len(_calls), 0)
check("...and produces no verdict", _v, [])

_v, _calls = selection(["claude-code", "codex"])
check("a mixed sweep asks once", len(_v), 1)
check("...two requests", len(_calls), 2)
check("...and names only the harness that sends it", _v[0].harnesses, ["claude-code"])

_v, _ = selection(["claude-code", "codex"],
                  lambda b: (500, LLAMA_CPP_500) if _is_question(b) else (200, ""))
_blocked = wireshape.blocked_harnesses(_v)
check("only the sender is blocked", sorted(_blocked), ["claude-code"])
check("the unaffected harness is untouched", "codex" in _blocked, False)

_v, _ = selection(["claude-code"], lambda b: (400, "something new") if
                  _is_question(b) else (200, ""))
check("an unrecognised refusal blocks nobody",
      wireshape.blocked_harnesses(_v), {})

_v, _ = selection([])
check("an empty selection asks nothing", _v, [])


# ---------------------------------------------------------------------------
# Patching the template
# ---------------------------------------------------------------------------

print("\n-- template patch --")

# The real loop, verbatim from the weights that failed.
REAL_LOOP = """{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- endif %}
{%- endfor %}"""

patch = wireshape.patch_template(REAL_LOOP)
check("the real refusal is patched", patch.ok, True)
check("...one edit", len(patch.changes), 1)
check("...at the line it is on", patch.changes[0].startswith("line 5:"), True)
check("the refusal is gone", "raise_exception" in patch.text, False)
check("...replaced by a system turn",
      "'<|im_start|>system\\n' + content" in patch.text, True)
# The property the whole approach rests on: only the aborting branch moves.
_before = REAL_LOOP.splitlines()
_after = patch.text.splitlines()
check("every other line is byte-identical",
      [i for i, (a, b) in enumerate(zip(_before, _after, strict=True)) if a != b],
      [4])

# Applying it twice must not double the emit.
check("patching a patched template is a no-op",
      wireshape.patch_template(patch.text).ok, False)

# Refusals, which are the safe answer whenever the edit cannot be shown correct.
check("an empty template is refused", wireshape.patch_template("").ok, False)
check("...with a reason", len(wireshape.patch_template("").why) > 20, True)

_clean = "{%- for m in messages %}{{- '<|im_start|>' + m.role }}{%- endfor %}"
_p = wireshape.patch_template(_clean)
check("a template with nothing to patch is refused", _p.ok, False)
check("...and says so rather than inventing an edit", "nothing here to patch" in _p.why,
      True)

_not_chatml = REAL_LOOP.replace("<|im_start|>", "[INST]")
_p = wireshape.patch_template(_not_chatml)
check("a non-ChatML template is refused", _p.ok, False)
check("...because the markers would be wrong", "not ChatML" in _p.why, True)

_no_content = REAL_LOOP.replace(
    "{%- set content = render_content(message.content, true)|trim %}", ""
)
_p = wireshape.patch_template(_no_content)
check("a loop that does not bind `content` is refused", _p.ok, False)
check("...rather than emitting an undefined variable", "`content`" in _p.why, True)

# Two raises in one template are both edited.
_twice = REAL_LOOP + "\n" + REAL_LOOP
check("every occurrence is patched", len(wireshape.patch_template(_twice).changes), 2)


# ---------------------------------------------------------------------------
# The same failure, after the fact
# ---------------------------------------------------------------------------

print("\n-- naming it afterwards --")

_f = diagnose.explain(LLAMA_CPP_500)
check("the raw server error is named", _f.id if _f else None,
      "system-message-position")
_f = diagnose.explain(HARBOR_EXCEPTION)
check("...and so is the trial exception that wrapped it",
      _f.id if _f else None, "system-message-position")
check_true("...carrying the command that fixes it",
           _f and any("template-fix" in fix for fix in _f.fixes))

# A different template failure is still named, one step less specifically.
_f = diagnose.explain("Error: Jinja Exception: Unexpected message role.")
check("another template failure is named generically", _f.id if _f else None,
      "chat-template-error")

# The signature has to keep matching what the shape probe itself receives, or
# the run-time check and the after-the-fact explanation disagree about the same
# server saying the same thing.
check_true("the probe and the signature agree on the same text",
           wireshape._recognised(wireshape.SYSTEM_NOT_FIRST, LLAMA_CPP_500))


# ---------------------------------------------------------------------------
# Surfacing it: the message must keep the end, which is where the reason is
# ---------------------------------------------------------------------------

print("\n-- what the dashboard is given --")

_msg = salient_error(HARBOR_EXCEPTION)
check_true("the reason survives truncation",
           "System message must be at the beginning" in _msg)
check_true("...and so does the command that failed", "claude --verbose" in _msg)
check_true("...without the whole capture", len(_msg) < 1400)
check_true("...and the cut is marked", "[...]" in _msg)
check("a short message is passed through whole", salient_error("boom"), "boom")
check("nothing is nothing", salient_error(""), None)
check("None is safe", salient_error(None), None)
# Colour codes must not eat the budget the explanation needs.
check("colour is stripped", salient_error("\x1b[31mboom\x1b[0m"), "boom")


print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
