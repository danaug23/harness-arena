"""Guard the rules that decide which runs may be compared.

head_to_head is the panel that carries the whole experiment, and a bad pairing
does not look broken -- it renders a confident disagreement set that is actually
measuring the wrong variable. These are the rules that keep it honest.

    python tests/test_collect.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows hands Python a legacy code page (cp1252) when stdout is a pipe, which is
# what CI gives it. The feed renders "->" as an arrow and elides with an ellipsis,
# so echoing an expected value here raised UnicodeEncodeError and failed the job on
# Windows only -- an encoding accident reported as a broken test suite.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench.activity import (  # noqa: E402
    MAX_TAIL_BYTES,
    TAIL_BYTES,
    TARGET_TEXT_CHARS,
    agent_started_at,
    parse_entries,
    read_feed,
)
from bench.collect import (  # noqa: E402
    _token_totals,
    build_index,
    head_to_head,
    load_run,
    wilson_interval,
)

failures: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<52} got={got!r} want={want!r}")
    if not ok:
        failures.append(label)


def make_run(**over):
    run = {
        "run_id": "r", "harness": "hermes", "harness_label": "Hermes",
        "model_fingerprint": "fp1", "model_label": "M", "n_done": 3,
        "model_quant": "TESTQ", "model_params": 1e11, "model_n_ctx": 131072,
        "is_partial": True, "subset": "stratified-25",
        "dataset": "terminal-bench@2.0",
        "agent_timeout_multiplier": 8.0, "status": "complete",
        "started_at": "2026-08-09T01:00:00+00:00",
        "tasks": [
            {"task_name": "a", "resolved": True},
            {"task_name": "b", "resolved": False},
            {"task_name": "c", "resolved": True},
        ],
    }
    run.update(over)
    return run


def index_of(runs):
    """build_index over hand-made runs, for the parts that are not comparisons."""
    import bench.collect as collect

    original = collect.load_runs
    collect.load_runs = lambda runs_dir=None: runs
    try:
        return build_index(Path("."))
    finally:
        collect.load_runs = original


def pairs(runs):
    """Run build_index's pairing rules over hand-made runs."""
    import bench.collect as collect

    original = collect.load_runs
    collect.load_runs = lambda runs_dir=None: runs
    try:
        return build_index(Path("."))["comparisons"]
    finally:
        collect.load_runs = original


def models_of(runs):
    """build_index's model summary over hand-made runs."""
    import bench.collect as collect

    original = collect.load_runs
    collect.load_runs = lambda runs_dir=None: runs
    try:
        return build_index(Path("."))["models"]
    finally:
        collect.load_runs = original


def test_model_summary_reports_the_window_that_was_used() -> None:
    """The window shown must be the one the harnesses got, not the probe's.

    llama.cpp reports the loaded window, so the probe alone was enough. Ollama
    reports none, so a run with a perfectly definite configured window
    summarised as "unknown" -- the panel read as broken on exactly the setup
    that needs the setting most.
    """
    ollama = make_run(model_n_ctx=0, context_window=65536)
    check("a configured window is reported", models_of([ollama])[0]["context_window"], 65536)

    # A server that answers still works, with nothing configured.
    llamacpp = make_run(model_n_ctx=131072, context_window=None)
    summary = models_of([llamacpp])[0]
    check("a probed window still lands", summary["n_ctx"], 131072)
    check("...and is not overwritten", summary["context_window"], None)

    # Runs arrive newest first; a run predating the field must not mask a newer
    # one that has it.
    old = make_run(run_id="old", context_window=None,
                   started_at="2026-08-01T01:00:00+00:00")
    new = make_run(run_id="new", context_window=32768,
                   started_at="2026-08-09T01:00:00+00:00")
    check("the newest run with a window wins",
          models_of([new, old])[0]["context_window"], 32768)
    check("...regardless of input order",
          models_of([old, new])[0]["context_window"], 32768)



def test_a_harness_that_stops_on_its_own_output_cap_is_not_a_clean_trial(tmp: Path) -> None:
    """A run that self-terminated on max_tokens must not read as clean trials.

    Harbor's Claude Code adapter raises OutputTokenExceededError for this, so the
    trial carries an exception and counts as an error. The DeepSeek Harness writes
    a turn/end whose reason is max-tokens, exits, and records no exception at all
    -- so the same event arrived as a completed trial with a reward of zero,
    indistinguishable from an honest wrong answer. On the run that produced this
    check, 8 of 25 trials ended that way and every one scored zero, while the
    other 17 averaged 0.765. The headline was measuring the cap.
    """
    job = tmp / "dsh__m__20260818T171816Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "dsh", "harness_label": "DeepSeek Harness",
        "model": {"label": "M", "fingerprint": "fp1", "provider": "openai-compatible"},
    }), encoding="utf-8")

    _write_trial(job, "solved-it", resolved=True, tokens=900, suffix="a")
    # Wrong answer, ordinary. Its trace ends on a normal completion.
    _write_trial(job, "just-wrong", resolved=False, tokens=800, suffix="b",
                 log='{"type":"turn/end","data":{"turn":1,'
                     '"reason":{"kind":"completed"}}}\n')
    # Same reward, entirely different event: the harness stopped itself.
    _write_trial(job, "ran-out-of-output", resolved=False, tokens=16384, suffix="c",
                 log='{"type":"step/end","data":{"turn":1,"step":1}}\n'
                     '{"type":"turn/end","data":{"turn":1,'
                     '"reason":{"kind":"max-tokens"}}}\n')

    run = load_run(job)
    by_task = {t["task_name"]: t for t in run["tasks"]}
    check("the trial the cap ended is flagged as such",
          by_task["ran-out-of-output"]["hit_output_cap"], True)
    check("an ordinary wrong answer is not",
          by_task["just-wrong"]["hit_output_cap"], False)
    check("a solved trial is never checked for it",
          by_task["solved-it"]["hit_output_cap"], False)
    check("the run counts them where a reader will see them",
          run["n_output_cap"], 1)

    # Flagged, and deliberately nothing more. The dsh adapter swallows the
    # non-zero exit for a max-tokens turn precisely so the workspace is scored
    # rather than discarded ungraded, and a graded result is not an error
    # anywhere else in this module. Reclassifying it here would undo that on
    # purpose and put an error glyph on the one outcome that is a real
    # measurement of what the agent managed before it was cut off.
    check("...without calling a graded trial an error",
          by_task["ran-out-of-output"]["error_type"], None)
    check("...without charging it to the harness as a crash",
          by_task["ran-out-of-output"]["fault"], None)
    check("...and without inflating the error count", run["n_errors"], 0)
    check("...or shrinking the denominator", run["n_done"], 3)


def test_tokens_are_recovered_from_the_agent_log_when_harbor_recorded_none(
    tmp: Path,
) -> None:
    """A trial Harbor could not account for is a gap, not a harness spending nothing.

    Measured: two Codex trials in one sweep recorded no tokens at all, because
    Harbor's trajectory conversion could not open the session rollout -- the path
    was 260 and 264 characters on a Windows box with long paths off. One of them
    was a solve worth 2.26M input and 140k output, and the usage was sitting in
    the harness's own stdout the whole time.
    """
    job = tmp / "codex__m__20260818T010700Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "codex", "harness_label": "Codex CLI",
        "model": {"label": "M", "fingerprint": "fp1", "provider": "openai-compatible"},
    }), encoding="utf-8")

    _write_trial(job, "reported-normally", resolved=True, tokens=1000, suffix="a")
    # Harbor recorded nothing; Codex printed its own totals as it always does.
    _write_trial(job, "harbor-lost-it", resolved=True, tokens=None, suffix="b",
                 log='{"type":"item.completed","item":{"id":"item_1"}}\n'
                     '{"type":"turn.completed","usage":{"input_tokens":2255620,'
                     '"cached_input_tokens":2083198,"output_tokens":140514}}\n')

    run = load_run(job)
    by_task = {t["task_name"]: t for t in run["tasks"]}
    got = by_task["harbor-lost-it"]
    check("the harness's own total is used", got["n_input_tokens"], 2255620)
    check("...including the output half", got["n_output_tokens"], 140514)
    check("...and says where it came from", got["tokens_source"], "agent trace")
    check("a trial Harbor accounted for is untouched",
          by_task["reported-normally"]["tokens_source"], "harbor")
    check("the run says how many it had to recover", run["n_tokens_recovered"], 1)


def test_a_local_endpoint_is_never_given_a_price(tmp: Path) -> None:
    """A server you host bills nothing, so a dollar figure beside it is wrong.

    Harbor prices a trial by looking the model *name* up in LiteLLM's table, which
    cannot tell a local alias from a hosted model. Two measured runs against a free
    llama.cpp box were recorded at $44.31 and $141.19 -- and only for the two
    adapters that compute it, so a cost column ranked those two against blanks.
    """
    def one(provider: str | None) -> float | None:
        job = tmp / f"h__m__{provider or 'none'}"
        job.mkdir(parents=True)
        model = {"label": "M", "fingerprint": "fp1"}
        if provider:
            model["provider"] = provider
        (job / "harness-bench.json").write_text(json.dumps({
            "harness": "h", "model": model,
        }), encoding="utf-8")
        _write_trial(job, "t", resolved=True, tokens=1000, cost=3.99)
        return load_run(job)["tasks"][0]["cost_usd"]

    check("a self-hosted endpoint reports no cost", one("openai-compatible"), None)
    check("a billed provider keeps its cost", one("openrouter"), 3.99)
    # Every manifest written before providers carried the flag came from the
    # local-only setup this rig started as. Inventing a price for those is the
    # failure being fixed, not a conservative default.
    check("a manifest that does not say is treated as free", one(None), None)


def test_a_task_that_ate_the_run_and_failed_is_called_out(tmp: Path) -> None:
    """One task taking half a run's prompt budget and scoring zero is a finding.

    Measured: a claude-code run scored 0.24, and build-pov-ray took 48.4M of its
    96.5M input tokens across 1,471 model calls before scoring zero. It raised no
    exception and hit no timeout the run reported -- it looped until the agent
    clock ran out, and nothing on screen told it apart from a hard task.
    """
    job = tmp / "claude-code__m__20260815T030858Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "claude-code", "model": {"label": "M", "fingerprint": "fp1"},
    }), encoding="utf-8")

    # A 25-task run, the shape this rig actually produces. Scale matters: the
    # threshold is a share of an even split, so what counts as disproportionate
    # depends on how many trials there are to be disproportionate against.
    for i in range(23):
        _write_trial(job, f"ordinary-{i:02d}", resolved=i < 12, tokens=1000,
                     suffix=f"s{i}")
    # A third of the run's prompt spend, and it failed.
    _write_trial(job, "looped-forever", resolved=False, tokens=25000, suffix="big")
    # The same spend, but it worked. Expensive is not runaway, and the
    # per-solve token figures already say so.
    _write_trial(job, "expensive-but-solved", resolved=True, tokens=25000, suffix="ok")

    run = load_run(job)
    names = [w["task_name"] for w in run["runaways"]]
    check("the task that ate the run and failed is reported",
          names, ["looped-forever"])
    check("...with the share that makes it worth reading",
          round(run["runaways"][0]["share"], 2), 0.34)
    check("an expensive solve is not a runaway",
          "expensive-but-solved" in names, False)


def test_a_short_run_has_no_runaways(tmp: Path) -> None:
    """With four tasks, one holding a third of the budget is arithmetic."""
    job = tmp / "h__m__20260815T030858Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "h", "model": {"label": "M", "fingerprint": "fp1"},
    }), encoding="utf-8")
    _write_trial(job, "a", resolved=False, tokens=90000, suffix="a")
    for i in range(3):
        _write_trial(job, f"b{i}", resolved=False, tokens=100, suffix=f"b{i}")
    check("four trials are not enough to call one disproportionate",
          load_run(job)["runaways"], [])


def test_pairing() -> None:
    a = make_run(run_id="a", harness="hermes")
    b = make_run(run_id="b", harness="omp", harness_label="omp")

    check("same budget + subset pairs", len(pairs([a, b])), 1)

    # The case that matters right now: a 16x run beside the 8x baseline.
    b16 = make_run(run_id="b", harness="omp", agent_timeout_multiplier=16.0)
    check("different time budget does NOT pair", len(pairs([a, b16])), 0)

    b_sub = make_run(run_id="b", harness="omp", subset="stratified-10")
    check("different subset does NOT pair", len(pairs([a, b_sub])), 0)

    b_full = make_run(run_id="b", harness="omp", is_partial=False, subset=None)
    check("subset vs full does NOT pair", len(pairs([a, b_full])), 0)

    b_same = make_run(run_id="b", harness="hermes")
    check("same harness does NOT pair", len(pairs([a, b_same])), 0)

    b_model = make_run(run_id="b", harness="omp", model_fingerprint="fp2")
    check("different model does NOT pair", len(pairs([a, b_model])), 0)

    b_empty = make_run(run_id="b", harness="omp", n_done=0, tasks=[])
    check("run with no results does NOT pair", len(pairs([a, b_empty])), 0)

    # Two benchmarks are not one experiment. This is the clause multi-benchmark
    # support needed and did not get: a full Terminal-Bench 2 run and a full
    # aider-polyglot run both carry subset=None and is_partial=False, so every
    # other rule here passes and they paired. Task names rarely collide across
    # datasets, which makes the result worse rather than harmless -- the
    # comparison renders with an empty shared set, reading as two harnesses that
    # agreed on nothing rather than as two runs that were never comparable.
    b_ds = make_run(run_id="b", harness="omp", dataset="aider/aider-polyglot")
    check("different dataset does NOT pair", len(pairs([a, b_ds])), 0)

    # Same dataset still pairs -- the clause must not have cost the real case.
    b_same_ds = make_run(run_id="b", harness="omp", dataset="terminal-bench@2.0")
    check("  but the same dataset still does", len(pairs([a, b_same_ds])), 1)

    # A run recorded before --dataset was captured carries None. Two of those
    # are the same unknown, not two different benchmarks.
    a_old = make_run(run_id="a", harness="hermes", dataset=None)
    b_old = make_run(run_id="b", harness="omp", dataset=None)
    check("two runs with no recorded dataset still pair", len(pairs([a_old, b_old])), 1)
    check("  but not against one that has a dataset", len(pairs([a, b_old])), 0)

    # The index has to offer the benchmarks it holds, or the page cannot scope
    # itself to one.
    index_datasets = index_of([a, b_ds])["datasets"]
    check("the index lists each benchmark once", len(index_datasets), 2)
    check("  labelled from the catalog",
          sorted(d["label"] for d in index_datasets),
          ["Aider Polyglot", "Terminal-Bench 2"])
    check("  with the run-directory slug",
          sorted(d["slug"] for d in index_datasets), ["polyglot", "tb2"])


def test_wilson() -> None:
    lo, hi = wilson_interval(6, 18)
    check("Wilson 6/18 lower", round(lo * 100), 16)
    check("Wilson 6/18 upper", round(hi * 100), 56)
    check("Wilson 0/0 is degenerate", wilson_interval(0, 0), (0.0, 0.0))
    lo, hi = wilson_interval(25, 25)
    check("Wilson 25/25 upper bounded at 1", hi, 1.0)


def test_feed_renders_unknown_event_shapes() -> None:
    """A growing log must never render an empty feed.

    Events used to be dropped whenever their `type` was not one this module
    knew, so a harness whose lines are all typed and none of whose types match
    produced a feed stuck on "waiting for the first output" while the log grew
    to hundreds of KB. That points the reader at the model when the problem is
    here.
    """
    typed_stream = "\n".join([
        '{"type":"session_capture","content":"<redacted>"}',
        '{"type":"assistant","text":"Let me look at the failing test."}',
        '{"type":"metadata","meta":{"input_tokens":10}}',
        '{"type":"error","error":"connection refused"}',
        '{"type":"done"}',
    ])
    entries = parse_entries(typed_stream)
    check("typed-event text is rendered",
          any("failing test" in e["text"] for e in entries), True)
    check("typed-event errors are rendered",
          any("connection refused" in e["text"] for e in entries), True)
    # A redacted session id is not something the agent said.
    check("bookkeeping events are skipped",
          any("<redacted>" in e["text"] for e in entries), False)

    # An event shape nobody has seen still must not render an empty panel.
    check("unknown shapes fall back to raw lines",
          len(parse_entries('{"type":"quux","payload":{"a":1}}')), 1)

    # ...but an event we recognized and chose to hide is not "unreadable", and
    # dumping it raw greets the reader with a wall of braces. opencode opens
    # every session with exactly this one line and nothing else.
    opencode_open = json.dumps({
        "type": "step_start", "timestamp": 1, "sessionID": "ses_x",
        "part": {"id": "prt_x", "messageID": "msg_x", "type": "step-start"},
    })
    check("a lone session-open event renders nothing",
          parse_entries(opencode_open), [])

    # opencode nests the payload under "part", so a top-level scan sees only
    # bookkeeping. The text has to be found inside the envelope.
    wrapped = json.dumps({
        "type": "part_updated",
        "part": {"type": "text", "text": "Reading the checkpoint header."},
    })
    check("text nested in an envelope is found",
          [e["text"] for e in parse_entries(wrapped)],
          ["Reading the checkpoint header."])

    # The formats that already worked must keep working.
    check("omp NDJSON still parses",
          parse_entries(
              '{"type":"message_end","message":'
              '{"role":"assistant","content":"hi"}}')[0]["text"], "hi")
    check("prose logs still parse",
          parse_entries("plain line")[0]["text"], "plain line")
    check("an empty log stays empty", parse_entries("  \n "), [])


def test_feed_reassembles_token_streams() -> None:
    """A harness that streams one event per token must read as prose.

    A token-level stream emits a `content` event per token -- 6,623 of them in
    a two-minute trial, measured -- so one entry per event renders a column of
    single words. The leading space in " user" is the word break, which is why the
    fallback must not strip before these are rejoined.
    """
    stream = "\n".join(
        json.dumps({"type": "content", "content": tok})
        for tok in ["The", " user", " wants", " a", " C", " program", " that", " runs"]
    )
    entries = parse_entries(stream)
    check("a token stream becomes one entry", len(entries), 1)
    check("tokens rejoin with their spacing",
          entries[0]["text"], "The user wants a C program that runs")

    # Whole-message harnesses must not get welded together by the same path.
    lines = "\n".join(
        json.dumps({"type": "note", "text": f"step {i}: doing a piece of work"})
        for i in range(8)
    )
    entries = parse_entries(lines)
    check("whole messages stay separate", len(entries), 8)

    # Tool activity is most of what a feed is for; it must survive an
    # unfamiliar harness rather than being dropped as an unknown type.
    tools = "\n".join([
        '{"type":"tool_use","name":"Bash","input":{"command":"ls /app"}}',
        '{"type":"tool_result","name":"Bash","status":"error","output":"not found"}',
        '{"type":"turn_usage","turn":1,"input_tokens":10,"output_tokens":2}',
    ])
    entries = parse_entries(tools)
    check("tool calls render", [e["kind"] for e in entries], ["tool", "result"])
    check("the tool name and args render",
          entries[0]["text"], '→ Bash\n{"command": "ls /app"}')
    check("a failed tool shows its status",
          entries[1]["text"], "Bash (error)\nnot found")

    # opencode buries all of that: the name sits on `part`, the input, output
    # and status one level deeper on `part.state`. Reading only the outer event
    # rendered "→ tool" with no name and no arguments.
    nested = json.dumps({
        "type": "tool_use", "timestamp": 1, "sessionID": "ses_x",
        "part": {
            "type": "tool", "tool": "write", "callID": "c1",
            "state": {
                "status": "completed",
                "input": {"filePath": "/app/gpt2.c"},
                "output": "Wrote file successfully.",
                "title": "app/gpt2.c",
            },
        },
    })
    entries = parse_entries(nested)
    check("a nested tool call finds its name", len(entries), 1)
    check("...and its arguments",
          '"filePath": "/app/gpt2.c"' in entries[0]["text"], True)
    # A call reported only once it has finished carries its own result;
    # dropping it would show what the agent tried and never whether it worked.
    check("...and the result it already carries",
          "completed → Wrote file successfully." in entries[0]["text"], True)

    # An event-sourced session log names its events with a slash and nests the
    # message under `data`. dsh writes one, and it is the only record its run
    # leaves: `dsh --profile headless` prints nothing until its final answer,
    # so a feed that cannot read this shape stays blank for the whole trial and
    # reads as a hung agent.
    #
    # The log holds the same turn twice -- the raw provider stream chunk by
    # chunk, then the assembled message that commits the step -- so the chunks
    # are dropped the way `message_update` is, or every sentence prints twice.
    session_log = "\n".join([
        json.dumps({"type": "assistant/chunk", "seq": 1, "data": {
            "turn": 1, "step": 1,
            "chunk": {"type": "text-delta", "delta": "Looking"}}}),
        json.dumps({"type": "assistant/chunk", "seq": 2, "data": {
            "turn": 1, "step": 1,
            "chunk": {"type": "usage", "usage": {"inputTokens": 12}}}}),
        json.dumps({"type": "assistant/message", "seq": 3, "data": {
            "turn": 1, "step": 1, "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Looking at the tests."},
                {"type": "tool-call", "id": "c1", "name": "bash",
                 "arguments": '{"command":"pytest -x"}'}]}}}),
        # Stored with the *user* role, so reading the role rather than the
        # event type would file every command's output as the prompt.
        json.dumps({"type": "tool/result", "seq": 4, "data": {
            "turn": 1, "step": 1, "message": {"role": "user", "content": [
                {"type": "tool-result", "callId": "c1", "isError": False,
                 "content": [{"type": "text", "text": "1 failed, 4 passed"}]}]}}}),
    ])
    entries = parse_entries(session_log)
    check("a session log renders its turn",
          [e["kind"] for e in entries], ["tool", "result"])
    check("...the message text once, not twice",
          entries[0]["text"].count("Looking at the tests."), 1)
    check("...naming the tool it called",
          "→ bash" in entries[0]["text"], True)
    check("...and the result nested inside its block",
          entries[1]["text"], "1 failed, 4 passed")

    # Step boundaries bracket every opencode turn and carry nothing to read.
    check("step boundaries render nothing",
          parse_entries(json.dumps({
              "type": "step_finish",
              "part": {"type": "step-finish", "tokens": {"total": 9589}},
          })), [])

    # Claude Code emits a `thinking_tokens` progress counter once per couple of
    # tokens of reasoning: measured at 32,505 of 32,666 lines and 91% of the
    # bytes of one 6.6MB trial log. It is typed as a plain "system", so only the
    # subtype separates it from something worth reading.
    #
    # Two failures compounded here, and the second is the one that emptied the
    # panel: each line was dumped as raw JSON because it was recognizable but
    # carried no prose, and those dumps then counted toward read_feed's text
    # budget, so the window never grew past 24KB and the real transcript a few
    # hundred KB back was never reached.
    counter = json.dumps({
        "type": "system", "subtype": "thinking_tokens",
        "estimated_tokens": 521, "estimated_tokens_delta": 3,
        "uuid": "9fc5604b", "session_id": "2d9c07ac",
    })
    check("a token counter renders nothing", parse_entries(counter), [])
    check("...and is not dumped as unreadable JSON",
          "".join(e["text"] for e in parse_entries(counter)), "")
    # It must be suppressed rather than merely unrendered, or the surrounding
    # prose is lost with it.
    check("...while the reasoning around it survives",
          [e["text"] for e in parse_entries(
              counter
              + "\n" + json.dumps({"type": "assistant", "text": "Let me compile it."})
              + "\n" + counter
          )],
          ["Let me compile it."])

    # --debug-capture sets RUST_LOG so a transport failure can be attributed
    # afterwards. Harnesses that ignore it cost nothing; Codex is Rust and does
    # not ignore it, logging one line per SSE event -- so per token. Measured on
    # one trial: 65,965 lines and 97% of a 24.9MB log. None of it is JSON, so it
    # landed in the prose path and the panel rendered 23KB blobs of tracing
    # where the agent's output belonged.
    #
    # The filter is on the *view* only; the log on disk keeps every line,
    # because that record is the whole point of the flag. WARN and ERROR stay
    # visible for the same reason -- a reset stream announces itself there, and
    # dropping those would remove exactly what the diagnostics were turned on
    # to catch.
    tracing = "\n".join([
        '2026-08-11T17:06:19.316005Z  INFO codex_otel.log_only:'
        ' event.name="codex.sse_event" event.kind=response.reasoning_text.delta',
        "2026-08-11T17:06:19.400000Z DEBUG hyper_util::client: connecting to"
        " 192.0.2.10:8002",
        "2026-08-11T17:06:19.500000Z TRACE reqwest::async_impl: reading body",
        "2026-08-11T17:06:20.000000Z  WARN codex_core: stream disconnected",
        "2026-08-11T17:06:21.000000Z ERROR codex_core: connection reset by peer",
        "Process exited with code 0",
    ])
    kept = "\n".join(e["text"] for e in parse_entries(tracing))
    check("per-token INFO tracing is dropped", "sse_event" in kept, False)
    check("...and DEBUG", "connecting to" in kept, False)
    check("...and TRACE", "reading body" in kept, False)
    # The half that must not regress: these are why diagnostics get enabled.
    check("a WARN survives", "stream disconnected" in kept, True)
    check("an ERROR survives", "connection reset by peer" in kept, True)
    check("ordinary agent output survives", "Process exited" in kept, True)

    # A tracing record is one line only until a field value contains a newline.
    # Codex's telemetry preview does, so the record's remaining fields print
    # with no timestamp in front of them and the rule above -- anchored on the
    # timestamp -- never sees them. They were the whole of what still looked
    # like machine exhaust in the panel.
    trailer = "\n".join([
        "[... telemetry preview truncated ...] mcp_server= mcp_server_origin="
        " event.timestamp=2026-08-11T17:22:26.027Z"
        " conversation.id=019ff1d8-b6bf-7c02-bca3-c517eeacd8fb"
        ' app.version=0.147.0 auth_mode="ApiKey" originator=codex_exec',
        " mcp_server= mcp_server_origin= event.timestamp=2026-08-11T17:23:15.277Z"
        " conversation.id=019ff1d8-b6bf-7c02-bca3-c517eeacd8fb",
        # The agent really did run `od` on a binary checkpoint. A hex dump is
        # its output, not exhaust, and hiding it would hide the work.
        "Output:",
        "000000 03 ef f5 3e c0 82 06 bf 90 c8 db be 36 e5 52 be  >...>........6.R.<",
    ])
    kept = "\n".join(e["text"] for e in parse_entries(trailer))
    check("a field-list trailer is dropped", "conversation.id=" in kept, False)
    check("...including the truncation marker",
          "telemetry preview truncated" in kept, False)
    check("but the agent's own hex dump is kept", "000000 03 ef f5" in kept, True)


def test_feed_reads_far_enough_back(tmp: Path) -> None:
    """The feed's budget is characters of text, not bytes of log.

    A token-level stream spends ~90 bytes of JSON envelope per token, so only
    about 5% of what is read is prose. Budgeting the read in bytes gave the
    panel ~1,200 characters -- roughly five seconds of output against a
    five-second refresh, so every poll replaced the whole panel and there was
    nothing to scroll back through.
    """
    verbose = tmp / "verbose.txt"
    verbose.write_text("\n".join(
        json.dumps({
            "type": "content", "content": f" word{i}",
            "schema_version": 1, "schema": "some.exec-stream",
        })
        for i in range(20_000)
    ), encoding="utf-8")

    entries = read_feed(verbose)
    prose = sum(len(e["text"]) for e in entries)
    check("a verbose stream still fills the panel",
          prose >= TARGET_TEXT_CHARS, True)
    # The whole point is overlap between polls: the panel must hold much more
    # than one refresh worth of output, or the reader never sees it twice.
    check("it reads well past one TAIL_BYTES window",
          prose > TAIL_BYTES * 0.05 * 4, True)

    # A prose log is mostly text, so it must not pay for any of that.
    prose_log = tmp / "prose.txt"
    prose_log.write_text("thinking hard\n\nthen acting\n", encoding="utf-8")
    check("a short log reads once and stops",
          [e["text"] for e in read_feed(prose_log)],
          ["thinking hard", "then acting"])
    check("a missing log is not an error", read_feed(tmp / "gone.txt"), [])

    # A log that is almost entirely filtered. With --debug-capture, Codex
    # writes ~97% Rust tracing; one turn of reasoning made a 10MB log whose
    # last 400KB -- the old ceiling -- held 1,115 lines and not one of agent
    # output. Reaching content in a log that is mostly not content needs a
    # scan budget rather than a render budget, so the two are now separate.
    noisy = tmp / "noisy.txt"
    noise = ('2026-08-11T17:06:19.316005Z  INFO codex_otel.log_only: '
             'event.name="codex.sse_event" event.kind=response.reasoning_text.delta\n')
    with noisy.open("w", encoding="utf-8") as handle:
        handle.write("the agent said something worth reading\n")
        handle.write(noise * 6_000)          # ~700KB, well past the old ceiling
    check("content survives being buried in tracing",
          any("worth reading" in e["text"] for e in read_feed(noisy)), True)

    # And when it is buried past even the scan ceiling, the panel must say so.
    # Silence reads as a hung agent, which is the one thing this module exists
    # not to do.
    buried = tmp / "buried.txt"
    with buried.open("w", encoding="utf-8") as handle:
        handle.write("unreachable output\n")
        handle.write(noise * (MAX_TAIL_BYTES // len(noise) + 2_000))
    entries = read_feed(buried)
    check("a fully buried log explains itself instead of going blank",
          len(entries) == 1 and entries[0]["kind"] == "meta", True)
    check("...and names the cause", "tracing" in entries[0]["text"], True)


def _plus_seconds(stamp: str, seconds: int) -> str:
    from datetime import datetime, timedelta
    base = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (base + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _write_trial(job: Path, task: str, *, resolved: bool, log: str = "",
                 exception: str | None = None, message: str = "",
                 checks: tuple[int, int] | None = None, suffix: str = "x",
                 started: str = "2026-08-10T04:00:00Z",
                 finished: str = "2026-08-10T04:03:00Z",
                 agent_s: int = 0, spliced: str = "",
                 tokens: int | None = None, no_reward: bool = False,
                 cost: float | None = None) -> None:
    trial = job / f"{task}__{suffix}"
    (trial / "agent").mkdir(parents=True)
    if checks:
        passed, total = checks
        (trial / "verifier").mkdir(parents=True)
        (trial / "verifier" / "ctrf.json").write_text(json.dumps({
            "results": {"tests": [
                {"name": f"check_{i}", "status": "passed" if i < passed else "failed"}
                for i in range(total)
            ]}
        }), encoding="utf-8")
    if log:
        (trial / "agent" / "agent.txt").write_text(log, encoding="utf-8")
    result = {
        "task_name": task,
        "verifier_result": {"rewards": {"reward": 1.0 if resolved else 0.0}},
        "started_at": started,
        "finished_at": finished,
    }
    # A trial the verifier could not score has no verifier_result at all --
    # Harbor raises before building one. Passing a reward dict anyway would
    # test a shape that never reaches disk.
    if no_reward:
        result.pop("verifier_result")
    # Harbor writes the block either way; a harness that reported no usage
    # leaves its fields null, which is what a real run looked like.
    result["agent_result"] = {
        "n_input_tokens": None if tokens is None else tokens * 4,
        "n_cache_tokens": None,
        "n_output_tokens": tokens,
        "cost_usd": cost,
    }
    if agent_s:
        result["agent_execution"] = {
            "started_at": started,
            "finished_at": _plus_seconds(started, agent_s),
        }
    if spliced:
        (trial / "rerun.json").write_text(
            json.dumps({"supersedes": spliced}), encoding="utf-8"
        )
    if exception:
        result["exception_info"] = {
            "exception_type": exception, "exception_message": message,
        }
    (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")


def test_the_clock_starts_when_the_model_does(tmp: Path) -> None:
    """A run's headline time is the model's time, not the container's.

    A trial is four phases and only one of them is the model working: pull the
    image, build the container, install the harness, run the agent, verify.
    Measured on a real trial, setup was 72 seconds -- and a cold image pull is
    minutes. Charging that to the harness under test is not a rounding error in
    the wrong direction, it is the *same* cost for every harness on a dataset,
    so folding it in narrows every gap the rig exists to measure.

    Both clocks are kept. Wall clock is what the run cost you; agent time is
    what the harness is answerable for.
    """
    job = tmp / "someharness__m__20260811T000000Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "someharness", "harness_label": "Some Harness",
        "model": {"label": "M", "fingerprint": "fp1", "total_slots": 1},
        "n_concurrent": 1, "n_concurrent_agents": 1,
        "started_at": "2026-08-10T04:00:00+00:00",
    }), encoding="utf-8")
    (job / "result.json").write_text(json.dumps({
        "finished_at": "2026-08-09T23:03:00",
    }), encoding="utf-8")

    # Two trials, three minutes of wall clock, two minutes of agent between
    # them. The other minute is setup and verifiers.
    _write_trial(job, "a", resolved=True, agent_s=60)
    _write_trial(job, "b", resolved=False, agent_s=60,
                 started="2026-08-10T04:01:30Z", finished="2026-08-10T04:03:00Z")

    run = load_run(job)
    check("the run reports the model's own time", run["agent_total_s"], 120.0)
    check("...and keeps the wall clock beside it", run["wall_clock_s"], 180.0)
    check("...with the ratio between them", round(run["llm_busy_pct"], 1), 66.7)


def test_a_trial_in_flight_advances_the_clock_but_its_setup_does_not(
    tmp: Path,
) -> None:
    """The model clock has to move during a trial, and only once the model does.

    Agent time comes from Harbor's own phase timestamps, which it writes when a
    trial *ends*. Reading only those would freeze the number for the length of a
    trial -- up to an hour -- so a working run would look like a stalled one.
    The trial in flight is measured from the agent log instead, which Harbor
    creates when it launches the agent, so the container work before it is not
    on the model's account.
    """
    job = tmp / "someharness__m__20260811T000000Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "someharness", "harness_label": "Some Harness",
        "model": {"label": "M", "fingerprint": "fp1"},
        "started_at": "2026-08-10T04:00:00+00:00",
    }), encoding="utf-8")
    _write_trial(job, "done", resolved=True, agent_s=60)

    # In flight: a trial directory with no result.json. It exists for a real
    # interval before the agent log appears, which is the setup this test is
    # about -- so create the directory, wait, then start the log.
    live = job / "running__y"
    (live / "agent").mkdir(parents=True)
    setup_seconds = 1.5
    time.sleep(setup_seconds)
    (live / "agent" / "agent.txt").write_text("thinking\n", encoding="utf-8")

    # Let the agent clock actually start before reading it. Without this the
    # log is measured microseconds after it was created, and whether the live
    # trial contributes anything at all comes down to the agreement between
    # time.time() and the filesystem's own timestamp -- which on Windows is
    # close enough to zero to land on either side. It did: `agent_total_s >
    # 60.0` failed once on windows/py3.12 while every other matrix cell passed,
    # on a commit that touched none of this. The interval below is not part of
    # what is being tested; it just has to be larger than the clocks disagree
    # by. The setup gap the test is actually about is unaffected, since this
    # moves the agent clock and the directory's age together.
    time.sleep(0.25)

    trial_age = time.time() - (live).stat().st_ctime
    run = load_run(job)
    live_s = run["agent_total_s"] - 60.0

    check("the finished trial still counts in full",
          run["agent_total_s"] > 60.0, True)
    # The whole point: the directory is older than the agent clock by the setup
    # interval. Asserted as a gap rather than an exact figure, because the only
    # thing under test is that setup is outside the number.
    check("...and the trial in flight adds only its agent time",
          round(trial_age - live_s, 1) >= 1.0, True)
    check("...which is not the age of its directory",
          live_s < trial_age, True)


def test_appending_to_a_log_does_not_move_the_agent_start(tmp: Path) -> None:
    """The start of the agent phase must not drift as the agent writes.

    `st_ctime` is the obvious way to ask when a file was created and is right on
    exactly one platform: on Windows it is the creation time, on Linux it is the
    inode *change* time, which every append moves forward. Reading it there
    would report a trial two hours in as seconds old -- a clock that resets
    itself whenever the thing it is timing makes progress, which is the one
    failure mode that would never look like a bug in a timestamp.
    """
    trial = tmp / "task__z"
    (trial / "agent").mkdir(parents=True)
    log = trial / "agent" / "agent.txt"
    log.write_text("first\n", encoding="utf-8")

    started = agent_started_at(log)
    time.sleep(1.1)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("more output\n")
    after = agent_started_at(log)

    check("a growing log keeps its start time",
          round(abs(after - started), 1), 0.0)
    check("...which is when it was created, not when it was last written",
          after < log.stat().st_mtime - 1.0, True)


def test_throughput_stops_the_clock_when_a_run_ends(tmp: Path) -> None:
    """A finished run is not a running one, and only `stopped_at` said so.

    bench/throughput.py ended its window at `now` unless the manifest carried a
    `stopped_at`, which the supervisor writes only when a run is *killed*. Every
    run that simply finished -- nearly all of them -- was therefore treated as
    still in flight, so its elapsed grew without bound long after the work was
    over, dragging min/trial up and llm busy% down with it.

    Measured on a real seven-harness sweep a few hours after it ended: a 3m36s
    run of five trials reported 3.4h elapsed, 40.3 min/trial and 1% utilization.
    Every number in the table was wrong for any run not killed by hand.

    The completion signal is the job result's `finished_at`, used for its
    *presence* only. Harbor writes that one in naive local time while the
    manifest records aware UTC, so subtracting across the two shifts the answer
    by the machine's offset -- five hours here. This fixture makes the naive
    stamp deliberately absurd to prove it is never used as a value.
    """
    import contextlib
    import io

    from bench import throughput

    job = tmp / "someharness__m__tb2__full__20260815T091000Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "someharness", "harness_label": "Some Harness",
        "model": {"label": "M", "fingerprint": "fp1"},
        "n_concurrent": 2, "n_concurrent_agents": 1,
        "agent_timeout_multiplier": 8.0,
        # Aware UTC, as bench.runner records it.
        "started_at": "2026-08-15T09:10:00+00:00",
    }), encoding="utf-8")

    # Two trials, six minutes of wall clock, two of it generating.
    for name, end, agent in (
        ("alpha", "2026-08-15T09:13:00Z", ("09:11:00", "09:12:00")),
        ("beta", "2026-08-15T09:16:00Z", ("09:14:00", "09:15:00")),
    ):
        trial = job / f"{name}__x"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(json.dumps({
            "task_name": name,
            "verifier_result": {"rewards": {"reward": 0.0}},
            "started_at": "2026-08-15T09:10:00Z",
            "finished_at": end,
            "agent_execution": {
                "started_at": f"2026-08-15T{agent[0]}Z",
                "finished_at": f"2026-08-15T{agent[1]}Z",
            },
        }), encoding="utf-8")

    # Harbor's job-level result: naive local time, hours away from the aware
    # stamps above. Its presence ends the run; its value must never be used.
    (job / "result.json").write_text(json.dumps({
        "n_total_trials": 2,
        "started_at": "2026-08-15T04:10:00.000000",
        "finished_at": "2026-08-15T04:16:00.000000",
    }), encoding="utf-8")

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        throughput.main(["--runs-dir", str(job.parent)])
    line = [ln for ln in out.getvalue().splitlines()
            if ln.startswith("someharness")][0]

    check("a finished run is not reported as running",
          line.split()[-1], "complete")
    # 09:10 to the last trial at 09:16 is six minutes, whatever the clock says
    # now and whatever the naive local stamp claims.
    check("...and its elapsed is its own, not time since it ended",
          line.split()[-4], "0.1h")
    check("...so min/trial is the real figure", line.split()[-3], "3.0")
    # Two of the six minutes were spent generating.
    check("...and utilization is measured over that window",
          line.split()[-2], "33%")


def test_throughput_still_runs_the_clock_on_a_live_run(tmp: Path) -> None:
    """The other half: a run still going has to keep measuring to now.

    Freezing at the last completed trial would report a run whose next trial is
    still working as though it had finished, which is the failure the original
    `now` behaviour existed to prevent. Only the *ending* condition was wrong.
    """
    import contextlib
    import io
    from datetime import UTC, datetime, timedelta

    from bench import throughput

    started = datetime.now(UTC) - timedelta(minutes=30)
    job = tmp / "someharness__m__tb2__full__live"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "someharness", "harness_label": "Some Harness",
        "model": {"label": "M", "fingerprint": "fp1"},
        "agent_timeout_multiplier": 8.0,
        "started_at": started.isoformat(),
    }), encoding="utf-8")
    trial = job / "alpha__x"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({
        "task_name": "alpha",
        "verifier_result": {"rewards": {"reward": 0.0}},
        "started_at": started.isoformat(),
        "finished_at": (started + timedelta(minutes=5)).isoformat(),
    }), encoding="utf-8")
    # Ten of twenty trials done, and no job-level result yet: still going.
    (job / "result.json").write_text(json.dumps({"n_total_trials": 20}),
                                     encoding="utf-8")

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        throughput.main(["--runs-dir", str(job.parent)])
    line = [ln for ln in out.getvalue().splitlines()
            if ln.startswith("someharness")][0]

    check("a live run still reads as running", line.split()[-1], "running")
    # Half an hour in, not the five minutes its one finished trial took.
    check("...and is measured to now", line.split()[-4], "0.5h")


def test_a_submission_the_verifier_could_not_score_is_a_failure(tmp: Path) -> None:
    """A build failure is a wrong answer, not a broken rig.

    Benchmarks that compile their tests against the agent's code -- every
    aider-polyglot task in C++, Go, Rust or Java -- cannot score code that does
    not compile, so Harbor raises RewardFileNotFoundError. That is what "did not
    implement it" looks like there, which makes it the *normal* failure rather
    than an exceptional one.

    Classified as an error it wore the glyph reserved for the harness falling
    over, so a matrix of red exclamation marks read as infrastructure failure
    when it meant the model could not code. Observed on a live 225-task
    aider-polyglot run: 2 of the first 5 trials, both C++, both because the test
    file failed to compile against the agent's missing declarations.

    The exception is only reachable *after* the verifier's tests have run --
    Verifier.verify executes them, then looks for a reward file -- so it is
    proof the work was evaluated, on any dataset.
    """
    job = tmp / "someharness__m__polyglot__full__20260815T000000Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "someharness", "harness_label": "Some Harness",
        "dataset": "aider/aider-polyglot",
        "model": {"label": "M", "fingerprint": "fp1"},
    }), encoding="utf-8")

    _write_trial(job, "solved", resolved=True)
    _write_trial(job, "scored-zero", resolved=False)
    # The agent never defined the symbols the test file needs, so the build
    # failed and no reward was written.
    _write_trial(job, "did-not-build", resolved=False, no_reward=True,
                 exception="RewardFileNotFoundError",
                 message="No reward file found at .../verifier/reward.txt")
    # A genuine crash still has to read as one, or the distinction is worthless.
    _write_trial(job, "harness-crashed", resolved=False,
                 exception="NonZeroAgentExitCodeError",
                 log="panic: index out of range\n")

    run = load_run(job)
    by_task = {t["task_name"]: t for t in run["tasks"]}

    unscorable = by_task["did-not-build"]
    check("an unscorable submission is not an error",
          unscorable["error_type"], None)
    check("...but says why it has no score",
          unscorable["no_reward_reason"], "RewardFileNotFoundError")
    check("...and did not pass", unscorable["resolved"], False)
    # The whole point: it is the harness's failure, so it must stay in the
    # denominator. Excluding it would quietly inflate the pass rate.
    check("...and stays the harness's failure", unscorable["fault"], "harness")

    check("a real crash is still an error",
          by_task["harness-crashed"]["error_type"], "NonZeroAgentExitCodeError")
    check("...and carries no no-score reason",
          by_task["harness-crashed"]["no_reward_reason"], None)
    check("a scored zero has neither",
          (by_task["scored-zero"]["error_type"],
           by_task["scored-zero"]["no_reward_reason"]), (None, None))

    check("nothing is dropped from the denominator", run["n_unscored"], 0)
    check("every trial is still counted", run["n_attempted"], 4)
    check("the pass rate counts it as a failure",
          round(run["pass_rate"], 4), round(1 / 4, 4))
    # Only the genuine crash is an error.
    check("only the real crash is counted as an error", run["n_errors"], 1)
    check("...and the tally names only it",
          run["error_types"], {"NonZeroAgentExitCodeError": 1})


def test_a_run_of_only_build_failures_reports_no_errors(tmp: Path) -> None:
    """The job-level fallback must not undo the distinction.

    `load_run` falls back to Harbor's job-level exception_stats whenever the
    per-trial pass found no errors -- which is exactly what happens once
    unscorable submissions stop counting as errors, and Harbor records those at
    the job level too. Without filtering the fallback there as well, a run of
    nothing but build failures reports every one of them as a broken trial
    again, which is the bug this whole distinction exists to remove.
    """
    job = tmp / "someharness__m__polyglot__full__20260815T010000Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "someharness", "harness_label": "Some Harness",
        "dataset": "aider/aider-polyglot",
        "model": {"label": "M", "fingerprint": "fp1"},
    }), encoding="utf-8")
    for task in ("cpp-one", "cpp-two"):
        _write_trial(job, task, resolved=False, no_reward=True,
                     exception="RewardFileNotFoundError")
    # Harbor's own job-level record of the same two trials.
    (job / "result.json").write_text(json.dumps({
        "n_total_trials": 2,
        "stats": {"evals": {"e": {"exception_stats": {
            "RewardFileNotFoundError": ["cpp-one__x", "cpp-two__x"],
        }}}},
    }), encoding="utf-8")

    run = load_run(job)
    check("a run of build failures reports no errors", run["n_errors"], 0)
    check("...and an empty error tally", run["error_types"], {})
    check("...while still scoring zero", run["pass_rate"], 0.0)
    check("...over every trial", run["n_attempted"], 2)

    # A setup failure has to survive the same filter: it never reached a
    # verifier, so it can never raise this, and it is the case the fallback
    # exists for.
    (job / "result.json").write_text(json.dumps({
        "n_total_trials": 2,
        "stats": {"evals": {"e": {"exception_stats": {
            "RewardFileNotFoundError": ["cpp-one__x"],
            "AgentSetupError": ["cpp-two__x"],
        }}}},
    }), encoding="utf-8")
    check("a genuine setup failure still surfaces",
          load_run(job)["error_types"], {"AgentSetupError": 1})


def test_partial_credit_excludes_the_tasks_it_solved(tmp: Path) -> None:
    """A solved task passes every check, so including it mostly restates the rate.

    Measured on four real runs: one showed 62.1% of checks overall, of which 43
    of 43 came from solved tasks, and its partial credit on everything else was
    35.0%. Ranking by the all-inclusive figure put that run first; ranking by
    what it salvaged from its failures put it last. Both numbers are reported,
    and the one the panel leads with is the half the pass rate has not already
    given you.
    """
    job = tmp / "someharness__m__20260811T000000Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "someharness", "harness_label": "Some Harness",
        "model": {"label": "M", "fingerprint": "fp1", "total_slots": 1},
        "n_concurrent": 1, "n_concurrent_agents": 1,
    }), encoding="utf-8")

    # Two solved (6 of 6 each, by definition) and two missed (1 of 4, 3 of 4).
    _write_trial(job, "solved-a", resolved=True, checks=(6, 6))
    _write_trial(job, "solved-b", resolved=True, checks=(6, 6))
    _write_trial(job, "missed-a", resolved=False, checks=(1, 4))
    _write_trial(job, "missed-b", resolved=False, checks=(3, 4))

    run = load_run(job)
    check("the all-inclusive total still counts everything",
          (run["n_checks_passed"], run["n_checks_total"]), (16, 20))
    check("partial credit counts only the tasks it missed",
          (run["n_checks_missed_passed"], run["n_checks_missed_total"]), (4, 8))
    check("...and is rated against that denominator",
          run["missed_check_rate"], 0.5)
    check("...over the right number of tasks", run["n_missed"], 2)
    # The point of the split: the two numbers disagree, and the all-inclusive
    # one is the flattering half.
    check("the all-inclusive rate is the more flattering one",
          run["check_rate"] > run["missed_check_rate"], True)


def test_a_run_that_solved_everything_has_no_partial_credit(tmp: Path) -> None:
    """Nothing missed means no denominator -- and no rate rather than a 0%."""
    job = tmp / "perfect__m__20260811T010000Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "perfect", "harness_label": "Perfect",
        "model": {"label": "M", "fingerprint": "fp1", "total_slots": 1},
    }), encoding="utf-8")
    _write_trial(job, "solved-a", resolved=True, checks=(3, 3))

    run = load_run(job)
    check("no missed tasks means no partial-credit denominator",
          run["n_checks_missed_total"], 0)
    check("...and no rate at all, rather than a misleading zero",
          run["missed_check_rate"], None)


def test_endpoint_faults_leave_the_denominator(tmp: Path) -> None:
    """A dropped connection is infrastructure, not evidence about the harness.

    Counting it as a failure is how a working adapter comes to look broken: the
    harness is charged for the benchmark's own plumbing falling over.
    """
    job = tmp / "someharness__m__20260810T040813Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "someharness", "harness_label": "Some Harness",
        "model": {"label": "M", "fingerprint": "fp1", "total_slots": 1},
        "n_concurrent": 2, "n_concurrent_agents": 1, "max_retries": 1,
    }), encoding="utf-8")

    _write_trial(job, "solved-one", resolved=True)
    _write_trial(job, "genuinely-failed", resolved=False)
    # The harness crashed on its own: no endpoint in sight.
    _write_trial(job, "harness-crashed", resolved=False,
                 exception="NonZeroAgentExitCodeError",
                 log="panic: index out of range\n")
    # The connection to the model died mid-turn. Same exception type -- only
    # the log distinguishes them, which is exactly why the log is read.
    _write_trial(job, "lost-endpoint", resolved=False,
                 exception="NonZeroAgentExitCodeError",
                 log='{"type":"error","error":"error sending request for url '
                     '(http://model-endpoint.invalid:8000/v1/chat/completions)"}\n')

    run = load_run(job)
    faults = {t["task_name"]: t["fault"] for t in run["tasks"]}
    check("a dropped connection is an endpoint fault",
          faults["lost-endpoint"], "transport")
    check("a harness crash is still the harness's",
          faults["harness-crashed"], "harness")
    check("a clean pass has no fault", faults["solved-one"], None)

    check("the unscored trial leaves the denominator", run["n_done"], 3)
    check("...and is reported, not hidden", run["n_unscored"], 1)
    check("every attempt is still counted", run["n_attempted"], 4)
    check("pass rate uses the reduced denominator",
          round(run["pass_rate"], 4), round(1 / 3, 4))
    check("the harness is still charged for its own crash", run["n_errors"], 1)
    check("the run still reads as complete", run["status"], "complete")

    # A task one run never got a fair attempt at must not show up as a
    # difference between the harnesses.
    other = tmp / "omp__m__20260810T050000Z"
    other.mkdir(parents=True)
    (other / "harness-bench.json").write_text(json.dumps({
        "harness": "omp", "harness_label": "omp",
        "model": {"label": "M", "fingerprint": "fp1"},
    }), encoding="utf-8")
    for task in ("solved-one", "genuinely-failed", "harness-crashed", "lost-endpoint"):
        _write_trial(other, task, resolved=True)

    h2h = head_to_head(load_run(other), run)
    check("an unscored task is not a head-to-head disagreement",
          "lost-endpoint" in h2h["only_a"], False)
    check("...and is not in the shared set at all", h2h["n_shared"], 3)


def test_a_grafted_rerun_scores_but_does_not_stretch_the_clock(tmp: Path) -> None:
    """A trial re-run later and grafted in counts, but did not happen in the run.

    Re-running one task after a fix and dropping the result beside the original
    is the only way to correct a cell without re-running the other 24. But the
    graft carries its own clock: measured here, an opencode run that took 3.7
    hours reported 27.1 once a next-day re-run landed in it, and its LLM-busy
    share fell from 93% to 13% -- both wrong, and wrong in the direction that
    flatters nothing. The score is the graft's; the stopwatch is the run's.
    """
    job = tmp / "someharness__m__20260811T000000Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "someharness", "harness_label": "Some Harness",
        "model": {"label": "M", "fingerprint": "fp1", "total_slots": 1},
        "n_concurrent": 1, "n_concurrent_agents": 1,
        "started_at": "2026-08-10T04:00:00+00:00",
    }), encoding="utf-8")
    # Present so the run reads as finished; never used as a timestamp.
    (job / "result.json").write_text(json.dumps({
        "finished_at": "2026-08-09T23:03:00",
    }), encoding="utf-8")

    _write_trial(job, "kept", resolved=True, checks=(2, 2), agent_s=120)
    # Failed on the day, then re-run 24h later and grafted in. Same task name,
    # so the graft is the attempt that counts -- and it sorts last.
    _write_trial(job, "fixed", resolved=False, checks=(0, 2), exception="Boom",
                 agent_s=60)
    _write_trial(job, "fixed", resolved=False, checks=(2, 2), suffix="zz-rerun",
                 started="2026-08-11T04:00:00Z", finished="2026-08-11T04:30:00Z",
                 agent_s=1800, spliced="fixed__x")

    run = load_run(job)

    check("the graft is the attempt that counts", run["n_checks_passed"], 4)
    check("...and its error does not follow it", run["n_errors"], 0)
    check("the clock still ends with the run",
          run["wall_clock_s"], 180.0)
    # 120s of a 180s window. Counting the graft's 1800s here would report the
    # model as busy 1067% of the time.
    check("busy share is measured over the run's own work",
          round(run["llm_busy_pct"], 1), 66.7)
    check("and the graft is flagged as one",
          [t["task_name"] for t in run["tasks"] if t["spliced"]], ["fixed"])


def test_output_tokens_are_reported_per_solve_and_per_trial(tmp: Path) -> None:
    """Per-solve alone flatters a harness that fails expensively.

    Measured on one real sweep: claude-code averaged 26k output tokens per
    solved task and 125k per trial, because the tasks it solved were the cheap
    ones and the ones it lost ran to six figures. opencode's two figures were
    within 10% of each other. Ranking by per-solve makes the first look like
    the economical one; the gap between the two numbers is the actual finding,
    so both are reported.
    """
    job = tmp / "someharness__m__20260811T000000Z"
    job.mkdir(parents=True)
    (job / "harness-bench.json").write_text(json.dumps({
        "harness": "someharness", "harness_label": "Some Harness",
        "model": {"label": "M", "fingerprint": "fp1", "total_slots": 1},
        "n_concurrent": 1, "n_concurrent_agents": 1,
    }), encoding="utf-8")

    _write_trial(job, "cheap-win", resolved=True, tokens=10_000)
    _write_trial(job, "costly-loss", resolved=False, tokens=200_000)
    # Finished, but the harness never reported usage for it.
    _write_trial(job, "no-usage", resolved=False, tokens=None)

    run = load_run(job)

    check("per solve counts only what worked",
          run["mean_output_tokens_per_solve"], 10_000)
    check("per trial counts what it spent losing too",
          run["mean_output_tokens_per_trial"], 105_000)
    # Folding the unreported trial in as zero would report 70,000 here and
    # reward whichever harness logs its usage worst.
    check("a trial with no usage is left out, not zeroed",
          run["n_token_samples"], 2)


def test_prompt_totals_are_repaired_when_input_excludes_cache() -> None:
    """Runs written before 0.1.10 recorded hermes/opencode input net of cache.

    Cache reads are a subset of the prompt, so `n_input_tokens < n_cache_tokens`
    cannot happen in a correctly-recorded trial -- which makes it a safe marker
    for the old form. Repairing on read matters because the runs already on disk
    are the ones being compared, and understating a prompt total by 18x lands
    directly in cost-per-solve.
    """
    old = {"agent_result": {"n_input_tokens": 1_870_000,
                            "n_cache_tokens": 32_600_000,
                            "n_output_tokens": 892_410}}
    check("pre-0.1.10 input is repaired",
          _token_totals(old)["n_input_tokens"], 1_870_000 + 32_600_000)

    # The inclusive form is already correct and must be left exactly alone.
    new = {"agent_result": {"n_input_tokens": 90_127_657,
                            "n_cache_tokens": 87_532_092,
                            "n_output_tokens": 2_716_386}}
    check("inclusive input is untouched",
          _token_totals(new)["n_input_tokens"], 90_127_657)

    # Equal counts are legal (a fully cached prompt) and must not be doubled.
    same = {"agent_result": {"n_input_tokens": 500, "n_cache_tokens": 500,
                             "n_output_tokens": 10}}
    check("a fully cached prompt is not doubled",
          _token_totals(same)["n_input_tokens"], 500)

    # No cache field at all: nothing to compare against, so report as given.
    bare = {"agent_result": {"n_input_tokens": 1234, "n_output_tokens": 10}}
    check("input without a cache counter is unchanged",
          _token_totals(bare)["n_input_tokens"], 1234)

    # Output and cost must not be disturbed by the repair.
    check("output survives the repair",
          _token_totals(old)["n_output_tokens"], 892_410)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as scratch:
        test_output_tokens_are_reported_per_solve_and_per_trial(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_a_grafted_rerun_scores_but_does_not_stretch_the_clock(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_the_clock_starts_when_the_model_does(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_a_trial_in_flight_advances_the_clock_but_its_setup_does_not(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_appending_to_a_log_does_not_move_the_agent_start(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_endpoint_faults_leave_the_denominator(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_throughput_stops_the_clock_when_a_run_ends(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_throughput_still_runs_the_clock_on_a_live_run(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_a_submission_the_verifier_could_not_score_is_a_failure(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_a_run_of_only_build_failures_reports_no_errors(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_partial_credit_excludes_the_tasks_it_solved(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_a_run_that_solved_everything_has_no_partial_credit(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_feed_reads_far_enough_back(Path(scratch))
    test_feed_reassembles_token_streams()
    test_feed_renders_unknown_event_shapes()
    test_model_summary_reports_the_window_that_was_used()
    test_prompt_totals_are_repaired_when_input_excludes_cache()
    with tempfile.TemporaryDirectory() as scratch:
        test_a_harness_that_stops_on_its_own_output_cap_is_not_a_clean_trial(
            Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_tokens_are_recovered_from_the_agent_log_when_harbor_recorded_none(
            Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_a_local_endpoint_is_never_given_a_price(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_a_task_that_ate_the_run_and_failed_is_called_out(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        test_a_short_run_has_no_runaways(Path(scratch))
    test_pairing()
    test_wilson()
    print("\n" + ("FAILED: " + ", ".join(failures) if failures else "all checks passed"))
    raise SystemExit(1 if failures else 0)
