"""Live view of what the agent is doing right now.

A benchmark that takes days is unwatchable if the only feedback is a pass rate
that appears at the end. This tails the in-flight trial's agent log so you can
see the model reason in something close to real time.

Two things this deliberately surfaces, because both look like a hang and
neither is one:

* **Silence.** Agent logs are piped through ``stdbuf -oL tee``, which flushes
  per line. A long unbroken reasoning block produces no newline and therefore no
  output until it ends -- observed going quiet for 9+ minutes mid-generation.
* **Nothing on disk.** An agent can spend half an hour reasoning before writing
  its first file. That is normal, not a stall.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from bench import RUNS_DIR
from bench.config import strip_ansi

MANIFEST_NAME = "harness-bench.json"

#: First read. Enough for a prose log, and cheap to do every poll.
TAIL_BYTES = 24_000
#: Ceiling on how far back the feed will *scan* for one refresh.
#:
#: Bytes scanned and bytes rendered stopped being the same budget once a log
#: could be almost entirely filtered. With --debug-capture, Codex writes about
#: 97% Rust tracing, and one turn of reasoning produced a 10MB log whose last
#: 400KB held 1,115 lines and not one line of agent output. A ceiling sized for
#: a log that is mostly content cannot reach content in a log that is mostly
#: not, so the panel went silent while the run was working perfectly.
#:
#: These bytes are scanned and discarded, never rendered, so the cost is a read
#: and a regex per line rather than anything the browser sees. A normal log
#: still costs exactly one read of TAIL_BYTES, because the loop stops as soon
#: as it has enough to show.
MAX_TAIL_BYTES = 16_000_000
#: How much readable text the feed tries to hold, in characters. Roughly a
#: screen and a few scrollbacks; see `read_feed` for why bytes are the wrong
#: unit to budget in.
TARGET_TEXT_CHARS = 12_000
MAX_ENTRIES = 200

# Liveness is measured by watching the log's *size*, not its mtime. On Windows,
# a file being appended to through a Docker bind mount can report a stale mtime
# for many minutes -- observed claiming 24 minutes of silence while the file
# grew by 5KB. Size is monotonic and cannot lie in that direction.
# Maps log path -> (size_last_seen, monotonic time that size first appeared).
_GROWTH: dict[str, tuple[int, float]] = {}


def _quiet_seconds(path: Path, size: int) -> float:
    """Seconds since this log last grew, tracked across calls."""
    key = str(path)
    now = time.monotonic()
    seen = _GROWTH.get(key)
    if seen is None or size != seen[0]:
        _GROWTH[key] = (size, now)
        return 0.0
    return now - seen[1]

# Agent log filenames, in the order we prefer them. Each installed agent tees
# its stdout to exactly one of these.
#
# A harness whose log is not named here has no live feed at all: the panel stays
# blank for the whole run and reads as a hung agent rather than a missing entry.
# opencode and minion were both absent, so adding a harness means adding its
# _OUTPUT_FILENAME here too. tests/test_local_agents.py checks that every
# adapter's filename is covered, which is cheaper than remembering.
LOG_NAMES = (
    "hermes.txt", "omp.txt", "pi.txt", "opencode.txt", "minion.txt",
    "claude-code.txt", "codex.txt", "agent.txt",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        parsed = json.loads(text) if text.strip() else None
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def find_active_trial(runs_dir: Path = RUNS_DIR) -> dict[str, Any] | None:
    """The most recently touched trial that has not produced a result yet."""
    if not runs_dir.exists():
        return None

    best: dict[str, Any] | None = None
    for job_dir in runs_dir.iterdir():
        if not job_dir.is_dir():
            continue
        manifest = _read_json(job_dir / MANIFEST_NAME) or {}
        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir() or (trial_dir / "result.json").exists():
                continue
            log_path = find_log(trial_dir)
            # Ordering by the agent log's mtime, not the directory's: the
            # directory timestamp stops moving once setup finishes.
            when = log_path.stat().st_mtime if log_path else 0.0
            if best is None or when > best["_mtime"]:
                best = {
                    "_mtime": when,
                    "job_dir": job_dir,
                    "trial_dir": trial_dir,
                    "manifest": manifest,
                    "log_path": log_path,
                }
    return best


def find_log(trial_dir: Path) -> Path | None:
    agent_dir = trial_dir / "agent"
    if not agent_dir.is_dir():
        return None
    for name in LOG_NAMES:
        candidate = agent_dir / name
        if candidate.exists():
            return candidate
    logs = [p for p in agent_dir.glob("*.txt") if p.is_file()]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def agent_started_at(log_path: Path) -> float | None:
    """When the model actually started working on this trial, as a POSIX time.

    A trial is four phases -- environment build, harness install, agent, then
    the verifier -- and only the third is the model doing the task. Harbor
    records all four in `agent_execution.started_at`, but writes result.json
    once, at the end, so there is nothing to read while a trial is in flight.
    The agent log is the substitute: Harbor launches the agent with its output
    teed into that file, so the file appearing *is* the agent starting.
    Measured against a finished trial's own record, the log's birth time landed
    252 ms after `agent_execution.started_at`, against 72 seconds of setup that
    the trial directory's timestamp would have counted as model time.

    `st_ctime` is not that timestamp and is the trap here: on Windows it is the
    creation time, on Linux it is the inode *change* time, which every append to
    a growing log moves forward -- a trial two hours in would report seconds.
    So use a real birth time where the platform keeps one (Windows and macOS),
    and fall back to the containing directory's mtime, which is stamped when the
    log is added to it and left alone while the log is written to.

    Only sound for a trial that is still running, which is the only trial that
    needs it. Once the agent finishes, the harness's trajectory file lands in
    the same directory and moves that mtime to the *end* of the agent phase --
    on the trial measured above, to within a second of `agent_execution
    .finished_at`. Finished trials must be read from result.json instead.
    """
    try:
        born = getattr(log_path.stat(), "st_birthtime", None)
        if born:
            return float(born)
        return log_path.parent.stat().st_mtime
    except OSError:
        return None


def tail(path: Path, n_bytes: int = TAIL_BYTES) -> str:
    """Last n_bytes of a file, decoded leniently and cut at a line boundary."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > n_bytes:
                handle.seek(size - n_bytes)
            raw = handle.read()
    except OSError:
        return ""
    text = strip_ansi(raw.decode("utf-8", errors="replace"))
    # A mid-character or mid-line start reads as garbage; drop the partial head.
    if size > n_bytes and "\n" in text:
        text = text.split("\n", 1)[1]
    return text


def _content_text(content: Any) -> str:
    """Flatten a message content field, which may be a string or block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "thinking" and isinstance(
                    block.get("thinking"), str
                ):
                    parts.append(block["thinking"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _entry_from_event(event: dict[str, Any]) -> dict[str, str] | None:
    """Render one structured agent event (omp/pi NDJSON) as a feed entry."""
    kind = event.get("type")
    if kind in ("message_start", "message_update"):
        return None  # deltas; the authoritative message follows in message_end

    if kind == "message_end":
        message = event.get("message") or {}
        role = message.get("role")
        text = _content_text(message.get("content"))
        calls = message.get("toolCalls") or message.get("tool_calls") or []
        if calls:
            names = []
            for call in calls:
                if isinstance(call, dict):
                    fn = call.get("function") or {}
                    names.append(str(call.get("name") or fn.get("name") or "tool"))
            label = ", ".join(names) or "tool"
            return {"kind": "tool", "text": f"→ {label}\n{text}".strip()}
        if role == "assistant" and text:
            return {"kind": "assistant", "text": text}
        if role in ("toolResult", "tool") and text:
            return {"kind": "result", "text": text}
        return None

    if kind in ("turn_end", "agent_end", "session_start"):
        return {"kind": "meta", "text": str(kind).replace("_", " ")}
    return _entry_from_unknown(event)


#: Keys that carry human-readable text across the event shapes harnesses use.
#: Ordered by how likely each is to be the thing worth reading.
_TEXT_KEYS = (
    "error", "text", "content", "delta", "thinking", "reasoning", "chunk", "output",
)

#: Bookkeeping events whose payload is an id, a token count or a status word.
#: They carry a "content" or "text" field often enough to be mistaken for
#: output, and showing a redacted session id as if the agent said it is worse
#: than showing nothing.
_STRUCTURAL = {
    "metadata", "done", "session_capture", "ping", "heartbeat",
    "usage", "turn_usage", "token_usage", "init", "ready",
    # Step boundaries. opencode brackets every turn with these and hangs its
    # token accounting off step_finish; neither carries anything to read.
    "step_start", "step_finish", "step_end",
}

#: Deltas we drop because the authoritative message follows them.
_SILENT = {"message_start", "message_update"}

#: Bookkeeping identified by `subtype` rather than by `type`. Claude Code types
#: every one of these as a plain "system", so the type alone cannot separate a
#: running token counter from something worth reading.
#:
#: `thinking_tokens` is a progress counter emitted once per couple of tokens of
#: reasoning. Measured on one trial: 32,505 of 32,666 lines and 91% of a 6.6MB
#: log. It matters twice over, because the two failures compound:
#:
#:   * Each line is recognizable but carries no prose, so it fell through to the
#:     unreadable-event path and was dumped into the feed as raw JSON.
#:   * Those dumps then *counted* toward read_feed's text budget -- 123 lines of
#:     envelope overshoot it on their own -- so the window never grew, and the
#:     real output a few hundred KB back was never reached. The panel showed a
#:     wall of JSON, and anything readable scrolled out of a 24KB window within
#:     seconds of appearing.
#:
#: Suppressing them fixes both: nothing renders, the budget stays unmet, and
#: read_feed reads back until it finds the actual transcript.
_STRUCTURAL_SUBTYPES = {"thinking_tokens"}


def _subtype_of(event: Any) -> str:
    if not isinstance(event, dict):
        return ""
    return str(event.get("subtype", "")).replace("-", "_").lower()

#: Wrappers harnesses put the interesting payload inside. opencode nests
#: everything under "part"; others use "data" or "payload". Without this the
#: text-key scan only ever sees the envelope's own bookkeeping fields.
_ENVELOPE_KEYS = ("part", "data", "payload", "event", "message", "delta")

#: Tool activity, named differently by every harness. Worth rendering even from
#: an unfamiliar one: "which tools did it reach for" is most of what a feed is
#: for, and the payload is a structure rather than prose, so it needs its own
#: shaping instead of falling through to the text-key scan.
_TOOL_CALL = {"tool_use", "tool_call", "toolcall", "tool_start", "tool"}
_TOOL_RESULT = {"tool_result", "toolresult", "tool_output", "tool_end"}

#: A tool can return a whole file. The feed is a glance, not a transcript.
_TOOL_BODY_CHARS = 1200


def _kind_of(event: Any) -> str:
    """An event's type, normalized across the -/_ spellings harnesses use."""
    if not isinstance(event, dict):
        return ""
    return str(event.get("type", "")).replace("-", "_").lower()


def _plain_text(value: Any) -> str:
    """Text from a value that is already prose, never a structure.

    A dict returns nothing on purpose: it is an envelope to look inside, and
    rendering it as JSON would put braces in the feed where prose belongs.
    """
    if isinstance(value, str):
        return value
    return _content_text(value) if isinstance(value, list) else ""


def _stringify(value: Any) -> str:
    """Render a tool payload, which is a structure rather than prose."""
    if isinstance(value, str):
        return value
    text = _content_text(value)
    if text:
        return text
    if value in (None, {}, []):
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _layers(event: dict[str, Any]) -> list[dict[str, Any]]:
    """The event and the wrappers nested in it, innermost first.

    Harnesses bury the detail that matters: opencode names the tool on `part`
    and puts its input, output and status one level deeper on `part.state`, so
    reading the outer event alone yields a call with no name and no arguments.
    """
    layers = [event]
    node = event
    for _ in range(3):
        nested = next(
            (
                node[key]
                for key in (*_ENVELOPE_KEYS, "state")
                if isinstance(node.get(key), dict)
            ),
            None,
        )
        if nested is None:
            break
        layers.append(nested)
        node = nested
    return list(reversed(layers))


def _pick(layers: list[dict[str, Any]], *keys: str) -> Any:
    """First of `keys` present anywhere in `layers`, innermost layer winning."""
    for layer in layers:
        for key in keys:
            if key in layer:
                return layer[key]
    return None


def _clip(text: str, limit: int = _TOOL_BODY_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def _tool_entry(event: dict[str, Any], kind: str) -> dict[str, str] | None:
    """Render a tool call or its result from an unfamiliar harness."""
    layers = _layers(event)
    name = str(_pick(layers, "tool", "name") or "tool")
    status = _pick(layers, "status")
    status = status if isinstance(status, str) and status.strip() else ""
    output = _stringify(_pick(layers, "output", "result"))

    if kind in _TOOL_RESULT:
        head = f"{name} ({status})" if status else name
        return {"kind": "result", "text": f"{head}\n{_clip(output)}".strip()}

    head = f"→ {name}"
    title = _pick(layers, "title")
    if isinstance(title, str) and title.strip() and title.strip() != name:
        head += f"  {title.strip()}"

    body = _clip(_stringify(_pick(layers, "input", "arguments", "args")))
    # opencode reports a call only once it has finished, so the same event
    # carries the result. Dropping it would show what the agent tried and
    # never whether it worked.
    if output.strip():
        body = f"{body}\n\n{status or 'done'} → {_clip(output, _TOOL_BODY_CHARS // 3)}"
    return {"kind": "tool", "text": f"{head}\n{body}".strip()}


def _entry_from_unknown(
    event: dict[str, Any], depth: int = 0
) -> dict[str, str] | None:
    """Best-effort rendering of an event shape this module does not know.

    Without this, a harness whose event names differ from the ones handled
    above has every line dropped: the log grows, nothing renders, and the panel
    sits on "waiting for the first output" while the agent is working. One
    harness shipped here did exactly that: every one of its events was typed,
    and not one of the types was a name this module knew.

    Structural events (metadata, done, token counts) carry no prose and are
    still skipped; the point is to surface text, not to dump the stream.

    Text is returned *unstripped*: in a token-level stream the leading space of
    " user" is the word break, and stripping it here would weld the words
    together when `_join_deltas` reassembles them.
    """
    kind = _kind_of(event)
    if kind in _STRUCTURAL or _subtype_of(event) in _STRUCTURAL_SUBTYPES:
        return None
    if kind in _TOOL_CALL or kind in _TOOL_RESULT:
        return _tool_entry(event, kind)
    for key in _TEXT_KEYS:
        text = _plain_text(event.get(key)) if key in event else ""
        if text.strip():
            if key == "error":
                return {"kind": "meta", "text": text.strip()}
            # Possibly one token of a stream; _join_deltas decides.
            return {"kind": "assistant", "text": text, "_delta": "1"}

    # Nothing at the top level: the payload is probably wrapped. Recurse once,
    # letting the inner type speak for itself when it has one.
    if depth == 0:
        for key in _ENVELOPE_KEYS:
            inner = event.get(key)
            if not isinstance(inner, dict):
                continue
            if "type" not in inner:
                inner = {**inner, "type": event.get("type")}
            found = _entry_from_unknown(inner, depth=1)
            if found:
                return found
    return None


def _suppressed(event: dict[str, Any]) -> bool:
    """True when the event was recognized and deliberately not rendered.

    This is the difference between "we hid this" and "we could not read this",
    and only the second justifies dumping raw JSON into the feed. opencode
    opens a session with a lone ``step_start``, so without this distinction the
    panel greets every run with a wall of braces.
    """
    if _kind_of(event) in _STRUCTURAL or _kind_of(event) in _SILENT:
        return True
    if _subtype_of(event) in _STRUCTURAL_SUBTYPES:
        return True
    return any(
        _kind_of(event.get(key)) in _STRUCTURAL for key in _ENVELOPE_KEYS
    )


#: A token-level stream emits one event per token, so rendering one entry per
#: event produces a column of single words -- measured at 6,623 `content`
#: events in two minutes on one harness. Nothing in an event says
#: whether a harness streams deltas or whole messages, so it is read off the
#: population: many pieces, most of them shorter than a word or two, is a
#: delta stream, and deltas concatenate verbatim into the original prose.
_DELTA_MIN_EVENTS = 8
_DELTA_MAX_MEDIAN = 16


def _is_delta_stream(pieces: list[str]) -> bool:
    if len(pieces) < _DELTA_MIN_EVENTS:
        return False
    lengths = sorted(len(piece) for piece in pieces)
    return lengths[len(lengths) // 2] <= _DELTA_MAX_MEDIAN


def _join_deltas(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Reassemble a token-level stream into paragraphs, or leave it alone.

    The decision is made once for the whole tail rather than per event, so a
    log cannot come out half-concatenated. When the events are not deltas, the
    only effect is the strip that `_entry_from_unknown` deliberately skipped.
    """
    joining = _is_delta_stream([e["text"] for e in entries if e.get("_delta")])

    merged: list[dict[str, str]] = []
    for entry in entries:
        if not entry.pop("_delta", None):
            merged.append(entry)
            continue
        if not joining:
            entry["text"] = entry["text"].strip()
            merged.append(entry)
            continue
        previous = merged[-1] if merged else None
        if previous is not None and previous.get("_streamed"):
            previous["text"] += entry["text"]
        else:
            merged.append({**entry, "_streamed": "1"})

    for entry in merged:
        entry.pop("_streamed", None)
        entry["text"] = entry["text"].strip()
    return [entry for entry in merged if entry["text"]]


#: A Rust `tracing` line: an ISO-8601 timestamp, a level, then a target.
#:
#: `--debug-capture` sets RUST_LOG so a transport failure can be attributed
#: afterwards instead of guessed at, and the harnesses that ignore it cost
#: nothing. Codex does not ignore it: it is Rust, and at INFO it logs one line
#: per SSE event -- so per token of reasoning. Measured on one trial: 65,965
#: lines, 24.2MB of a 24.9MB log, 97% of the bytes. None of it is JSON, so it
#: landed in the prose path and the panel rendered two 23KB blobs of
#: `codex_otel.log_only: event.name="codex.sse_event"` where the agent's own
#: output should have been.
#:
#: WARN and ERROR are deliberately kept. They are the reason the flag exists --
#: a refused connection or a reset stream says so at those levels -- and there
#: are few enough of them to read. Only the per-token bookkeeping is dropped.
_RUST_LOG_NOISE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z?\s+(TRACE|DEBUG|INFO)\b"
)

#: The tail of a tracing record whose fields ran onto their own line.
#:
#: A `tracing` record is one line only until a *field value* contains a newline.
#: Codex's telemetry preview does, so the record's remaining fields
#: (`mcp_server= ... event.timestamp= ... conversation.id= ... auth_mode=`) are
#: printed with no timestamp in front of them, and the rule above -- which
#: anchors on the timestamp -- never sees them. Measured on one trial: 20 such
#: lines out of the 338 that survived, and they were the whole of what still
#: looked like machine exhaust in the panel.
#:
#: Matched on two fields together rather than on any single one. `event.` and
#: `conversation.` appear in agent prose often enough on their own; a line
#: carrying both assignments is a field list, not a sentence. Deliberately
#: narrow, because the last over-broad guess here hid real output.
_RUST_LOG_TRAILER = re.compile(
    r"event\.timestamp=.*conversation\.id=|conversation\.id=.*event\.timestamp="
)


def parse_entries(text: str) -> list[dict[str, str]]:
    """Turn a raw log tail into feed entries, whatever format it is in.

    omp emits NDJSON; hermes emits prose. Rather than branch on the harness,
    try to parse each line and fall back to text, so a new harness's log is
    readable on day one without an adapter here.
    """
    entries: list[dict[str, str]] = []
    plain: list[str] = []
    unreadable: list[str] = []

    def flush_plain() -> None:
        if not plain:
            return
        blob = "\n".join(plain).strip()
        plain.clear()
        if blob:
            entries.append({"kind": "assistant", "text": blob})

    for line in text.splitlines():
        stripped = line.strip()
        # Dropped from the *view* only. The log on disk keeps every line, which
        # is the point of --debug-capture: the record is for attributing a
        # transport failure afterwards, not for reading live at one line per
        # token.
        if _RUST_LOG_NOISE.match(stripped) or _RUST_LOG_TRAILER.search(stripped):
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                plain.append(line)
                continue
            if isinstance(event, dict) and "type" in event:
                flush_plain()
                entry = _entry_from_event(event)
                if entry:
                    entries.append(entry)
                elif not _suppressed(event):
                    unreadable.append(line)
                continue
        plain.append(line)
    flush_plain()

    # A token-level stream arrives as thousands of one-word entries; put the
    # words back together before anything downstream tries to lay them out.
    entries = _join_deltas(entries)

    # Prose logs come back as one huge blob; split on blank lines so the feed
    # scrolls in readable chunks instead of a single wall of text.
    expanded: list[dict[str, str]] = []
    for entry in entries:
        if entry["kind"] == "assistant" and "\n\n" in entry["text"]:
            for chunk in re.split(r"\n\s*\n", entry["text"]):
                chunk = chunk.strip()
                if chunk:
                    expanded.append({"kind": "assistant", "text": chunk})
        else:
            expanded.append(entry)

    # Last resort: lines arrived that we could not read at all. Show them raw
    # rather than an empty panel -- "waiting for the first output" while the log
    # grows is a lie, and it points the reader at the model when the problem is
    # here. Events we recognized and chose to hide are not in `unreadable`, so
    # a run that has only opened its session still shows the waiting message.
    if not expanded and unreadable:
        expanded = [{"kind": "meta", "text": line} for line in unreadable[-MAX_ENTRIES:]]

    return expanded[-MAX_ENTRIES:]


def read_feed(path: Path) -> list[dict[str, str]]:
    """Feed entries, reading back far enough to actually fill the panel.

    Budgeting the read in *bytes* silently assumes a log is mostly text. A
    token-level stream is not: measured on one, about 90 bytes of JSON envelope
    per token, so only 5% of what is read is prose, and a 24KB window holds
    roughly five seconds of output. With a five-second refresh that means every
    poll replaces the whole panel and there is nothing to scroll back through.

    So the budget is in characters of rendered text, and the window grows until
    it meets that budget, runs out of file, or hits the ceiling. A prose log
    still costs exactly one read of TAIL_BYTES.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return []

    window = TAIL_BYTES
    while True:
        entries = parse_entries(tail(path, window))
        rendered = sum(len(entry["text"]) for entry in entries)
        if (
            rendered >= TARGET_TEXT_CHARS
            or window >= size
            or window >= MAX_TAIL_BYTES
        ):
            if not entries and window < size:
                # Scanned the ceiling and found nothing renderable, in a log
                # that is still bigger than what was read. Silence here reads
                # as a hung agent, and it is the one thing this module exists
                # to not do -- so say what is actually happening. The usual
                # cause is diagnostics: RUST_LOG tracing is filtered from the
                # view, and at ~97% of the bytes it can bury the output past
                # any ceiling.
                return [{
                    "kind": "meta",
                    "text": (
                        f"No agent output in the last {window // 1_000_000}MB of a "
                        f"{size // 1_000_000}MB log. The rest is diagnostic tracing, "
                        f"which is kept on disk but not shown here. The agent is "
                        f"generating; output appears when it finishes a step."
                    ),
                }]
            return entries
        # Scale by what this read actually yielded rather than doubling blindly,
        # so a very verbose envelope is caught in one more read, not five.
        ratio = TARGET_TEXT_CHARS / max(rendered, 1)
        window = min(int(window * min(max(ratio, 2.0), 8.0)), MAX_TAIL_BYTES)


def read_activity(runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    """Everything the live feed panel needs, or {"active": False}."""
    found = find_active_trial(runs_dir)
    if not found:
        return {"active": False}

    manifest = found["manifest"]
    trial_dir: Path = found["trial_dir"]
    log_path: Path | None = found["log_path"]

    # Trial dirs are named "<task>__<suffix>"; the task is the readable half.
    task_name = trial_dir.name.rsplit("__", 1)[0]

    payload: dict[str, Any] = {
        "active": True,
        "run_id": found["job_dir"].name,
        "task": task_name,
        "harness": manifest.get("harness"),
        "harness_label": manifest.get("harness_label") or manifest.get("harness"),
        "model_label": (manifest.get("model") or {}).get("label"),
        "phase": "agent" if log_path else "setting up",
        "entries": [],
        "log_bytes": 0,
        "silent_s": None,
    }

    # Two clocks, because they answer different questions and merging them
    # answered neither. The trial directory is created when the trial starts, so
    # its creation time is when the *container* work began; the agent log
    # appearing is when the model began. Reporting the first as "elapsed" put
    # image pulls and harness installs on the model's account -- on one measured
    # trial, 72 seconds of them -- and there is no ceiling on how wrong that
    # gets: a cold image pull is minutes, and a trial can show more elapsed than
    # the agent timeout allows, which is what gave this away.
    now = time.time()
    try:
        trial_started = trial_dir.stat().st_ctime
    except OSError:
        trial_started = None

    agent_started = agent_started_at(log_path) if log_path else None
    # None while setting up, deliberately: there is no agent clock yet, and
    # showing a zero would claim the model has been working for no time rather
    # than that it has not started.
    payload["elapsed_s"] = max(0.0, now - agent_started) if agent_started else None
    # Runs to now while setting up, then freezes at what setup cost. Kept
    # visible either way -- it is the number that explains a trial sitting at
    # "setting up" for ten minutes, and the one that says why a run's wall clock
    # exceeds its model time.
    if trial_started:
        payload["setup_s"] = max(0.0, (agent_started or now) - trial_started)
    else:
        payload["setup_s"] = None

    if log_path and log_path.exists():
        size = log_path.stat().st_size
        payload["log_name"] = log_path.name
        payload["log_bytes"] = size
        payload["silent_s"] = _quiet_seconds(log_path, size)
        payload["entries"] = read_feed(log_path)

    return payload
