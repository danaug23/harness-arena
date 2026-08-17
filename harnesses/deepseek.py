"""Harbor agent adapter for DeepSeek Harness (``dsh``), pointed at an arbitrary endpoint.

dsh is a plugin composition rather than a CLI with flags: ``dsh --profile
headless "task"`` boots an ordered stack of patch layers (the ``dsh-base``
bundle, then ``dsh-headless``, then the profile's own file, then every
``--patch`` overlay in argv order) and runs one task to quiescence. So this
adapter configures almost nothing through arguments and almost everything
through one generated overlay file, which is the last layer and therefore wins.

Six things decide how it is written. All six were measured against the
published package and the upstream source rather than inferred, because every
one of them fails silently or fatally:

* **The task needs TWO ``--``.** The launcher and the one-shot app each parse
  with commander, and the first ``--`` is consumed by the launcher's parser --
  so the app still sees a task beginning with ``-`` as an unknown option and
  exits 1 on the usage message. Exactly one Terminal-Bench 2 instruction opens
  with a bullet (pytorch-model-recovery), which is the worst frequency for a
  bug like this. Verified by running both parser configurations against
  commander 15: with one ``--`` the app reports ``unknown option '- You are
  given a PyTorch state dictionary...'``; with two it receives the task intact.
  opencode and omp needed one; dsh needs two, and that is the difference.

* **base_url keeps its ``/v1``.** The DeepSeek adapter posts to
  ``<baseURL>/chat/completions`` (``packages/llm/llm-deepseek/src/adapter.ts``),
  so the endpoint's configured URL is used as-is -- the same spelling Codex
  wants and the opposite of Claude Code's. Its own default is
  ``https://api.deepseek.com``, i.e. no version segment, which is why leaving
  it unset would not merely be wrong but would send the task to DeepSeek.

* **``DSH_PERMISSION_MODE=danger-full-access`` is what makes it unattended.**
  The base bundle reads that variable for both its sandbox mode and its
  approval policy, defaulting to ``workspace-write`` + ``ask``: nobody is there
  to answer, and the trial burns its budget waiting. The same setting is what
  keeps the run alive at all -- ``dsh-sandbox-local`` fails *closed* with
  ``SANDBOX_UNAVAILABLE`` when neither bubblewrap nor a Landlock-enforcing
  kernel is usable, which is the normal state of a task container, and
  ``danger-full-access`` is documented as the one mode that "deliberately
  bypasses ``ctx.sandbox``". The container is the sandbox here, as it is for
  every other harness in this catalog.

* **The session log is the run.** ``dsh --profile headless`` prints only the
  final assistant message, at the end, and keeps stderr empty on success --
  a live feed tailing its stdout would stay blank for the whole trial and read
  as a hung agent. The event stream lives in the JSONL session log instead, so
  the overlay puts that log under ``/logs/agent`` in its uncompressed,
  one-event-per-line form (it is ``.jsonl.zstd`` with packed chunk runs by
  default) and the run mirrors it into the tailable file as it is written.

* **Token accounting comes from that log, not from the stream.** Usage rides
  on per-step ``assistant/chunk {type: 'usage'}`` records, with
  ``assistant/message.usage`` as the committed-step fallback. Counts are
  *disjoint* upstream: ``inputTokens`` excludes cache, so billed input is the
  sum of the three input fields, and ``reasoningTokens`` is already inside
  ``outputTokens`` and must not be added again.

* **It needs a C++ toolchain, unlike every other harness here.** dsh is
  npm-only and its dependency tree reaches ``node-pty``, a native addon with no
  prebuilt binary, so the install compiles it. omp and opencode both install a
  standalone release binary and needed nothing but curl; copying their install
  shape cost a whole run, which failed in setup on every trial with
  ``gyp ERR! not ok``. Measured against alexgshaw/gpt2-codegolf:20251031: with
  python3, make and g++ added the whole chain takes 49s, against Harbor's
  360s agent-setup budget.

* **The shipped extras that call a model are switched off.** Session titling
  is a second model call per session that is not part of the task, and
  ``web_search`` is a full auxiliary request to DeepSeek's own API carrying
  task text off the machine. Neither is a property of the coding harness being
  measured. Compaction is left on: that *is* the harness.
"""

from __future__ import annotations

import json
from typing import Any, override

import yaml
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

#: The npm package. `dsh` is its only binary.
_PACKAGE = "@deepseek-ai/dsh"

#: nvm installs node into a directory it puts on PATH by sourcing this script,
#: and `npm install -g` puts `dsh` beside that node. Sourced in the *current*
#: shell rather than a subshell, or the PATH it exports never reaches the run.
#: Guarded so an image that already carries a new enough node still works, and
#: closed with `true` so `set -e` does not see the test's exit status.
_LOAD_NODE = (
    'export NVM_DIR="$HOME/.nvm"; '
    '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" >/dev/null 2>&1; '
    "true"
)

#: Where the generated overlay lives. Outside $DSH_HOME on purpose: it is an
#: invocation argument, not part of the profile.
_CONFIG_DIR = "/tmp/dsh"
_PATCH = f"{_CONFIG_DIR}/arena.cordis.yml"

#: Profile state (settings, credentials, the anonymous id, the profile itself).
#: Pinned rather than left to $HOME so nothing baked into a task image can seed
#: it, and so a stray `~/.dsh` cannot make two trials differ.
_DSH_HOME = "/tmp/dsh-home"

#: The one-shot profile. It auto-initializes from a shipped template.
_PROFILE = "headless"

#: The provider route the native DeepSeek adapter owns. Not configurable: the
#: plugin registers this exact string and refuses a second adapter on it.
_PROVIDER = "deepseek-official"

#: Where the JSONL session logs are written, under the directory Harbor copies
#: back out of the container.
_SESSION_ROOT = "/logs/agent/dsh/sessions"

#: ``turn/end`` reasons that are the agent's own outcome rather than the
#: harness failing, so the workspace is worth grading. Anything else -- notably
#: ``error`` (a structured LLM or transport failure) and ``aborted`` (a
#: cancellation) -- keeps its non-zero exit, because the rig classifies a
#: transport fault from the exception and excludes it from the denominator.
#: An outage swallowed here would be scored as a fair attempt instead.
_AGENT_STOP_REASONS = ("max-tokens", "blocked")

#: This rig's reasoning vocabulary is Codex's (none/minimal/low/medium/high);
#: dsh publishes exactly three efforts and fails plugin load on anything else,
#: before any network I/O. Mapping is therefore not a preference but the
#: difference between a run and a run in which every trial dies identically.
#: "off" is a real setting rather than an omission: it serializes
#: `thinking: {type: disabled}` and sends no `reasoning_effort` at all.
_EFFORTS = {
    "none": "off",
    "off": "off",
    "minimal": "high",
    "low": "high",
    "medium": "high",
    "high": "high",
    "max": "max",
}


class DeepSeekHarness(BaseInstalledAgent):
    """DeepSeek Harness against an arbitrary OpenAI-compatible endpoint."""

    _OUTPUT_FILENAME = "dsh.txt"

    def __init__(
        self,
        *args: Any,
        base_url: str | None = None,
        api_key: str | None = None,
        # No default: the served model's real window is probed once by the
        # runner and substituted in. A constant here would be a second,
        # silently-wrong answer the moment a different model is loaded -- and
        # dsh's own fallback is 1,000,000 tokens, which would size compaction
        # for a model nobody is running.
        context_window: int | str | None = None,
        max_tokens: int | str | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._base_url = base_url or self._get_env("DSH_BASE_URL")
        self._api_key = api_key
        self._context_window = int(context_window or 0) or None
        self._max_tokens = int(max_tokens or 0) or None
        self._reasoning_effort = (reasoning_effort or "").strip().lower() or None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @staticmethod
    @override
    def name() -> str:
        return "dsh"

    @override
    def get_version_command(self) -> str | None:
        return f"{_LOAD_NODE}; dsh --version"

    @override
    def parse_version(self, stdout: str) -> str:
        lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
        return lines[-1] if lines else stdout.strip()

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    @staticmethod
    def _packages_command() -> str:
        """Install the system packages ``npm install -g dsh`` cannot do without.

        Task images are not uniform -- most are Debian-derived but some are
        Alpine or RPM-based -- so a hard-coded apt-get would turn those into an
        install failure that reads like an agent bug.

        Unlike the omp and opencode adapters this one is *not* best-effort,
        because dsh is npm-only. Both of those install a standalone release
        binary with no runtime prerequisites, which is exactly why curl alone
        was enough for them. dsh's dependency tree reaches ``node-pty``, a
        native addon that ships no prebuilt binary, so ``npm install -g`` runs
        ``node-gyp rebuild`` and needs python3, make and a C++ compiler.
        Terminal-Bench 2's images carry gcc and none of the other three, so
        every trial failed in setup with ``gyp ERR! not ok`` before the agent
        started -- measured, on alexgshaw/gpt2-codegolf:20251031.

        Checked one tool at a time at the end, so a toolchain that could not be
        installed says which piece is missing here rather than forty seconds
        later inside npm's gyp output.
        """
        return (
            "set -u; "
            "pkgs='curl ca-certificates git ripgrep python3 make g++'; "
            "if command -v apt-get >/dev/null 2>&1; then "
            "  apt-get update >/dev/null 2>&1; "
            "  apt-get install -y $pkgs >/dev/null 2>&1 || apt-get install -y curl ca-certificates python3 make g++; "
            "elif command -v apk >/dev/null 2>&1; then "
            "  apk add --no-cache $pkgs >/dev/null 2>&1 || apk add --no-cache curl ca-certificates python3 make g++; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "  dnf install -y curl ca-certificates git ripgrep python3 make gcc-c++ >/dev/null 2>&1 || dnf install -y curl python3 make gcc-c++; "
            "elif command -v yum >/dev/null 2>&1; then "
            "  yum install -y curl ca-certificates git python3 make gcc-c++ >/dev/null 2>&1 || yum install -y curl python3 make gcc-c++; "
            "fi; "
            "for tool in curl python3 make g++; do "
            "  command -v $tool >/dev/null 2>&1 || { "
            "    echo \"dsh install needs $tool: @deepseek-ai/dsh builds node-pty from source with node-gyp, and this image has no way to install one.\" >&2; "
            "    exit 1; "
            "  }; "
            "done"
        )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=self._packages_command(),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        # dsh declares engines "^22.19.0 || >=24.0.0". nvm's node 22 line is
        # Harbor's default and is what the other npm-installed harnesses in
        # this catalog get, so the runtime under the agent stays comparable.
        spec = f"{_PACKAGE}@{self._version}" if self._version else _PACKAGE
        await self.exec_as_agent(
            environment,
            command=(
                "set -eu; "
                f"{nvm_node_install_snippet()} && "
                f"npm install -g {spec} && "
                "dsh --version"
            ),
        )

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def _model_id(self) -> str:
        """The bare model id, without the ``local/`` provider prefix."""
        if not self.model_name:
            raise ValueError(
                "dsh requires --model. Use '<provider>/<model-id>', e.g. "
                "'local/my-model'."
            )
        return (
            self.model_name.split("/", 1)[1]
            if "/" in self.model_name
            else self.model_name
        )

    def _effort(self) -> str | None:
        """This rig's reasoning effort, in dsh's own three-value vocabulary."""
        if not self._reasoning_effort:
            return None
        # An unmapped value is passed through rather than vetoed, which is this
        # rig's rule for a knob a new release may widen. dsh rejects it at
        # plugin load with UNSUPPORTED_REASONING_EFFORT, before any request --
        # loud and immediate, not a run of silently identical failures.
        return _EFFORTS.get(self._reasoning_effort, self._reasoning_effort)

    def _build_patch_yaml(self) -> str:
        """The ``--patch`` overlay: the last layer, so every row here wins.

        A patch replaces the targeted row's whole ``config`` rather than merging
        into it, so each row below states everything it needs.
        """
        model = self._model_id()
        effort = self._effort()

        entry: dict[str, Any] = {"id": model, "name": model}
        if self._context_window:
            entry["contextWindow"] = self._context_window
        if self._max_tokens:
            entry["maxTokens"] = self._max_tokens

        adapter: dict[str, Any] = {
            # Used as-is: the adapter appends "/chat/completions".
            "baseURL": self._base_url,
            # Configuration carries a reference, never a literal key: the
            # adapter resolves it per request through the credential seam,
            # which reads the process environment first.
            "apiKeyEnv": "DEEPSEEK_API_KEY",
            "models": [entry],
        }
        if effort == "off":
            # A deployment lock: publishes only "off", and pairing it with a
            # high/max effort fails plugin loading outright.
            adapter["thinking"] = "disabled"
        elif effort:
            adapter["thinking"] = "enabled"
            adapter["reasoningEffort"] = effort
        if self._context_window:
            # The per-model entry above answers for this model; this answers
            # for anything else the endpoint is asked for, in place of dsh's
            # 1,000,000-token default.
            adapter["defaultContextWindow"] = self._context_window
        if self._max_tokens:
            adapter["maxTokens"] = self._max_tokens

        patch: list[dict[str, Any]] = [
            {"id": "llm-deepseek", "config": adapter},
            # What a fresh Agent resolves to when nothing selected a model.
            # The one-shot runner reads exactly this service.
            {
                "id": "agent-default-model",
                "config": {"provider": _PROVIDER, "model": model},
            },
            # The run's event stream, in the one place Harbor copies out.
            # Uncompressed and unpacked: the default is zstd frames with
            # multi-event rows, which neither the live feed nor the token
            # parser can read.
            {
                "id": "session-persistence-jsonl",
                "config": {
                    "root": _SESSION_ROOT,
                    "compression": "none",
                    "packChunks": False,
                },
            },
            # Titling a one-shot session costs a model call that has nothing to
            # do with the task. The non-LLM fallback title still applies.
            {"id": "session-title-llm", "disabled": True},
            # Off by default already; stated so a changed default cannot start
            # uploading session logs from a benchmark run.
            {"id": "session-telemetry-otel", "disabled": True},
            # web_search is a full auxiliary request to DeepSeek's own API,
            # authenticated with the same key and carrying task text off the
            # machine. No other harness in this catalog is given a search tool,
            # and a task container is not on a network that would serve it.
            {"id": "web", "disabled": True},
            {"id": "web-search-deepseek", "disabled": True},
            {"id": "tool-web", "disabled": True},
        ]
        return yaml.dump(patch, sort_keys=False)

    def _agent_env(self, instruction: str = "") -> dict[str, str]:
        env = {
            "DSH_HOME": _DSH_HOME,
            # Both halves of unattended: no approval prompt, and no sandbox
            # runner to be missing. See the module docstring.
            "DSH_PERMISSION_MODE": "danger-full-access",
            # Any non-empty value opts the process out.
            "DSH_TELEMETRY_DISABLED": "1",
            "HARBOR_INSTRUCTION": instruction,
        }
        # A local server ignores the key, but the adapter refuses a request
        # with no credential anywhere (MISSING_CREDENTIAL) before it is built.
        env["DEEPSEEK_API_KEY"] = self._api_key or "local"
        return env

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self._base_url:
            raise ValueError(
                "dsh needs an endpoint. Pass --ak base_url=http://host:port/v1 "
                "(keep the /v1: dsh requests <base_url>/chat/completions)."
            )

        env = self._agent_env(instruction)

        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {_CONFIG_DIR} {_DSH_HOME} {_SESSION_ROOT} && "
                f"cat > {_PATCH} << 'PATCHEOF'\n{self._build_patch_yaml()}PATCHEOF"
            ),
            env=env,
            timeout_sec=30,
        )

        await self.exec_as_agent(
            environment, command=self._run_command(), env=env
        )

    def _run_command(self) -> str:
        """The whole invocation, in one place so it can be checked.

        Two details are load-bearing and a container is an expensive place to
        find that out, so this is kept apart from the exec that runs it: the
        doubled ``--`` before the instruction, and the background mirror that
        gives the live feed something to tail.
        """
        # The session log's path is not predictable -- it carries a normalized
        # copy of the cwd and a random session id -- so it is discovered once
        # it appears rather than guessed at, and followed from there. `head -1`
        # rather than `find -quit`, which BusyBox's find does not have.
        mirror = (
            "( log=''; "
            "  while [ -z \"$log\" ]; do "
            f"    log=$(find {_SESSION_ROOT} -name session.jsonl 2>/dev/null | head -n 1); "
            "    [ -n \"$log\" ] && break; "
            "    sleep 1; "
            "  done; "
            f'  exec tail -n +1 -F "$log" >> /logs/agent/{self._OUTPUT_FILENAME} 2>/dev/null'
            ") & MIRROR=$!"
        )

        # The instruction rides in an env var: task instructions routinely
        # contain quotes, backticks and newlines that no amount of shell
        # quoting survives cleanly through the exec -> sh -c chain.
        #
        # `-- --` is not a typo. The launcher's parser eats the first one and
        # hands the second to the app, which needs it to read an instruction
        # beginning with `-` as its task rather than as an unknown option.
        run = (
            f"dsh --profile {_PROFILE} --patch {_PATCH} -- -- \"$HARBOR_INSTRUCTION\" "
            f">> /logs/agent/{self._OUTPUT_FILENAME} 2>&1"
        )

        # dsh exits 1 whenever the final `turn/end` reason is anything but
        # `completed`, which folds "the model stopped early" into the same
        # signal as "the harness broke". Harbor reads a non-zero agent command
        # as a trial error and never runs the verifier, so a run that did the
        # work and had its last message clipped at the output cap is thrown
        # away ungraded rather than scored. Measured: llm-inference-batching-
        # scheduler ended `{"kind":"max-tokens"}` after 13 tool calls, and the
        # trial errored with the workspace never looked at.
        #
        # So the agent's own outcomes are translated back to 0 and everything
        # else still propagates. `error` and `aborted` deliberately stay
        # non-zero: an `error` turn is usually the endpoint failing, and
        # bench/collect.py needs the exception to reach it to classify the
        # trial as a transport fault and keep it out of the denominator.
        # Swallowing that would score an outage as a fair attempt.
        reasons = "|".join(f'*\'"kind":"{r}"\'*' for r in _AGENT_STOP_REASONS)
        salvage = (
            "if [ $status -ne 0 ]; then "
            f"  log=$(find {_SESSION_ROOT} -name session.jsonl 2>/dev/null | head -n 1); "
            "  last=$(grep '\"type\":\"turn/end\"' \"$log\" 2>/dev/null | tail -n 1); "
            "  case \"$last\" in "
            f"    {reasons}) status=0 ;; "
            "  esac; "
            "fi"
        )

        return (
            f"{_LOAD_NODE}; "
            f"{mirror}; "
            f"{run}; "
            "status=$?; "
            # The backend coalesces writes on a ~200ms window; give the mirror
            # a moment to catch the final flush before it is killed.
            "sleep 2; kill $MIRROR 2>/dev/null || true; "
            f"{salvage}; "
            "exit $status"
        )

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------

    @staticmethod
    def _usage_of(record: dict[str, Any]) -> dict[str, Any] | None:
        """The ``TokenUsage`` object a session record carries, if any.

        Two shapes, both from the same log: a usage chunk nests it under
        ``data.chunk.usage``, while a committed assistant message carries the
        step's usage at ``data.usage``.
        """
        data = record.get("data")
        if not isinstance(data, dict):
            return None
        kind = record.get("type")
        if kind == "assistant/chunk":
            chunk = data.get("chunk")
            if not isinstance(chunk, dict) or chunk.get("type") != "usage":
                return None
            usage = chunk.get("usage")
        elif kind == "assistant/message":
            usage = data.get("usage")
        else:
            return None
        return usage if isinstance(usage, dict) else None

    def _session_totals(self) -> dict[str, int] | None:
        """Token totals across every session log the run wrote.

        Subagent and workflow children each own a session file, and their calls
        are real calls, so every file under the root counts.

        A step's usage is counted once. The usage chunk is authoritative and
        the assistant message repeats it, so the message is read only for a
        step that produced no chunk -- which is how upstream defines the
        fallback, and what keeps a retried step (two chunks, one message) from
        being halved instead of summed.
        """
        root = self.logs_dir / "dsh" / "sessions"
        if not root.exists():
            return None

        chunked: dict[tuple[str, Any, Any], list[dict[str, Any]]] = {}
        committed: dict[tuple[str, Any, Any], dict[str, Any]] = {}

        for path in sorted(root.rglob("session.jsonl")):
            for line in path.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A trial killed by the timeout leaves a half-written line.
                    continue
                if not isinstance(record, dict):
                    continue
                usage = self._usage_of(record)
                if usage is None:
                    continue
                data = record["data"]
                key = (str(path), data.get("turn"), data.get("step"))
                if record.get("type") == "assistant/chunk":
                    chunked.setdefault(key, []).append(usage)
                else:
                    committed[key] = usage

        steps = [
            usage
            for key, usages in chunked.items()
            for usage in usages
        ] + [
            usage for key, usage in committed.items() if key not in chunked
        ]
        if not steps:
            return None

        totals = dict.fromkeys(("input", "output", "cache_read", "cache_write"), 0)
        for usage in steps:
            for field, key in (
                ("inputTokens", "input"),
                ("outputTokens", "output"),
                ("cacheReadTokens", "cache_read"),
                ("cacheWriteTokens", "cache_write"),
            ):
                value = usage.get(field)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
        return totals

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        totals = self._session_totals()
        if not totals:
            return
        # Upstream's counts are disjoint: inputTokens is uncached input only,
        # and billed input is the sum of the three. Harbor's n_input_tokens is
        # the total *including* cache, as omp, opencode and minion report it,
        # so they are summed here -- one number meaning one thing across the
        # catalog.
        context.n_input_tokens = (
            totals["input"] + totals["cache_read"] + totals["cache_write"]
        )
        # reasoningTokens is documented as already inside outputTokens, so
        # adding it would double-count exactly the harnesses that think most.
        context.n_output_tokens = totals["output"]
        context.n_cache_tokens = totals["cache_read"]
