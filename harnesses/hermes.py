"""Harbor agent adapter for NousResearch's hermes-agent, pointed at a local endpoint.

Harbor ships a built-in `hermes` agent, but it cannot drive a local
OpenAI-compatible server. It forwards ``OPENAI_BASE_URL`` and writes
``provider: auto`` into hermes's config -- and hermes-agent never reads
``OPENAI_BASE_URL``. Its endpoint resolution is::

    self.base_url = (base_url                                  # --provider/config
                     or CLI_CONFIG["model"].get("base_url", "")
                     or os.getenv("OPENROUTER_BASE_URL", "")) or None

so with the built-in agent the URL stays empty, hermes falls through to
OpenRouter, and every trial dies on ``HTTP 401: Missing Authentication header``
after installing and booting cleanly -- a failure that looks like a model
problem but is pure routing.

This subclass keeps Harbor's install path and ATIF trajectory conversion and
replaces only the config generation and CLI invocation:

* ``model.base_url`` in config.yaml, which is the one channel hermes honors.
* ``provider: llamacpp`` (normalizes to hermes's virtual ``local`` provider).
  This matters for more than labeling: hermes refuses a *non-loopback*
  ``model.base_url`` unless the configured provider is already ``custom`` or a
  local-server alias, specifically so a stale cloud URL cannot hijack a local
  session. Any LAN address is non-loopback, so if your model server is on
  another machine the base URL is silently discarded without this.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, override

import yaml
from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.hermes import Hermes as HarborHermes
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

HERMES_HOME = "/tmp/hermes"

#: hermes-agent refuses to initialise below this and exits during startup:
#: "has a context window of N tokens, which is below the minimum 64,000
#: required by Hermes Agent". Harbor sees only a non-zero exit, so the reason
#: is in the agent log rather than in the run's error. bench.runner checks it
#: before launching; this constant is the same floor for a bare `harbor run`.
MIN_CONTEXT_WINDOW = 64_000


class Hermes(HarborHermes):
    """hermes-agent against an arbitrary OpenAI-compatible endpoint."""

    def __init__(
        self,
        *args: Any,
        base_url: str | None = None,
        api_key: str = "local",
        max_turns: int | str = 90,
        provider: str = "llamacpp",
        # hermes auto-detects the window from the provider, and its own example
        # config names our exact case as one where that goes wrong: "a local
        # server with a custom num_ctx". The runner probes the real value, so
        # pass it rather than let hermes guess. Absent -> hermes decides.
        context_window: int | str | None = None,
        max_tokens: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._base_url = base_url or self._get_env("OPENAI_BASE_URL")
        # Never empty. hermes picks OPENAI_API_KEY for any non-openrouter base
        # URL, and an empty value short-circuits credential resolution before
        # the request is built -- which surfaces as an auth failure against a
        # server that never wanted auth. A local server ignores the value.
        self._api_key = api_key or "local"
        self._max_turns = int(max_turns)
        self._provider = provider
        self._context_window = int(context_window or 0) or None
        self._max_tokens = int(max_tokens or 0) or None
        if self._context_window and self._context_window < MIN_CONTEXT_WINDOW:
            # Fail with the reason rather than letting hermes exit non-zero
            # into a bare NonZeroAgentExitCodeError. Only when a window was
            # supplied: absent means hermes detects its own, and second-guessing
            # that here would refuse runs that would have worked.
            raise ValueError(
                f"hermes-agent requires at least {MIN_CONTEXT_WINDOW:,} tokens "
                f"of context and was given {self._context_window:,}. Serve a "
                f"larger window and set endpoint.context_window to match."
            )

    @staticmethod
    @override
    def name() -> str:
        return "hermes"

    def _model_id(self) -> str:
        if not self.model_name:
            raise ValueError("hermes requires --model")
        return (
            self.model_name.split("/", 1)[1]
            if "/" in self.model_name
            else self.model_name
        )

    def _local_config_yaml(self) -> str:
        config: dict[str, Any] = {
            # Dict form: hermes merges these into its model defaults, and
            # model.base_url is the only endpoint channel it actually reads.
            "model": {
                "default": self._model_id(),
                "base_url": self._base_url,
                # context_length is the TOTAL window, input + output together --
                # it is what compression.threshold is a percentage of, so a wrong
                # value makes hermes compress at the wrong point. max_tokens is
                # the per-response output cap.
                **({"context_length": self._context_window}
                   if self._context_window else {}),
                **({"max_tokens": self._max_tokens} if self._max_tokens else {}),
            },
            "provider": self._provider,
            "toolsets": ["hermes-cli"],
            "agent": {"max_turns": self._max_turns},
            # Benchmark hygiene: hermes's headline feature is a learning loop that
            # persists skills and a user model across sessions. Left on, task N
            # would be solved by an agent shaped by tasks 1..N-1, so the run would
            # measure accumulated memory rather than the harness. Harbor's own
            # built-in agent disables these for the same reason.
            "memory": {"memory_enabled": False, "user_profile_enabled": False},
            "compression": {"enabled": True, "threshold": 0.85},
            "terminal": {"backend": "local", "timeout": 180},
            "delegation": {"max_iterations": 50},
            "checkpoints": {"enabled": False},
        }
        return yaml.dump(config, default_flow_style=False, sort_keys=False)

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
                "hermes needs an endpoint. Pass --ak base_url=http://host:port/v1"
            )

        env = {
            "HERMES_HOME": HERMES_HOME,
            "TERMINAL_ENV": "local",
            # hermes picks OPENAI_API_KEY for any non-openrouter base URL.
            # llama-server ignores the value, but an empty key short-circuits
            # credential resolution before the request is ever built.
            "OPENAI_API_KEY": self._api_key,
            "HARBOR_INSTRUCTION": instruction,
        }

        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {HERMES_HOME} && "
                f"cat > {HERMES_HOME}/config.yaml << 'EOF'\n{self._local_config_yaml()}EOF"
            ),
            env=env,
            timeout_sec=30,
        )

        for extra in (
            self._build_register_mcp_servers_command(),
            self._build_register_skills_command(),
        ):
            if extra:
                await self.exec_as_agent(
                    environment, command=extra, env=env, timeout_sec=30
                )

        parts = [
            "hermes --yolo chat",
            '-q "$HARBOR_INSTRUCTION"',
            "-Q",
            f"--model {shlex.quote(self._model_id())}",
            f"--provider {shlex.quote(self._provider)}",
        ]
        toolsets = self._resolved_flags.get("toolsets")
        if toolsets:
            parts.append(f"--toolsets {shlex.quote(str(toolsets))}")

        run_cmd = (
            'export PATH="$HOME/.local/bin:$PATH" && '
            f"{' '.join(parts)} "
            "2>&1 | stdbuf -oL tee /logs/agent/hermes.txt"
        )

        try:
            await self.exec_as_agent(environment, command=run_cmd, env=env)
        finally:
            # Export the session even on failure -- a crashed run's transcript is
            # exactly the one worth reading.
            #
            # No `--source cli`: Harbor's built-in agent passes it, and it
            # matches nothing. Verified in a live container -- with the filter
            # `hermes sessions export` reports "Exported 0 sessions" and writes a
            # 0-byte file; without it, "Exported 1 sessions" and 30KB. The
            # session's own record even reads source='cli', so the filter is
            # matching some other field. A trial container holds exactly one
            # session, so the filter buys nothing anyway.
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        'export PATH="$HOME/.local/bin:$PATH" && '
                        "hermes sessions export /logs/agent/hermes-session.jsonl "
                        "2>/dev/null || true"
                    ),
                    env={"HERMES_HOME": HERMES_HOME},
                    timeout_sec=60,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------

    def _session_totals(self) -> dict[str, int] | None:
        """Token totals from the exported session's own summary fields.

        hermes does not put usage on individual messages -- every message in a
        real export has no ``usage`` key at all -- so Harbor's inherited parser,
        which walks ``messages[].usage.prompt_tokens``, always yields zero. The
        real numbers are session-level aggregates on the record itself.
        """
        path = self.logs_dir / "hermes-session.jsonl"
        if not path.exists():
            return None

        totals = {"input": 0, "output": 0, "cache_read": 0, "reasoning": 0}
        found = False
        # encoding= is not optional: without it this is the locale encoding,
        # which mojibakes the transcript on a cp1252 machine instead of failing
        # loudly. errors= then covers a transcript that is not valid UTF-8.
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            for key, field in (
                ("input", "input_tokens"),
                ("output", "output_tokens"),
                ("cache_read", "cache_read_tokens"),
                ("reasoning", "reasoning_tokens"),
            ):
                value = record.get(field)
                if isinstance(value, int):
                    totals[key] += value
                    found = True
        return totals if found else None

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        # Let Harbor build its ATIF trajectory.json from the transcript first;
        # only the token numbers it derives are wrong.
        #
        # Guarded because this is bookkeeping running *after* a finished
        # trajectory: Harbor reads the session with an unqualified read_text(),
        # so under a non-UTF-8 locale a single character the model typed can
        # raise and take the whole trial with it. bench.runner sets PYTHONUTF8
        # so that cannot happen here, but this adapter is also reachable
        # through a bare `harbor run`, where nothing has set it. Losing the
        # trajectory is a real cost; losing the trial is a much larger one, and
        # the token totals below are recovered either way.
        try:
            super().populate_context_post_run(context)
        except UnicodeDecodeError as exc:
            self.logger.warning(
                f"Harbor could not decode the hermes session ({exc}); "
                f"no ATIF trajectory for this trial. Token totals are "
                f"unaffected. Run through `harness-arena bench`, which sets "
                f"PYTHONUTF8=1, to avoid this."
            )

        totals = self._session_totals()
        if not totals:
            return
        # Against a local llama-server both cache counters are 0, so whether
        # hermes counts cache reads inside input_tokens is unobservable here.
        # Reported separately rather than summed, so a cached setup would show
        # an obvious anomaly instead of a silently inflated input count.
        context.n_input_tokens = totals["input"]
        context.n_output_tokens = totals["output"] + totals["reasoning"]
        context.n_cache_tokens = totals["cache_read"]
