"""Harbor agent adapter for opencode, pointed at an arbitrary endpoint.

opencode is a TypeScript agent that normally runs under Bun. Four things matter
here, three of them found by running the CLI rather than reading the docs:

* **Install the released binary, not the npm package.** The package expects a
  Bun runtime; the official installer drops a standalone binary with no runtime
  prerequisites. Same lesson the oh-my-pi adapter learned.

* **``run`` needs ``--auto``** ("auto-approve permissions that are not
  explicitly denied"). Without it the agent stops at the first permission
  prompt that nobody is there to answer, and the trial burns its budget waiting.
  Belt and braces: the generated config also sets ``permission: {"*": "allow"}``.

* **``--pure`` runs without external plugins.** A benchmark has to measure the
  harness, not whatever plugins happen to be installed in the image, so
  variability that is not the harness is switched off.

* **Token accounting cannot rely on the stream alone.** ``run --format json``
  emits JSONL, and the final ``step_finish`` event carries
  ``part.tokens.{input,output,reasoning,cache.read,cache.write}`` -- but it is a
  known upstream defect that the process can exit before emitting it. The
  session store still has the numbers, so the parser falls back to
  ``opencode export``. Without that fallback the efficiency panel would show a
  silent zero rather than an error.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent, CliFlag, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_INSTALL_SCRIPT = "https://opencode.ai/install"
_INSTALL_DIR = "$HOME/.opencode/bin"
_CONFIG_DIR = "/tmp/opencode"
_CONFIG = f"{_CONFIG_DIR}/opencode.json"

#: Provider id we register the endpoint under. opencode lets you name your own
#: provider, so this one is ours to choose.
_PROVIDER = "local"


class OpenCode(BaseInstalledAgent):
    """opencode against an arbitrary OpenAI-compatible endpoint."""

    _OUTPUT_FILENAME = "opencode.txt"
    _EXPORT_FILENAME = "opencode-session.json"

    CLI_FLAGS = [
        CliFlag("agent", cli="--agent", type="str"),
        CliFlag("variant", cli="--variant", type="str"),
    ]

    def __init__(
        self,
        *args: Any,
        base_url: str | None = None,
        api_key: str | None = None,
        # No default: the served model's real window is probed once by the
        # runner and substituted in. A constant here would be a second,
        # silently-wrong answer the moment a different model is loaded, so
        # when it is absent the setting is omitted and the harness decides.
        context_window: int | str | None = None,
        max_tokens: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._base_url = base_url or self._get_env("OPENCODE_BASE_URL")
        self._api_key = api_key
        self._context_window = int(context_window or 0) or None
        self._max_tokens = int(max_tokens or 0) or None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @staticmethod
    @override
    def name() -> str:
        return "opencode"

    @override
    def get_version_command(self) -> str | None:
        return f'export PATH="{_INSTALL_DIR}:$PATH"; opencode --version'

    @override
    def parse_version(self, stdout: str) -> str:
        lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
        return lines[-1] if lines else stdout.strip()

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # Only curl is a hard requirement; the rest sharpen opencode's own
        # tooling when present, so the whole step is best-effort.
        await self.exec_as_root(
            environment,
            command=(
                "set -u; "
                "pkgs='curl ca-certificates git unzip ripgrep'; "
                "if command -v apt-get >/dev/null 2>&1; then "
                "  apt-get update >/dev/null 2>&1; "
                "  apt-get install -y $pkgs >/dev/null 2>&1 || apt-get install -y curl ca-certificates unzip; "
                "elif command -v apk >/dev/null 2>&1; then "
                "  apk add --no-cache $pkgs >/dev/null 2>&1 || apk add --no-cache curl ca-certificates unzip; "
                "elif command -v dnf >/dev/null 2>&1; then "
                "  dnf install -y curl ca-certificates git unzip >/dev/null 2>&1 || dnf install -y curl unzip; "
                "elif command -v yum >/dev/null 2>&1; then "
                "  yum install -y curl ca-certificates git unzip >/dev/null 2>&1 || yum install -y curl unzip; "
                "fi; "
                "command -v curl >/dev/null 2>&1"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        version_env = f"VERSION={shlex.quote(self._version)} " if self._version else ""
        await self.exec_as_agent(
            environment,
            command=(
                "set -eu; "
                f"{version_env}curl -fsSL {_INSTALL_SCRIPT} | bash && "
                f'export PATH="{_INSTALL_DIR}:$PATH" && '
                "opencode --version"
            ),
        )

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def _model_id(self) -> str:
        """The bare model id, without the ``local/`` provider prefix."""
        if not self.model_name:
            raise ValueError(
                "opencode requires --model. Use '<provider>/<model-id>', e.g. "
                "'local/my-model'."
            )
        return (
            self.model_name.split("/", 1)[1]
            if "/" in self.model_name
            else self.model_name
        )

    def _model_selector(self) -> str:
        return f"{_PROVIDER}/{self._model_id()}"

    def _build_config_json(self) -> str:
        model = self._model_id()
        config: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
            "model": self._model_selector(),
            # Every role resolves to our endpoint. small_model left pointing at
            # the same place, or opencode reaches for a cloud default we have no
            # credentials for and the run dies mid-task on an auth error.
            "small_model": self._model_selector(),
            "provider": {
                _PROVIDER: {
                    # openai-compatible speaks /v1/chat/completions; the plain
                    # openai package expects /v1/responses, which a local server
                    # does not implement.
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "harness-bench endpoint",
                    "options": {
                        "baseURL": self._base_url,
                        # A local server ignores the value but several clients
                        # abandon the request when the key is empty.
                        "apiKey": self._api_key or "local",
                    },
                    "models": {
                        model: {
                            "name": model,
                            **({"limit": {
                                **({"context": self._context_window}
                                   if self._context_window else {}),
                                **({"output": self._max_tokens}
                                   if self._max_tokens else {}),
                            }} if self._context_window or self._max_tokens else {}),
                        }
                    },
                }
            },
            # Fully autonomous: there is no one to answer a prompt.
            "permission": {"*": "allow"},
            # Pinned behavior for a benchmark: no self-update mid-experiment,
            # and no session leaving the machine.
            "autoupdate": False,
            "share": "disabled",
        }
        return json.dumps(config, indent=2)

    def _agent_env(self, instruction: str) -> dict[str, str]:
        env = {
            "OPENCODE_CONFIG": _CONFIG,
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "HARBOR_INSTRUCTION": instruction,
        }
        if self._api_key:
            env["OPENAI_API_KEY"] = self._api_key
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
                "opencode needs an endpoint. Pass --ak base_url=http://host:port/v1"
            )

        env = self._agent_env(instruction)

        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {_CONFIG_DIR} && "
                f"cat > {_CONFIG} << 'EOF'\n{self._build_config_json()}\nEOF"
            ),
            env=env,
            timeout_sec=30,
        )

        try:
            await self.exec_as_agent(
                environment, command=self._run_command(), env=env
            )
        finally:
            # Export the session even on failure. It is both the transcript
            # worth reading after a crash and the fallback for token counts,
            # since the stream can end without its final step_finish event.
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f'export PATH="{_INSTALL_DIR}:$PATH" && '
                        f"opencode export --sanitize > /logs/agent/{self._EXPORT_FILENAME} "
                        "2>/dev/null || true"
                    ),
                    env=env,
                    timeout_sec=60,
                )
            except Exception:
                pass

    def _run_command(self) -> str:
        """The ``opencode run`` command line, in one place so it can be checked.

        Argument *order* is load-bearing here -- see the comment on the
        instruction below -- and a container is an expensive place to find that
        out, so this is kept separate from the exec that runs it.
        """
        parts = [
            "opencode run",
            "--format json",
            f"--model {shlex.quote(self._model_selector())}",
            # Without --auto the agent halts at the first permission prompt.
            "--auto",
            # Measure the harness, not whatever plugins the image happens to
            # carry.
            "--pure",
        ]
        for flag in ("agent", "variant"):
            value = self._resolved_flags.get(flag)
            if value:
                parts.append(f"--{flag} {shlex.quote(str(value))}")
        # The instruction goes last, behind `--`, and both halves of that
        # matter.
        #
        # `--` ends flag parsing. Shell quoting already makes the instruction a
        # single argv element, but the CLI still reads an element that *begins*
        # with `-` as a flag, so a task whose text opens with a bullet -- "- You
        # are given a PyTorch state dictionary" -- exits 1 with nothing but the
        # usage message. Exactly one of Terminal-Bench 2's 89 instructions does
        # that (pytorch-model-recovery), which is the worst frequency for a bug
        # like this: rare enough to look like a bad task, common enough to lose
        # a cell in every sweep. omp already guards it the same way.
        #
        # And because everything after `--` is a positional, a flag placed
        # after it would be swallowed into the prompt rather than parsed --
        # which is why the optional flags above are appended first.
        parts.append('-- "$HARBOR_INSTRUCTION"')

        return (
            f'export PATH="{_INSTALL_DIR}:$PATH" && '
            f"{' '.join(parts)} "
            f"2>&1 | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
        )

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate(node: Any, totals: dict[str, int]) -> bool:
        """Add one step's ``tokens`` object into the running totals."""
        if not isinstance(node, dict):
            return False
        tokens = node.get("tokens")
        if not isinstance(tokens, dict):
            return False
        found = False
        for key in ("input", "output", "reasoning"):
            value = tokens.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
                found = True
        cache = tokens.get("cache")
        if isinstance(cache, dict):
            value = cache.get("read")
            if isinstance(value, (int, float)):
                totals["cache_read"] += int(value)
                found = True
        return found

    def _stream_totals(self) -> dict[str, int] | None:
        """Token totals from ``step_finish`` events in the JSONL stream."""
        path = self.logs_dir / self._OUTPUT_FILENAME
        if not path.exists():
            return None

        totals = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0}
        found = False
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if str(event.get("type", "")).replace("-", "_") != "step_finish":
                continue
            found |= self._accumulate(event.get("part") or event, totals)
        return totals if found else None

    def _export_totals(self) -> dict[str, int] | None:
        """Fallback: the same numbers out of the exported session.

        Needed because ``run --format json`` is known to exit without emitting
        its final ``step_finish``. The session store has it regardless.
        """
        path = self.logs_dir / self._EXPORT_FILENAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(errors="replace"))
        except (json.JSONDecodeError, OSError):
            return None

        totals = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0}
        found = False

        def walk(node: Any) -> None:
            nonlocal found
            if isinstance(node, dict):
                kind = str(node.get("type", "")).replace("-", "_")
                if kind == "step_finish":
                    # The export nests tokens under "part" when it carries the
                    # stream event verbatim, and inline when it is a stored
                    # session part. Both shapes occur; accept either.
                    found |= self._accumulate(node.get("part") or node, totals)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(data)
        return totals if found else None

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        totals = self._stream_totals() or self._export_totals()
        if not totals:
            return
        context.n_input_tokens = totals["input"]
        # Reasoning tokens are billed and generated like output, so they belong
        # in the output count rather than being silently dropped.
        context.n_output_tokens = totals["output"] + totals["reasoning"]
        context.n_cache_tokens = totals["cache_read"]
