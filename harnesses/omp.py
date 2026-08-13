"""Harbor agent adapter for oh-my-pi (``omp``).

Modeled on Harbor's own ``harbor/agents/installed/pi.py`` -- omp is a fork of
Mario Zechner's Pi and keeps its ``--print --mode json`` NDJSON event stream, so
the token-accounting parser is line-for-line the same shape.

Two things differ from the built-in Pi agent:

* **Install** uses the official ``scripts/install.sh --binary`` path rather than
  npm. The published ``@oh-my-pi/pi-coding-agent`` package declares
  ``engines: {bun: '>=1.3.14'}`` and will not run under Node, so an
  ``npm install -g`` produces a binary that cannot start. The prebuilt
  ``omp-linux-x64`` release asset has no runtime prerequisites at all.
* **Model routing** goes through an explicit ``models.yml`` custom provider
  rather than ``OPENAI_BASE_URL``. A local llama-server advertises no context
  window, and omp needs one declared up front to size compaction correctly.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, override

import yaml
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

#: The installer, fetched from the pinned ref rather than from ``main``.
#: ``--ref`` selects which build the script *downloads*; the script itself
#: comes from whatever URL is used here, so pinning only the flag would leave
#: the installer floating and an upstream change to it would still reach a
#: pinned run. See harnesses/hermes.py for the run this actually cost.
_RAW_BASE = "https://raw.githubusercontent.com/can1357/oh-my-pi"


def _install_script_url(ref: str | None) -> str:
    return f"{_RAW_BASE}/{ref or 'main'}/scripts/install.sh"

# Where install.sh drops the binary (PI_INSTALL_DIR default).
_INSTALL_DIR = "$HOME/.local/bin"
_CONFIG_DIR = "$HOME/.omp/agent"

# Every model role omp supports. Anything left unset would resolve against
# omp's built-in catalog and try to reach a cloud provider we have no
# credentials for, so all of them are pinned to the local model via @default.
_MODEL_ROLES = (
    "smol",
    "slow",
    "vision",
    "plan",
    "designer",
    "commit",
    "tiny",
    "task",
    "advisor",
)


class Omp(BaseInstalledAgent):
    """Installs oh-my-pi in the task environment and runs it headless."""

    _OUTPUT_FILENAME = "omp.txt"
    _PROVIDER = "local"

    CLI_FLAGS = [
        CliFlag(
            "thinking",
            cli="--thinking",
            type="enum",
            choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        ),
        CliFlag("max_time", cli="--max-time", type="str"),
    ]

    def __init__(
        self,
        *args: Any,
        base_url: str | None = None,
        # No default: the served model's real window is probed once by the
        # runner and substituted in. A constant here would be a second,
        # silently-wrong answer the moment a different model is loaded, so
        # when it is absent the setting is omitted and the harness decides.
        context_window: int | str | None = None,
        max_tokens: int | str | None = None,
        api: str = "openai-completions",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Harbor passes --ak values through as strings.
        self._base_url = (
            base_url or self._get_env("OMP_BASE_URL") or self._get_env("OPENAI_BASE_URL")
        )
        self._context_window = int(context_window or 0) or None
        self._max_tokens = int(max_tokens or 0) or None
        self._api = api
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @staticmethod
    @override
    def name() -> str:
        return "omp"

    @override
    def get_version_command(self) -> str | None:
        return f'export PATH="{_INSTALL_DIR}:$PATH"; omp --version'

    @override
    def parse_version(self, stdout: str) -> str:
        # `omp --version` prints a banner; the version is the last non-empty line.
        lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
        return lines[-1] if lines else stdout.strip()

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # Terminal-Bench task images are not uniform -- most are Debian-derived
        # but some are Alpine or RPM-based, and a hard-coded apt-get turns those
        # into an install failure that reads like an agent bug. Only curl is
        # actually required; the rest sharpen omp's own tooling when present, so
        # the whole step is best-effort and the real gate is `omp --version`.
        await self.exec_as_root(
            environment,
            command=(
                "set -u; "
                "pkgs='curl ca-certificates git ripgrep'; "
                "if command -v apt-get >/dev/null 2>&1; then "
                "  apt-get update >/dev/null 2>&1; "
                "  apt-get install -y $pkgs >/dev/null 2>&1 || apt-get install -y curl ca-certificates; "
                "elif command -v apk >/dev/null 2>&1; then "
                "  apk add --no-cache $pkgs >/dev/null 2>&1 || apk add --no-cache curl ca-certificates; "
                "elif command -v dnf >/dev/null 2>&1; then "
                "  dnf install -y curl ca-certificates git ripgrep >/dev/null 2>&1 || dnf install -y curl; "
                "elif command -v yum >/dev/null 2>&1; then "
                "  yum install -y curl ca-certificates git >/dev/null 2>&1 || yum install -y curl; "
                "fi; "
                "command -v curl >/dev/null 2>&1"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        ref_flag = f" --ref {shlex.quote(self._version)}" if self._version else ""
        script_url = _install_script_url(self._version)
        await self.exec_as_agent(
            environment,
            command=(
                "set -eu; "
                f'export PI_INSTALL_DIR="{_INSTALL_DIR}"; '
                f"curl -fsSL {shlex.quote(script_url)} | sh -s -- --binary{ref_flag} && "
                f'export PATH="{_INSTALL_DIR}:$PATH" && '
                "omp --version"
            ),
        )

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def _model_id(self) -> str:
        """The bare model id, without the ``local/`` provider prefix."""
        if not self.model_name:
            raise ValueError(
                "omp requires --model. Use '<provider>/<model-id>', e.g. "
                "'local/my-model'."
            )
        return (
            self.model_name.split("/", 1)[1]
            if "/" in self.model_name
            else self.model_name
        )

    def _model_selector(self) -> str:
        return f"{self._PROVIDER}/{self._model_id()}"

    def _build_models_yaml(self) -> str:
        provider: dict[str, Any] = {
            "baseUrl": self._base_url,
            "api": self._api,
            "models": [
                {
                    "id": self._model_id(),
                    "name": self._model_id(),
                    **({"contextWindow": self._context_window}
                       if self._context_window else {}),
                    **({"maxTokens": self._max_tokens} if self._max_tokens else {}),
                }
            ],
        }
        if self._api_key:
            provider["apiKey"] = self._api_key
            provider["authHeader"] = True
        else:
            # Keyless local endpoint; without this omp demands a credential.
            provider["auth"] = "none"
        return yaml.dump({"providers": {self._PROVIDER: provider}}, sort_keys=False)

    def _build_config_yaml(self) -> str:
        selector = self._model_selector()
        config: dict[str, Any] = {
            "modelRoles": {
                "default": selector,
                **{role: "@default" for role in _MODEL_ROLES},
            },
        }
        return yaml.dump(config, sort_keys=False)

    def _build_skills_command(self) -> str | None:
        if not self.skills_dir:
            return None
        return (
            f"mkdir -p $HOME/.agents/skills && "
            f"cp -r {shlex.quote(self.skills_dir)}/* $HOME/.agents/skills/ "
            f"2>/dev/null || true"
        )

    def _build_mcp_command(self) -> str | None:
        if not self.mcp_servers:
            return None
        servers: dict[str, Any] = {}
        for server in self.mcp_servers:
            if server.transport == "stdio":
                servers[server.name] = {
                    "command": server.command,
                    "args": server.args,
                }
            else:
                servers[server.name] = {"url": server.url}
        blob = yaml.dump({"mcpServers": servers}, sort_keys=False)
        return f"cat >> {_CONFIG_DIR}/config.yml << 'MCPEOF'\n{blob}MCPEOF"

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
                "omp needs an endpoint. Pass --ak base_url=http://host:port/v1 "
                "or set OMP_BASE_URL via --ae."
            )

        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {_CONFIG_DIR} && "
                f"cat > {_CONFIG_DIR}/models.yml << 'MODELSEOF'\n"
                f"{self._build_models_yaml()}MODELSEOF\n"
                f"cat > {_CONFIG_DIR}/config.yml << 'CONFIGEOF'\n"
                f"{self._build_config_yaml()}CONFIGEOF"
            ),
            timeout_sec=30,
        )

        for extra in (self._build_mcp_command(), self._build_skills_command()):
            if extra:
                await self.exec_as_agent(environment, command=extra, timeout_sec=30)

        cli_flags = self.build_cli_flags()
        if cli_flags:
            cli_flags += " "

        # The instruction rides in an env var: task instructions routinely contain
        # quotes, backticks and newlines that no amount of shell quoting survives
        # cleanly through the exec -> sh -c -> pipeline chain.
        await self.exec_as_agent(
            environment,
            command=(
                f'export PATH="{_INSTALL_DIR}:$PATH"; '
                f"omp --print --mode json --yolo "
                f"--session-dir /logs/agent/omp/sessions "
                f"--model {shlex.quote(self._model_selector())} "
                f"{cli_flags}"
                # `--` ends flag parsing: a task instruction that happens to
                # begin with `-` or `@` would otherwise be read as a flag or a
                # file reference instead of as the prompt.
                f'-- "$HARBOR_INSTRUCTION" '
                f"2>&1 </dev/null "
                # --line-buffered matters: without it grep holds ~4KB before
                # flushing, so the log lags the run and a trial killed by the
                # timeout loses the tail -- precisely the part worth reading.
                f"| grep -v --line-buffered '\"type\":\"message_update\"' "
                f"| stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
            ),
            env={"HARBOR_INSTRUCTION": instruction},
        )

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        output_file = self.logs_dir / self._OUTPUT_FILENAME
        if not output_file.exists():
            return

        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cost = 0.0

        for line in output_file.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "message_end":
                continue
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            usage = message.get("usage") or {}
            total_input += usage.get("input", 0)
            total_output += usage.get("output", 0)
            total_cache_read += usage.get("cacheRead", 0)
            cost = usage.get("cost") or {}
            total_cost += cost.get("total", 0.0)

        context.n_input_tokens = total_input + total_cache_read
        context.n_output_tokens = total_output
        context.n_cache_tokens = total_cache_read
        context.cost_usd = total_cost if total_cost > 0 else None
