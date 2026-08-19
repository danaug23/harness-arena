"""Harbor agent adapter for OpenAI's Codex CLI, pointed at an arbitrary endpoint.

Harbor already ships a working ``codex`` agent, and this rig's rule is that a
built-in needs no adapter. This subclass exists because of one gap that matters
for a benchmark rather than for a user: **Codex takes its context window from
``model_context_window`` in config.toml and from nowhere else.** There is no
environment variable for it, so a registry-only block could not tell Codex the
window every other harness is told, and Codex would silently size compaction
against whatever its own table says about an unknown model id. That is exactly
the kind of difference this rig exists to hold constant, so it is worth a file.

Everything is expressed as ``-c key=value`` overrides through Harbor's
CLI_FLAGS descriptors, which are spliced into the command verbatim. That is the
whole adapter: no ``run()`` override, so Harbor keeps ownership of session
capture, the ATIF trajectory conversion and token accounting -- the parts most
likely to change upstream, and the parts worth not forking.

Three details decide how this is written, and all three were measured against
codex-cli 0.147.0 rather than read:

* **base_url keeps its ``/v1``.** Codex posts to ``<base_url>/responses``, so
  the endpoint's configured URL is used as-is. Verified by pointing Codex at a
  logging server: it requested ``/v1/responses`` from a base URL of
  ``http://host:8931/v1``. This is the opposite of what Claude Code needs from
  the same endpoint (see ``{base_url_root}`` in bench/runner.py), which is why
  the two harnesses are handed different spellings of one URL.

* **A named provider, not ``openai_base_url``.** Harbor's built-in path sets
  ``openai_base_url`` and leaves the reserved ``openai`` provider selected,
  which keeps OpenAI's own defaults in play. An explicit provider block states
  the wire protocol and the credential source outright, and it is what the
  local server was actually tested against.

* **``wire_api = "responses"``.** The current Codex documents this as the only
  supported value, and llama.cpp's server implements ``/v1/responses`` --
  including streaming, function calls and a usage object with cached-token
  counts. Left configurable anyway, because that is a property of the server
  rather than of Codex, and not every OpenAI-compatible server implements it.

Note that ``-c`` overrides are additive to config.toml rather than a
replacement for it, so Harbor's own MCP and auth setup still applies.

**Codex cannot be given an output cap, and that is not an omission here.**
Every other harness in the catalog takes ``{max_tokens}``; this one has nowhere
to put it. codex-cli 0.147.0's ``ConfigToml`` carries 96 keys -- among them
``model_context_window``, ``model_auto_compact_token_limit``,
``model_reasoning_effort`` and ``tool_output_token_limit`` -- and none of them
caps a single completion. ``model_max_output_tokens`` does not appear in the
binary at all. Measured by reading the shipped executable, the same way the
three details above were measured.

That matters for comparison rather than for correctness. On the 20260817 sweep
Claude Code was clamped at 16,384 output tokens and its largest completion was
exactly 16,384; Codex, uncapped, produced one of 92,436. The two harnesses tied
at 0.68 and were not running the same experiment. Nothing here can fix that --
the knob does not exist -- so the rig records which harnesses were capped
instead of assuming all of them were: see ``bench.runner.output_cap_for``, and
``agent_max_tokens_source`` in the manifest.
"""

from __future__ import annotations

from typing import Any, override

from harbor.agents.installed.base import CliFlag
from harbor.agents.installed.codex import Codex as _HarborCodex

#: Provider id registered for our endpoint. Codex reserves "openai", "ollama"
#: and "lmstudio", so this one is ours to choose.
_PROVIDER = "local"


class Codex(_HarborCodex):
    """Codex CLI against an arbitrary OpenAI-compatible endpoint."""

    # Order matters only for readability: Harbor joins these into one string.
    # `{{value}}` survives the f-string as the literal `{value}` that
    # build_cli_flags() formats. Values are spliced into the command without
    # shell quoting, so any TOML string value carries its own quotes -- and the
    # whole argument is wrapped in single quotes so the shell does not eat them.
    CLI_FLAGS = [
        *_HarborCodex.CLI_FLAGS,
        CliFlag(
            "model_provider",
            cli="-c",
            type="str",
            default=_PROVIDER,
            format="-c model_provider={value}",
        ),
        CliFlag(
            "base_url",
            cli="-c",
            type="str",
            format=f"-c 'model_providers.{_PROVIDER}.base_url=\"{{value}}\"'",
        ),
        CliFlag(
            "wire_api",
            cli="-c",
            type="enum",
            choices=["responses", "chat"],
            default="responses",
            format=f"-c 'model_providers.{_PROVIDER}.wire_api=\"{{value}}\"'",
        ),
        CliFlag(
            "api_key_env",
            cli="-c",
            type="str",
            default="OPENAI_API_KEY",
            format=f"-c 'model_providers.{_PROVIDER}.env_key=\"{{value}}\"'",
        ),
        CliFlag(
            "provider_label",
            cli="-c",
            type="str",
            default="harness-arena endpoint",
            format=f"-c 'model_providers.{_PROVIDER}.name=\"{{value}}\"'",
        ),
        # The reason this file exists. No default: the served model's real
        # window is probed once by the runner and substituted in, and a
        # constant here would be a second, silently-wrong answer the moment a
        # different model is loaded. Absent, Codex falls back to its own guess.
        CliFlag(
            "context_window",
            cli="-c",
            type="int",
            format="-c model_context_window={value}",
        ),
    ]

    @staticmethod
    @override
    def name() -> str:
        # Distinct from the built-in "codex" so a run is never ambiguous about
        # which of the two produced it.
        return "codex-local"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self._resolved_flags.get("base_url"):
            raise ValueError(
                "codex needs an endpoint. Pass --ak base_url=http://host:port/v1 "
                "(keep the /v1: codex requests <base_url>/responses)."
            )
