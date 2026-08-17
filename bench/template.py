"""``harness-arena template-fix`` -- repair a chat template that refuses a harness.

When an endpoint rejects a request shape (see `bench/wireshape.py`), the thing
that has to change is the model's chat template, not the harness's request. A
harness whose traffic we rewrite is no longer the harness being measured, and a
run altered that way cannot be compared to one that was not -- which is the
whole point of the rig.

So this reads the template the loaded weights carry, makes the smallest edit
that removes the refusal, writes it beside your config, and prints the flag
that puts it into service. It changes nothing on the server: applying it is a
restart you perform, because restarting someone's inference server out from
under them is not a diagnostic.

The edit is verifiable in the only way that means anything -- ask the endpoint
again once it is running. ``--verify`` does exactly that and reports whether
the shapes are now accepted, rather than claiming the patch worked because the
file was written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench import WORKSPACE, wireshape
from bench import registry as registry_mod
from bench.config import Config
from bench.probe import add_endpoint_args, config_from_args


def _served_id(config: Config) -> str:
    """The model id to ask about, without making its absence fatal."""
    if config.endpoint.model:
        return config.endpoint.model
    try:
        from bench.probe import probe

        return probe(config.endpoint).served_id
    except Exception:  # noqa: BLE001 - reported by the caller as "could not ask"
        return ""


def _report_shapes(config: Config, served_id: str) -> list[wireshape.Verdict]:
    """Ask the endpoint about every shape any catalogued harness sends."""
    catalog = registry_mod.load()
    return wireshape.check_selection(
        config.endpoint, catalog, sorted(catalog.get("harnesses") or {}),
        served_id=served_id,
    )


def _print_verdicts(verdicts: list[wireshape.Verdict]) -> None:
    if not verdicts:
        print("  no catalogued harness sends a shape worth asking about.")
        return
    for verdict in verdicts:
        senders = ", ".join(verdict.harnesses) or "-"
        if verdict.result == wireshape.ACCEPTED:
            print(f"  ok       {verdict.shape.id:<20} accepted  ({senders})")
        elif verdict.result == wireshape.REJECTED:
            print(f"  REFUSED  {verdict.shape.id:<20} HTTP {verdict.status}  ({senders})")
            if verdict.message:
                print(f"           {verdict.message}")
        else:
            print(f"  ?        {verdict.shape.id:<20} {verdict.why}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness-arena template-fix",
        description=(
            "Read the chat template behind the configured endpoint, report "
            "which harness request shapes it refuses, and write a patched copy "
            "that accepts them."
        ),
    )
    add_endpoint_args(parser)
    parser.add_argument(
        "-o", "--out", default="",
        help="Where to write the patched template. Default: alongside your "
             "config, named for the served model.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only re-ask the endpoint which shapes it accepts. Run this after "
             "restarting the server with the patched file -- it is the only "
             "check that establishes the patch actually took effect.",
    )
    parser.add_argument(
        "--print", dest="to_stdout", action="store_true",
        help="Write the patched template to stdout instead of to a file.",
    )
    args = parser.parse_args(argv)

    config = config_from_args(args)
    endpoint = config.endpoint.resolved_base_url()
    served_id = _served_id(config)

    print("harness-arena template-fix\n")
    print(f"  endpoint : {endpoint}")
    print(f"  model    : {served_id or '(could not be determined)'}")

    print("\n-- request shapes --")
    verdicts = _report_shapes(config, served_id)
    _print_verdicts(verdicts)

    if args.verify:
        refused = [v for v in verdicts if v.blocks]
        if refused:
            print("\nStill refused. If you have restarted the server with the "
                  "patched template, confirm the flag took effect:")
            print("  --chat-template-file <path>  (and check the server log "
                  "prints the file it loaded)")
            return 1
        unknown = [v for v in verdicts if v.result == wireshape.UNKNOWN]
        if unknown:
            print("\nNothing is refused, but not every shape could be asked "
                  "about. Nothing is blocked by that.")
            return 0
        print("\nEvery shape is accepted.")
        return 0

    print("\n-- chat template --")
    template, why = wireshape.fetch_template(config.endpoint)
    if not template:
        print(f"  {why}")
        return 1
    print(f"  {len(template.splitlines())} lines, {len(template)} chars")

    patch = wireshape.patch_template(template)
    if not patch.ok:
        print(f"  {patch.why}")
        # Not an error when the template is simply fine: a clean template and
        # no refusals is the state this command exists to reach.
        return 0 if not any(v.blocks for v in verdicts) else 1

    for change in patch.changes:
        print(f"  patched  {change}")
    print("\n  Every other branch is untouched, so a conversation that renders "
          "today renders identically. The only behaviour that changes is the "
          "one that currently aborts.")

    if args.to_stdout:
        sys.stdout.write(patch.text)
        return 0

    name = (served_id or "model").replace("/", "-").replace(":", "-")
    out = Path(args.out) if args.out else WORKSPACE / f"{name}.patched.jinja"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(patch.text, encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"\n  could not write {out}: {exc}", file=sys.stderr)
        return 1
    print(f"\n  wrote    {out}")

    print("\nNext:")
    print(f"  1. Restart llama-server with:  --chat-template-file {out}")
    print("  2. harness-arena template-fix --verify")
    print("\nThe patched file is a copy, not a change to your weights: drop the "
          "flag to go back to the template the GGUF carries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
