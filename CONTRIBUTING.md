# Contributing

The most useful contribution is usually **a new harness adapter**, that is the
extension point this repo exists for, and it is a YAML block plus, sometimes, a
Python file. See [Adding a harness](README.md#adding-a-harness).

## Setup

```bash
git clone https://github.com/danaug23/harness-arena
cd harness-arena

python -m venv .venv && source .venv/bin/activate    # or: conda env create -f environment.yml
pip install -e .

harness-arena init      # point it at your model server
harness-arena doctor    # confirm Docker, Harbor and the endpoint work
```

## Tests

Seven suites, all fast, none needing benchmark data, a model server, or Docker:

```bash
python tests/test_config.py      # config precedence and secret redaction
python tests/test_api.py         # control-plane auth, validation, redaction
python tests/test_supervisor.py  # run guards and stop semantics
python tests/test_collect.py     # comparison rules and statistics
python tests/test_tokens.py      # adapter parsers and generated configs
python tests/test_local_agents.py # endpoint routing for Claude Code and Codex
node   tests/test_dashboard.mjs  # every dashboard render path
```

They cover what a real run validates too slowly or too late. A benchmark run
takes hours, so a silent zero in token accounting or a bad run pairing would not
surface until the results were already wrong.

Lint with `ruff check .` after `pip install -e ".[dev]"`.

**Fixtures must be synthetic.** Several fixtures were shaped from real runs,
which is fine, but every *value* in a committed test must be invented. Model
names, session ids, weights paths and hostnames captured from a live container
are exactly the kind of thing that leaks a maintainer's setup into a public
repo, and a fixture is the last place anyone thinks to look.

## Two rules that CI enforces

**No credentials, anywhere.** `config.yaml` is gitignored; `registry.yaml` is
not. Keys reach a harness through the `{api_key}` placeholder, which is resolved
at run time and scrubbed back out of manifests and printed commands. If you add
a path that carries a key, add a test to `tests/test_config.py` proving it does
not reach disk. A leak here is silent, everything keeps working, the credential
is just also in a file someone published.

**No machine-specific data.** This repo was extracted from one person's setup,
and CI fails on home directories and private IP ranges. When you need an address
in code, a test, or a doc, use a documentation-reserved one:

| Instead of | Use |
|---|---|
| `10.x.x.x`, `192.168.x.x`, `172.16-31.x.x` | `192.0.2.10` (RFC 5737) or `localhost` |
| a real hostname | `example.invalid` (RFC 6761) |
| a home directory in a path | `~/` or a path relative to the repo root |

## Adding a harness

1. If Harbor already ships an agent for it (`harbor run --help` lists them) and
   that agent can reach your endpoint, you need no Python, just a registry block.
2. Otherwise add `harnesses/<name>.py`: a `BaseInstalledAgent` subclass with
   `install()`, `run()`, and usually `populate_context_post_run()` for token
   accounting. `harnesses/omp.py` is the worked example.
3. Add the block to `harnesses/registry.yaml`.

**Color is handled for you.** Agents write for a terminal and many emit ANSI
unconditionally with no way to switch it off. Every path that reads captured
output, the live feed, the run console and the error messages in the results
index, strips escape sequences and collapses spinner carriage returns
centrally, so an adapter never has to. Do not add per-harness handling for it.

**The live feed is format-agnostic.** Whatever your agent writes to its log,
prose, NDJSON, or a token-at-a-time event stream, `bench/activity.py` reads it
without an entry per harness: unfamiliar event types surface their text, tool
calls and results are recognized under the several names harnesses give them,
and a stream of one-token events is reassembled into paragraphs. If your log
renders badly, fix it there rather than in the adapter, so the next harness
inherits the fix.

Two things reviewers will look for:

- **Token accounting is real.** Harbor's inherited parsers assume a message
  shape many harnesses do not produce, and they fail by reporting zero rather
  than by raising. Both existing adapters had to override it. Add a fixture to
  `tests/test_tokens.py`.
- **Cross-task memory is off.** If a harness persists skills or a user profile
  between sessions, task N is solved by an agent shaped by tasks 1..N-1, and the
  run measures accumulated memory instead of the harness. Both existing adapters
  disable it explicitly.

## Reporting results

Numbers are only comparable when the model, subset, and `agent_timeout_multiplier`
all match, the dashboard refuses to pair runs that differ. When you report a
result, say which model and multiplier produced it. `harness-arena export`
writes a self-contained snapshot; check what is in it before posting one, since
it inlines run manifests.
