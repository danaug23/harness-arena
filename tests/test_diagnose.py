"""Every failure this rig has hit must still be recognised, with its fix.

The value of `bench/diagnose.py` is entirely in the matching: a signature that
stops matching is worse than no signature, because the reader is then shown raw
text they were promised an explanation for. So the cases here are the *actual*
failure text from the runs that produced each entry, not paraphrases.

    python tests/test_diagnose.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import diagnose  # noqa: E402
from bench.config import Config, EndpointConfig  # noqa: E402

failures: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<58} got={got!r} want={want!r}")
    if not ok:
        failures.append(label)


def check_true(label: str, got: object) -> None:
    check(label, bool(got), True)


# ---------------------------------------------------------------------------
# Signatures, against the text that actually appeared
# ---------------------------------------------------------------------------

print("-- signatures --")

# A tau3-bench sweep, all seven harnesses, each dead within seconds.
TAU3 = """
      Missing Environment Variables
  Variable         |  Phase
  OPENAI_API_KEY   |  [environment.env]
  OPENAI_BASE_URL  |  [environment.env]
  OPENAI_API_KEY   |  [verifier.env]
  OPENAI_BASE_URL  |  [verifier.env]

Export them in your shell or pass --env-file.
  [!] claude-code exited 1
"""

# The next failure, once the first was fixed: a file that was never written.
LONG_PATH = (
    "FileNotFoundError: [Errno 2] No such file or directory: "
    "'C:\\\\Users\\\\x\\\\.cache\\\\harbor\\\\tasks\\\\packages\\\\sierra-research"
    "\\\\tau3-bench__tau3-telecom-mobile-data-issue-bad-network-preference"
    "\\\\ebb4bbcc\\\\task.toml'"
)

# Two C++ trials of a 225-task aider-polyglot run.
UNSCORABLE = (
    "harbor.verifier.verifier.RewardFileNotFoundError: No reward file found at "
    "runs/job/trial/verifier/reward.txt or runs/job/trial/verifier/reward.json"
)

SUBSET = (
    "ValueError: No tasks matched the filter(s) ['broken-python']. There are "
    "225 tasks available in this dataset. Example task names: ['polyglot_cpp_x']"
)

for label, text, want in (
    ("a benchmark needing its own endpoint", TAU3, "dataset-env"),
    ("a task whose files never downloaded", LONG_PATH, "long-paths"),
    ("a subset that belongs to another benchmark", SUBSET,
     "subset-dataset-mismatch"),
    ("work that could not be scored", UNSCORABLE, "unscorable"),
    ("a cold image pull", "EnvironmentStartTimeoutError after 600s", "cold-pull"),
    ("an agent out of time", "AgentTimeoutError", "agent-timeout"),
    ("an unreachable endpoint",
     "APIConnectionError: connection refused", "endpoint-down"),
    ("a missing benchmark", "No dataset to run. Pass --dataset", "no-dataset"),
):
    found = diagnose.explain(text)
    check(label, found.id if found else None, want)

# Every match has to carry a way out, or it is a relabelled complaint.
for sig in diagnose.SIGNATURES:
    check_true(f"{sig.id} offers at least one fix", len(sig.fixes) >= 1)
    check_true(f"{sig.id} explains itself", len(sig.detail) > 40)
    check_true(f"{sig.id} has a known severity", sig.severity in diagnose.SEVERITIES)

# The honest answer for something nobody has seen is nothing at all: inventing
# an explanation is worse than the raw text, which the reader still has.
check("an unknown failure is not explained away",
      diagnose.explain("wharrgarbl 17 flibbertigibbet"), None)
check("empty input is safe", diagnose.explain(""), None)
check("None is safe", diagnose.explain(None), None)

# Real logs are full of colour and megabytes long; the fatal error is at the end.
check("colour codes do not stop a match",
      (diagnose.explain_log("\x1b[31m" + TAU3 + "\x1b[0m") or {}).id
      if diagnose.explain_log("\x1b[31m" + TAU3 + "\x1b[0m") else None,
      "dataset-env")
_buried = ("filler line\n" * 5000) + TAU3
check("a match at the end of a long log is still found",
      (diagnose.explain_log(_buried) or type("x", (), {"id": None})()).id,
      "dataset-env")
# ...and one scrolled far off the top is not, which is the point of a tail: a
# stale error from an hour ago must not be reported as the current one.
_stale = TAU3 + ("filler line\n" * 20000)
check("an error long since scrolled past is not resurrected",
      diagnose.explain_log(_stale), None)


# ---------------------------------------------------------------------------
# Findings are renderable by both front ends
# ---------------------------------------------------------------------------

print("\n-- findings --")

_f = diagnose.Finding(id="x", title="T", severity="warn", detail="d",
                      fixes=["a", "b"], docs="docs/TROUBLESHOOTING.md")
_d = _f.to_dict()
check("a finding serializes for the API",
      sorted(_d), ["detail", "docs", "fixes", "id", "ok", "severity", "title"])
check("...and reports its own severity as not-ok", _d["ok"], False)
check("an ok finding says so", diagnose.Finding(id="y", title="T").to_dict()["ok"],
      True)

check("worst() finds a failure",
      diagnose.worst([diagnose.Finding(id="a", title="a"),
                      diagnose.Finding(id="b", title="b", severity="fail")]),
      "fail")
check("worst() prefers a failure over a warning",
      diagnose.worst([diagnose.Finding(id="a", title="a", severity="warn"),
                      diagnose.Finding(id="b", title="b", severity="fail")]),
      "fail")
check("worst() of nothing wrong is ok",
      diagnose.worst([diagnose.Finding(id="a", title="a")]), "ok")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

print("\n-- checks --")

# The endpoint is skipped: it is the only check that makes a network request,
# and this suite must not need one.
with tempfile.TemporaryDirectory() as _tmp:
    _config = Config(
        endpoint=EndpointConfig(base_url="http://example.invalid:1/v1"),
        runs_dir=_tmp,
    )
    _findings = diagnose.run_checks(_config, include_endpoint=False)

check_true("checks produce findings", len(_findings) >= 5)
check("every check has an id", [f for f in _findings if not f.id], [])
check("every check has a title", [f.id for f in _findings if not f.title], [])
check("every severity is known",
      [f.id for f in _findings if f.severity not in diagnose.SEVERITIES], [])
# The rule that keeps the panel worth reading.
check("every problem carries a fix",
      [f.id for f in _findings if not f.ok and not f.fixes], [])
check("the endpoint was not contacted",
      [f.id for f in _findings if f.id == "endpoint"], [])

# Long paths is platform-specific and must not fail a Linux CI run.
_lp = diagnose.check_long_paths()
if sys.platform == "win32":
    check_true("long paths is judged on Windows", _lp.severity in ("ok", "warn"))
else:
    check("long paths is not applicable off Windows", _lp.severity, "ok")

# A catalog check must survive a catalog it cannot read rather than raising --
# it is the thing reporting problems, so it cannot become one.
check_true("the catalog check returns a finding either way",
           isinstance(diagnose.check_catalog(), diagnose.Finding))


print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
