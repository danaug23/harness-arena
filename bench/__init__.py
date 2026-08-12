"""harness-bench: run Terminal-Bench 2 across agent harnesses on one model."""

from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "harnesses" / "registry.yaml"

#: Default location for Harbor job output. Overridable via `runs_dir` in
#: config.yaml; use `Config.resolved_runs_dir()` rather than this constant
#: anywhere the user's configuration should win.
RUNS_DIR = ROOT / "runs"

#: fingerprint -> display label. Local to a machine and gitignored: it records
#: the weights *you* have loaded, including their paths on your disk.
MODELS_CACHE_PATH = ROOT / "bench" / "models.json"
