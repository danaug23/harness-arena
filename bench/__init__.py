"""harness-arena: run Terminal-Bench 2 across agent harnesses on one model.

Every path the tool touches is resolved here, because where they land depends
on how the tool was installed and getting that wrong is silent.

Run from a clone, or from `pip install -e .`, and the code sits in the checkout:
config, runs, caches and the harness catalog all belong beside it, which is what
the docs describe and where any existing run already is.

Installed from a wheel, the same code sits in site-packages. Writing runs or a
cache there would put your data inside a directory that the next `pip install`
is entitled to replace, and on a shared or read-only install it simply fails.
So an installed copy reads its packaged defaults from beside the code and keeps
everything you create in the directory you ran from.
"""

from pathlib import Path

#: Single source of truth. pyproject reads this, so a release cannot ship
#: a wheel whose --version disagrees with the version on the index.
__version__ = "0.1.2"

#: Where the code is. In a wheel install this is site-packages.
ROOT = Path(__file__).resolve().parent.parent

#: True when running from a source checkout, false when installed from a wheel.
#: pyproject.toml is the marker because it is the one file that is always at the
#: root of the checkout and never inside the installed package.
IS_CHECKOUT = (ROOT / "pyproject.toml").exists()

#: Where *your* files live. Identical to ROOT in a checkout, so nothing about a
#: clone changes; the working directory otherwise.
WORKSPACE = ROOT if IS_CHECKOUT else Path.cwd()

#: Default location for Harbor job output. Overridable via `runs_dir` in
#: config.yaml; use `Config.resolved_runs_dir()` rather than this constant
#: anywhere the user's configuration should win.
RUNS_DIR = WORKSPACE / "runs"

#: Everything an installed copy writes goes under one directory rather than
#: scattering dotfiles through whatever folder you happened to run from.
STATE_DIR = ROOT / "bench" if IS_CHECKOUT else WORKSPACE / ".harness-arena"

#: fingerprint -> display label. Local to a machine and gitignored: it records
#: the weights *you* have loaded, including their paths on your disk.
MODELS_CACHE_PATH = STATE_DIR / "models.json"

#: The harness catalog that ships inside the package, and the one you edit.
#: They are the same file in a checkout. Installed, the packaged copy is the
#: read-only default and the first edit writes a copy you own.
PACKAGED_REGISTRY_PATH = ROOT / "harnesses" / "registry.yaml"
USER_REGISTRY_PATH = (
    PACKAGED_REGISTRY_PATH if IS_CHECKOUT else STATE_DIR / "registry.yaml"
)

PACKAGED_SUBSET_DIR = ROOT / "bench" / "subsets"
USER_SUBSET_DIR = (
    PACKAGED_SUBSET_DIR if IS_CHECKOUT else STATE_DIR / "subsets"
)


def registry_path() -> Path:
    """The catalog to read: yours once you have edited one, else the packaged one."""
    return USER_REGISTRY_PATH if USER_REGISTRY_PATH.exists() else PACKAGED_REGISTRY_PATH


#: Resolved once for the many callers that just want to read it. Anything that
#: writes should target USER_REGISTRY_PATH, and anything long-running should
#: call registry_path() so a catalog created after import is still found.
REGISTRY_PATH = registry_path()


def subset_path(name: str) -> Path:
    """Where to read a named task list from, preferring one you made yourself."""
    mine = USER_SUBSET_DIR / f"{name}.txt"
    return mine if mine.exists() else PACKAGED_SUBSET_DIR / f"{name}.txt"


def subset_names() -> list[str]:
    """Every subset that can be run, packaged and user-created, deduplicated."""
    names = set()
    for directory in (PACKAGED_SUBSET_DIR, USER_SUBSET_DIR):
        if directory.is_dir():
            names.update(p.stem for p in directory.glob("*.txt"))
    return sorted(names)
