"""What to tell someone whose Docker is missing or stopped.

Docker is the one prerequisite this package cannot install for itself, so the
message shown when it is absent is the whole of the user's next step, and a
generic "install Docker" leaves a Windows user to work out on their own that
what they want is Docker Desktop with the WSL 2 backend rather than any of the
other things that phrase could mean.

Two callers print it -- ``harness-arena doctor`` and the dashboard's health
check -- and a hint that names the wrong installer for the platform is worse
than none at all, so it is written once here rather than twice and left to
drift.
"""

from __future__ import annotations

import sys

DESKTOP_URL = "https://www.docker.com/products/docker-desktop/"
ENGINE_URL = "https://docs.docker.com/engine/install/"


def install_hint() -> str:
    """The install instruction for this platform, when `docker` is not on PATH."""
    if sys.platform == "win32":
        # The reopen matters: the installer puts docker on PATH, and a shell
        # that was already open never sees it, which reads as a failed install.
        return (
            "not installed -- Docker Desktop, WSL 2 backend, "
            f"then reopen this terminal: {DESKTOP_URL}"
        )
    if sys.platform == "darwin":
        return f"not installed -- install Docker Desktop: {DESKTOP_URL}"
    return (
        "not installed -- install Docker Engine, then the sudo-less "
        f"post-install step: {ENGINE_URL}"
    )


def daemon_hint() -> str:
    """What to do when `docker` exists but the daemon does not answer."""
    if sys.platform in ("win32", "darwin"):
        return (
            "installed but not answering -- start Docker Desktop "
            "and wait for it to report Running"
        )
    return (
        "installed but not answering -- start it (sudo systemctl start docker), "
        "or check you are in the docker group"
    )
