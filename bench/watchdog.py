"""Ground truth on whether the endpoint was actually up.

When a trial dies with a transport error, the interesting question is whether
the endpoint was reachable at that moment. Nothing in a run answered it: the
harness reports that its request failed, not why, and by the time anyone looks
the container is gone. So a run could be blamed on "the endpoint dropped" with
no evidence either way -- which is exactly the wrong conclusion to reach by
default, because the alternative is a bug in the client or the adapter.

This samples the endpoint on a fixed cadence for the life of a run and writes
one JSON line per sample next to the trials. Correlating a trial's failure
timestamp against the samples around it answers the question directly.

Deliberately cheap and read-only: it asks for the server's *properties*, never
a completion. It takes no slot, generates no tokens, and cannot perturb the
measurement it exists to explain -- a watchdog that competed for the one slot
would manufacture the failures it was meant to observe.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HEALTH_FILENAME = "endpoint-health.jsonl"

#: Fast enough to bracket a trial that dies seconds after it starts, slow
#: enough that the file stays small over a benchmark measured in hours.
DEFAULT_INTERVAL_S = 10.0

#: A health check that hangs is itself a finding, but it must not hang forever.
PROBE_TIMEOUT_S = 10.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def health_url(base_url: str) -> str:
    """The cheapest read-only endpoint every OpenAI-compatible server exposes.

    ``/models`` rather than llama.cpp's ``/props`` so a hosted provider is
    sampled the same way, and it is a listing rather than a completion.
    """
    root = (base_url or "").rstrip("/")
    return f"{root}/models" if root.endswith("/v1") else f"{root}/v1/models"


def _probe_once(url: str, api_key: str = "") -> dict[str, Any]:
    """One read-only sample: reachable or not, and how slowly."""
    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", "harness-bench-watchdog")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    started = time.monotonic()
    sample: dict[str, Any] = {"at": _now(), "url": url}
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_S) as response:
            response.read(2048)
            sample.update(ok=True, status=response.status)
    except urllib.error.HTTPError as exc:
        # A status is still an answer: the server is up and talking, which is
        # the thing being measured. 4xx here is not an outage.
        sample.update(ok=True, status=exc.code, note="http error status")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        sample.update(ok=False, status=None, error=f"{type(exc).__name__}: {reason}")
    sample["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
    return sample


class EndpointWatchdog:
    """Samples an endpoint on a background thread for the life of a run."""

    def __init__(
        self,
        url: str,
        out_path: Path,
        *,
        api_key: str = "",
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        self.url = url
        self.out_path = out_path
        self.api_key = api_key
        self.interval_s = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=PROBE_TIMEOUT_S + 2)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = _probe_once(self.url, self.api_key)
            try:
                with self.out_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample) + "\n")
                    # Flushed per sample: the process is usually killed rather
                    # than shut down, and a buffered tail would lose exactly
                    # the samples around the failure worth reading.
                    handle.flush()
            except OSError:
                pass
            self._stop.wait(self.interval_s)


def load_samples(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / HEALTH_FILENAME
    if not path.exists():
        return []
    samples = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return samples


def health_around(run_dir: Path, when: str, window_s: float = 60.0) -> dict[str, Any]:
    """Endpoint state in a window around a moment, for one trial's failure.

    Returns {} when nothing was recorded, which is its own answer: diagnostics
    were off for that run, so nobody may claim the endpoint was or was not up.
    """
    samples = load_samples(run_dir)
    if not samples or not when:
        return {}
    try:
        target = datetime.fromisoformat(when.replace("Z", "+00:00"))
    except ValueError:
        return {}

    near = []
    for sample in samples:
        try:
            at = datetime.fromisoformat(str(sample.get("at")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if abs((at - target).total_seconds()) <= window_s:
            near.append(sample)
    if not near:
        return {}
    return {
        "n_samples": len(near),
        "n_unreachable": sum(1 for s in near if not s.get("ok")),
        "window_s": window_s,
        "worst_elapsed_ms": max((s.get("elapsed_ms") or 0) for s in near),
    }
