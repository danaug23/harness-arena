"""Start, watch and stop benchmark runs on behalf of the UI.

The CLI runs a benchmark in the foreground and you stop it with Ctrl-C. The
dashboard cannot do that, so this module owns the same subprocess with three
extra obligations the terminal handled implicitly:

**One run at a time.** Harness runs are strictly sequential by design -- one
endpoint backs every harness, so two overlapping runs would each measure the
other's queueing delay as if it were their own latency. A terminal enforced this
socially; a web button has to enforce it for real, so ``start()`` refuses while
a run is active.

**Stopping means the whole tree, and then the containers.** ``harbor run``
spawns child processes *and* Docker containers, and the two need different
handling. The children die with the process group, so the child is started in
its own group and the group is signalled. The containers do not: Docker owns
them, not us, and killing the process leaves them running with their agents
still generating against the endpoint -- so "stopped" looks to the user like
nothing happened. Harbor removes them when it exits cleanly, but the interrupt
that stops the runner usually kills it first. So after the process is gone we
give Harbor a moment and then remove whatever containers this run created,
identified by exclusion against the ones that were already running.

**A killed run must not read as still running.** A job that is terminated never
writes ``finished_at``, so without an explicit marker it renders as an in-flight
benchmark forever and its partial results look like a run still in progress
rather than a baseline. On stop we write ``stopped_at`` and ``stopped_reason``
into each manifest this run created; ``bench/collect.py`` already reads them.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench import ROOT, WORKSPACE
from bench.config import Config, scrub, strip_ansi

MANIFEST_NAME = "harness-bench.json"

#: Lines of output kept in memory for the live log panel. The full output always
#: goes to disk; this is only what the UI can scroll back through.
LOG_BUFFER_LINES = 2000

#: Grace period between asking a run to stop and killing it. Harbor tears down
#: containers on SIGINT, and that teardown is worth waiting for -- a hard kill
#: is what leaves orphaned containers holding disk.
STOP_GRACE_S = 20.0

#: How long to let Harbor remove its own containers after the process dies
#: before removing them ourselves. A graceful teardown is tidier, but a task
#: image container left running holds multiple GB and keeps calling the model.
CONTAINER_GRACE_S = 15.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Job:
    """One `harness-arena bench` invocation."""

    id: str
    argv: list[str]
    harnesses: list[str]
    started_at: str
    log_path: Path
    #: bench | prepull. Both are long subprocesses of ours that stream output,
    #: and neither should overlap the other -- a prepull competes with a running
    #: benchmark for disk and network on the same box.
    kind: str = "bench"
    #: starting | running | finished | failed | stopped
    status: str = "starting"
    returncode: int | None = None
    finished_at: str | None = None
    stopped_reason: str | None = None
    pid: int | None = None
    #: Job directories that existed before this run, so new ones are known to be
    #: ours without having to parse the runner's stdout.
    _preexisting: set[str] = field(default_factory=set)
    #: Task containers that existed before this run. Anything Harbor-shaped that
    #: appears afterwards belongs to us, and must be removed when we stop --
    #: containers are owned by the Docker daemon, not by the process we killed.
    _precontainers: set[str] = field(default_factory=set)
    #: Task networks that existed before this run. Each compose project creates
    #: one, they outlive both the kill and the container removal, and every one
    #: holds a subnet out of a pool of 32.
    _prenetworks: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "harnesses": self.harnesses,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "stopped_reason": self.stopped_reason,
            "pid": self.pid,
            "active": self.status in ("starting", "running"),
        }


class SupervisorError(RuntimeError):
    """A request the supervisor will not carry out."""


class Supervisor:
    """Owns at most one running benchmark process."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._job: Job | None = None
        self._process: subprocess.Popen[str] | None = None
        self._buffer: deque[str] = deque(maxlen=LOG_BUFFER_LINES)
        self._reader: threading.Thread | None = None

    # -- introspection ----------------------------------------------------

    @property
    def config(self) -> Config:
        return self._config

    def set_config(self, config: Config) -> None:
        """Adopt configuration edited through the UI.

        Refused mid-run: the model and endpoint are what a run's results *mean*,
        and swapping them under an in-flight benchmark would silently mislabel
        everything after the change.
        """
        with self._lock:
            if self.is_active():
                raise SupervisorError(
                    "A benchmark is running. Stop it before changing configuration."
                )
            self._config = config

    def is_active(self) -> bool:
        with self._lock:
            self._poll()
            return self._job is not None and self._job.status in ("starting", "running")

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._poll()
            return {
                "active": self.is_active(),
                "job": self._job.to_dict() if self._job else None,
            }

    def orphaned_containers(self) -> list[str]:
        """Harbor task containers that no benchmark owns.

        The distinction `harbor_containers` cannot make. A container's name says
        which task it runs, never whether something is still using it, so during
        a run the raw list *is* the run's own trials -- and anything that treats
        it as leftovers is offering to delete a live benchmark.

        A run in flight owns everything Harbor-shaped that appeared after it
        started, which is the same rule the reap on stop uses. Containers that
        predate it are still genuine leftovers from an earlier run and are still
        reported: they hold multiple GB, and waiting hours for the current run
        to end before saying so helps nobody.

        A run started from the terminal never constructs a Supervisor, so its
        ownership cannot be read from this object. bench.runner leaves a marker
        naming the containers that predated it, which is the same line drawn
        here -- so a benchmark is recognised whichever way it was started.
        """
        with self._lock:
            self._poll()
            running = harbor_containers()
            if self._job is not None and self._job.status in ("starting", "running"):
                return [c for c in running if c in self._job._precontainers]

        marker = active_run(self._config.resolved_runs_dir())
        if marker:
            predating = set(marker.get("precontainers") or [])
            return [c for c in running if c in predating]
        return running

    def log(self, limit: int = 200) -> list[str]:
        with self._lock:
            if limit <= 0 or limit >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-limit:]

    # -- lifecycle --------------------------------------------------------

    def start(
        self,
        *,
        harnesses: Iterable[str] | None = None,
        dataset: str | None = None,
        subset: str | None = None,
        n_tasks: int | None = None,
        tasks: Iterable[str] | None = None,
        agent_timeout_multiplier: float | None = None,
        n_concurrent: int | None = None,
        n_concurrent_agents: int | None = None,
        debug_capture: bool = False,
        allow_hosts: bool = False,
        dry_run: bool = False,
    ) -> Job:
        """Launch a benchmark.

        ``dry_run`` prints the `harbor run` command the settings produce and
        exits without touching Docker or the endpoint. It is what the UI's
        "preview command" does, and it is the only safe way to exercise this
        path in a test -- everything else here starts a real multi-hour run.
        """
        with self._lock:
            if self.is_active():
                raise SupervisorError(
                    "A benchmark is already running. Runs are sequential because "
                    "they share one endpoint; stop the current one first."
                )

            harness_list = [h for h in (harnesses or []) if h]
            argv = [
                sys.executable,
                "-m",
                "bench.runner",
                # Never prompt: there is no terminal to answer from, and a run
                # that blocks on stdin looks identical to one that hung.
                "--no-input",
            ]
            for harness in harness_list:
                argv += ["--harness", harness]
            # Omitted rather than defaulted: the runner already falls back to
            # the catalog's `defaults.dataset`, and passing a value here would
            # pin the UI's idea of the default into every run, so changing the
            # catalog would silently stop affecting runs started from the page.
            if dataset:
                argv += ["--dataset", dataset]
            if subset:
                argv += ["--subset", subset]
            if n_tasks:
                argv += ["--n-tasks", str(n_tasks)]
            for task in tasks or []:
                argv += ["--task", task]
            if agent_timeout_multiplier:
                argv += ["--agent-timeout-multiplier", str(agent_timeout_multiplier)]
            if n_concurrent:
                argv += ["--n-concurrent", str(n_concurrent)]
            if n_concurrent_agents:
                argv += ["--n-concurrent-agents", str(n_concurrent_agents)]
            if allow_hosts:
                argv.append("--allow-hosts")
            if debug_capture:
                argv.append("--debug-capture")
            if dry_run:
                argv.append("--dry-run")

            return self._launch(argv, kind="bench", harnesses=harness_list)

    def start_prepull(self, subset: str | None = None,
                      dataset: str | None = None) -> Job:
        """Cache task images ahead of a run.

        Shares the one-at-a-time rule with benchmarks rather than getting its own
        slot: pulling tens of GB while a benchmark runs competes for the same
        disk and network, and image pulls already happen inside a trial's
        environment-start budget.
        """
        with self._lock:
            if self.is_active():
                raise SupervisorError(
                    "Something is already running. Wait for it or stop it first."
                )
            argv = [sys.executable, "-m", "bench.prepull"]
            if dataset:
                argv += ["--dataset", dataset]
            if subset:
                argv += ["--subset", subset]
            return self._launch(argv, kind="prepull", harnesses=[])

    def _child_endpoint_env(self) -> dict[str, str]:
        """Pin the child to exactly the configuration the server is holding.

        The child would otherwise re-read config.yaml and could disagree with
        the server -- which is not hypothetical, because the UI can edit an
        endpoint and the two would then differ until the file was saved. A run
        labelled with one model and generated by another is the single most
        damaging way this rig can fail, since nothing about the output looks
        wrong.

        The API key travels in the environment, never on the command line: an
        argv is world-readable in the process table.
        """
        endpoint = self._config.endpoint
        env = {
            "HARNESS_ARENA_PROVIDER": endpoint.provider,
            "HARNESS_ARENA_BASE_URL": endpoint.resolved_base_url(),
            "HARNESS_ARENA_RUNS_DIR": str(self._config.resolved_runs_dir()),
        }
        if endpoint.model:
            env["HARNESS_ARENA_MODEL"] = endpoint.model
        if endpoint.label:
            env["HARNESS_ARENA_LABEL"] = endpoint.label

        key = endpoint.resolve_api_key()
        if key:
            # Hand it over under the name the child will look for: its own
            # api_key_env if configured, else the provider's conventional one.
            name = endpoint.api_key_env or endpoint.resolved_provider().default_api_key_env
            if name:
                env[name] = key
                env["HARNESS_ARENA_API_KEY_ENV"] = name
        return env

    def _launch(self, argv: list[str], *, kind: str, harnesses: list[str]) -> Job:
        """Spawn a child, wire up log capture, and record it as the active job."""
        runs_dir = self._config.resolved_runs_dir()
        runs_dir.mkdir(parents=True, exist_ok=True)
        preexisting = {p.name for p in runs_dir.iterdir() if p.is_dir()}

        job_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        log_dir = runs_dir / ".harness-arena" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{kind}-{job_id}.log"

        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing}" if existing else str(ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        env.update(self._child_endpoint_env())

        process = subprocess.Popen(
            argv,
            # The child works out of your workspace, not out of the code. They
            # are the same directory in a checkout; installed from a wheel, ROOT
            # is site-packages, and a child started there would resolve its
            # config and its label cache inside the installed package.
            # PYTHONPATH above still points at the code, which is what lets
            # Harbor import harnesses.* from wherever it was installed.
            cwd=str(WORKSPACE),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # This pipe carries Harbor's console output, which quotes the agent
            # -- so it carries whatever the model emitted. text=True alone
            # decodes with the *locale* encoding, cp1252 on a stock Windows
            # install, where a handful of byte values raise instead of mapping.
            # The pump below already writes UTF-8; without this the read side
            # would be the half that throws, killing the live console and the
            # job log for the run rather than the trial.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **_process_group_kwargs(),
        )

        job = Job(
            id=job_id,
            kind=kind,
            argv=argv,
            harnesses=harnesses,
            started_at=_utc_now(),
            log_path=log_path,
            status="running",
            pid=process.pid,
        )
        job._preexisting = preexisting
        # Snapshot both, so the reap on stop removes only what this run created
        # and leaves any other Docker work on the machine alone.
        job._precontainers = set(harbor_containers())
        job._prenetworks = set(orphaned_networks())

        self._job = job
        self._process = process
        self._buffer.clear()
        self._reader = threading.Thread(
            target=self._pump, args=(process, log_path), daemon=True
        )
        self._reader.start()
        return job

    def stop(self, reason: str = "stopped from the dashboard") -> dict[str, Any]:
        with self._lock:
            self._poll()
            if not self._job or not self._process or not self.is_active():
                raise SupervisorError("Nothing is running.")
            job, process = self._job, self._process

        # Outside the lock: terminating can take the full grace period, and the
        # UI must still be able to poll status while it happens.
        _terminate_tree(process)
        try:
            process.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            _kill_tree(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        # Killing the process does NOT stop its containers: Docker owns those,
        # not us. Harbor tears them down when it exits cleanly, but the
        # interrupt that stops the runner usually kills it before it gets that
        # far -- leaving task containers running, and the agents inside them
        # still generating against the endpoint. To the user, "stopped" then
        # looks like nothing happened at all.
        reaped = self._reap_run_containers(job)

        # The containers are only half of it. Each compose project also created
        # a network, and those survive both the kill and the container removal.
        # They cost nothing visible -- no CPU, no disk -- but each one holds a
        # subnet out of a default pool of 32. Enough stopped runs and Docker has
        # none left, at which point every task in the next run dies at network
        # creation with an error that never mentions subnets. Reaped after the
        # containers, since a network cannot be removed while one is attached.
        networks = reap_networks(
            [n for n in orphaned_networks() if n not in job._prenetworks]
        ).get("removed", [])

        with self._lock:
            job.status = "stopped"
            job.stopped_reason = reason
            job.finished_at = _utc_now()
            job.returncode = process.returncode
            self._mark_manifests_stopped(job, reason)
            result = job.to_dict()
            result["containers_removed"] = reaped
            result["networks_removed"] = networks
            return result

    def _reap_run_containers(self, job: Job) -> list[str]:
        """Remove task containers this run created, and only those.

        Identified by exclusion: anything Harbor-shaped that was not already
        running when the job started. Containers belonging to other work on the
        same machine are therefore left alone.

        Harbor is given a moment to clean up after itself first, because a
        graceful teardown is tidier than a forced removal -- but it is not
        trusted to have managed it.
        """
        mine = [c for c in harbor_containers() if c not in job._precontainers]
        if not mine:
            return []

        deadline = time.monotonic() + CONTAINER_GRACE_S
        while time.monotonic() < deadline:
            time.sleep(1.0)
            mine = [c for c in harbor_containers() if c not in job._precontainers]
            if not mine:
                return []

        return reap_containers(mine).get("removed", [])

    # -- internals --------------------------------------------------------

    def _pump(self, process: subprocess.Popen[str], log_path: Path) -> None:
        """Copy the child's output to disk and to the in-memory buffer.

        Every line goes through scrub(): harness output is captured verbatim, and
        a harness that echoes its own configuration would otherwise write the API
        key into a log file and into the browser.
        """
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as handle:
                for line in process.stdout or ():
                    clean = scrub(strip_ansi(line.rstrip("\n")), self._config)
                    handle.write(clean + "\n")
                    handle.flush()
                    with self._lock:
                        self._buffer.append(clean)
        except (OSError, ValueError):
            pass
        finally:
            process.wait()
            with self._lock:
                if self._job and self._job.status not in ("stopped",):
                    self._job.returncode = process.returncode
                    self._job.finished_at = _utc_now()
                    self._job.status = "finished" if not process.returncode else "failed"

    def _poll(self) -> None:
        """Reconcile job status with the actual process state."""
        if not self._job or not self._process:
            return
        if self._job.status not in ("starting", "running"):
            return
        code = self._process.poll()
        if code is not None and self._reader and not self._reader.is_alive():
            self._job.returncode = code
            self._job.finished_at = self._job.finished_at or _utc_now()
            self._job.status = "finished" if code == 0 else "failed"

    def job_dirs(self) -> list[Path]:
        """Run directories the current job created, newest first.

        The job knows which directories predated it, so "ours" needs no parsing
        of the child's stdout. Empty when nothing is running, or before the
        first harness has written anything.
        """
        with self._lock:
            job = self._job
        if job is None:
            return []
        runs_dir = self._config.resolved_runs_dir()
        if not runs_dir.exists():
            return []
        try:
            mine = [
                p for p in runs_dir.iterdir()
                if p.is_dir() and p.name not in job._preexisting
                and not p.name.startswith(".")
            ]
        except OSError:
            return []
        return sorted(mine, key=lambda p: p.name, reverse=True)

    def _mark_manifests_stopped(self, job: Job, reason: str) -> None:
        """Record the stop in every manifest this run created.

        Without this a killed job never writes ``finished_at`` and renders as
        running forever, so its partial results read as a benchmark in progress
        rather than the baseline they actually are.
        """
        runs_dir = self._config.resolved_runs_dir()
        if not runs_dir.exists():
            return
        stopped_at = _utc_now()
        for job_dir in runs_dir.iterdir():
            if not job_dir.is_dir() or job_dir.name in job._preexisting:
                continue
            manifest_path = job_dir / MANIFEST_NAME
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or manifest.get("stopped_at"):
                    continue
                manifest["stopped_at"] = stopped_at
                manifest["stopped_reason"] = reason
                manifest_path.write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
            except (OSError, json.JSONDecodeError):
                # A manifest we cannot rewrite is not worth failing a stop over;
                # the run is already terminated, which was the point.
                continue


# ---------------------------------------------------------------------------
# Process-tree handling
# ---------------------------------------------------------------------------


def _process_group_kwargs() -> dict[str, Any]:
    """Start the child as its own group so the whole tree can be signalled."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_tree(process: subprocess.Popen[Any]) -> None:
    """Ask the tree to stop, giving Harbor a chance to tear down containers."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            # CTRL_BREAK is the only console signal deliverable to a specific
            # process group on Windows, and Python maps it to KeyboardInterrupt.
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
    except (OSError, ValueError, AttributeError):
        try:
            process.terminate()
        except OSError:
            pass


def _kill_tree(process: subprocess.Popen[Any]) -> None:
    """Hard kill after the grace period. May orphan containers -- see reap()."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=30,
                check=False,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ValueError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


#: Written by bench.runner for the life of a run, inside runs_dir. It exists so
#: a *different* process can tell that a benchmark owns the Harbor containers on
#: this machine: the dashboard knows about runs it started, and this is how it
#: learns about one started from a terminal. A dotfile, and a file rather than a
#: directory, so collect.load_runs -- which iterates directories -- ignores it.
RUN_MARKER_NAME = ".active-run.json"


def run_marker_path(runs_dir: Path) -> Path:
    return Path(runs_dir) / RUN_MARKER_NAME


def _pid_alive(pid: int) -> bool:
    """Whether a process is still running, without depending on psutil.

    Liveness is what makes the marker self-healing: a run killed hard never
    reaches its cleanup, so the file outlives it and every later reader would
    believe a benchmark was running forever.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        # 0x1000 = PROCESS_QUERY_LIMITED_INFORMATION, which a non-elevated
        # process can open on its own children.
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process, but it exists -- which is the question asked.
        return True
    return True


def write_run_marker(runs_dir: Path, harnesses: Iterable[str]) -> Path:
    """Claim the Harbor containers on this machine for the life of this process.

    Records the containers that were already running, so a reader in another
    process can draw the same "appeared after we started" line the supervisor
    draws for its own jobs. Without it the only safe answer for an outside
    observer is "assume everything is owned", which hides real leftovers.
    """
    path = run_marker_path(runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "started_at": _utc_now(),
        "harnesses": list(harnesses),
        "precontainers": harbor_containers(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def clear_run_marker(runs_dir: Path) -> None:
    """Best effort: a marker left behind is handled by the liveness check."""
    try:
        run_marker_path(runs_dir).unlink()
    except (OSError, FileNotFoundError):
        pass


def active_run(runs_dir: Path) -> dict[str, Any] | None:
    """The marker for a run in flight anywhere on this machine, or None.

    None also covers a marker whose process is gone, which is the common case
    after a hard kill -- stale is indistinguishable from absent to every caller,
    so none of them has to think about it.
    """
    try:
        payload = json.loads(run_marker_path(runs_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not _pid_alive(int(payload.get("pid") or 0)):
        return None
    return payload


def harbor_containers() -> list[str]:
    """Every running Harbor task container, owned or not.

    Deliberately not called "orphaned": a name cannot tell you whether a
    benchmark owns it. During a run these *are* the run's own trials, so
    treating this list as leftovers is how a caller offers to delete a live
    benchmark. Ownership is decided by `Supervisor.orphaned_containers`, which
    is the only thing that knows whether a run is in flight and what predated
    it.
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [
        name.strip()
        for name in result.stdout.splitlines()
        if "__env-main" in name or name.strip().startswith("harbor")
    ]


#: Compose names every task network "<project>_default", and Harbor's project
#: names end in "__env". Matching on that rather than on "everything unused"
#: keeps this from touching Docker networks that belong to the user's own work.
_TASK_NETWORK_SUFFIX = "__env_default"


def orphaned_networks() -> list[str]:
    """Task networks left behind with nothing attached to them.

    A compose project creates a *network* as well as containers, and killing the
    run removes neither. Networks are easy to miss because they cost no CPU, no
    memory and no disk -- but each one holds a subnet, and Docker's default pool
    only carves out 32 of them. Once they are gone every `compose up` fails at
    network creation, before the image or the model is touched, so all 89 tasks
    error identically within seconds and none of it mentions networks.
    """
    try:
        result = subprocess.run(
            ["docker", "network", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []

    idle = []
    for name in (line.strip() for line in result.stdout.splitlines()):
        if not name.endswith(_TASK_NETWORK_SUFFIX):
            continue
        # Never remove one a container is still using: a running trial would
        # lose its network out from under it.
        if _network_is_idle(name):
            idle.append(name)
    return idle


def _network_is_idle(name: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "network", "inspect", name, "--format", "{{len .Containers}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "0"


def reap_networks(names: Iterable[str]) -> dict[str, Any]:
    """Remove the named task networks, freeing their subnets."""
    removed, failed = [], []
    for name in names:
        try:
            result = subprocess.run(
                ["docker", "network", "rm", name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            (removed if result.returncode == 0 else failed).append(name)
        except (OSError, subprocess.SubprocessError):
            failed.append(name)
    return {"removed": removed, "failed": failed}


def network_pool_pressure() -> dict[str, Any]:
    """How close Docker is to running out of subnets to hand out.

    Reported before a run because the failure it predicts is unreadable: every
    task dies with a compose error that never says the word "subnet".
    """
    try:
        result = subprocess.run(
            ["docker", "network", "ls", "--format", "{{.Driver}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    # Only bridge networks draw from the address pool; host and none do not.
    n_bridge = sum(1 for line in result.stdout.splitlines() if line.strip() == "bridge")
    idle = orphaned_networks()
    return {
        "n_bridge_networks": n_bridge,
        "capacity": DEFAULT_POOL_CAPACITY,
        "reclaimable": len(idle),
        # Two trials in flight need two more networks, so "one spare" is
        # already a failing run.
        "tight": n_bridge >= DEFAULT_POOL_CAPACITY - 2,
    }


#: Docker's built-in default-address-pools carve two private ranges into /16 and
#: /20 blocks: 16 + 16 allocatable networks. A host with custom pools configured
#: has more, so this is a floor used only to warn, never to block.
DEFAULT_POOL_CAPACITY = 32


def reap_containers(names: Iterable[str]) -> dict[str, Any]:
    """Force-remove the named containers.

    A hard kill leaves them, and a multi-GB task image container holding disk is
    the most expensive form of litter this rig produces.
    """
    removed, failed = [], []
    for name in names:
        try:
            result = subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            (removed if result.returncode == 0 else failed).append(name)
        except (OSError, subprocess.SubprocessError):
            failed.append(name)
    return {"removed": removed, "failed": failed}


def long_path(path: Path) -> str:
    """A path Windows will still open past 260 characters.

    Windows refuses a plain path longer than MAX_PATH unless long paths are
    enabled machine-wide, and it refuses it as ``WinError 3, the system cannot
    find the path specified`` -- which reads as "already gone" rather than as
    "too long to name", so the caller retries and gets the same answer forever.
    The ``\\\\?\\`` prefix bypasses the limit and needs no registry change; it
    requires an absolute path with no forward slashes, which ``resolve()``
    guarantees.

    This is not hypothetical and not only about deletion: a run directory
    already nests a job name, a trial name and Harbor's own per-task
    directories, and `harness-arena doctor` warns that tau3-bench reaches 287
    characters on its own. Anything writing inside a trial directory can push a
    path over the line, and one unreachable file is enough to strand the whole
    run on disk.

    A no-op off Windows, where nothing imposes the limit.
    """
    text = str(path)
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    return "\\\\?\\" + text


def delete_run(runs_dir: Path, run_id: str) -> dict[str, Any]:
    """Delete one run directory, refusing anything that is not one.

    ``run_id`` arrives from the browser, so it is treated as hostile: the
    resolved path must be a direct child of runs_dir. Without that check a
    crafted id traverses out of the tree and this becomes an arbitrary-delete
    endpoint on the host.
    """
    import shutil

    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        raise SupervisorError(f"Invalid run id: {run_id!r}")

    runs_root = runs_dir.resolve()
    target = (runs_root / run_id).resolve()
    if target.parent != runs_root or not target.is_dir():
        raise SupervisorError(f"No such run: {run_id!r}")

    # The walk takes the same treatment as the delete. A file too long to name
    # answers `is_file()` with False rather than raising, so a plain walk does
    # not fail here -- it quietly under-reports the space about to be freed,
    # and then rmtree raises on the file the walk pretended was not there.
    size = 0
    for dirpath, _dirnames, filenames in os.walk(long_path(target)):
        for name in filenames:
            try:
                size += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    shutil.rmtree(long_path(target))
    return {"deleted": run_id, "freed_bytes": size}
