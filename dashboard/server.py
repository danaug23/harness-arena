"""Live results server and control plane for harness-arena.

Two jobs. It rescans ``runs/`` on each request (behind a short cache) so a
benchmark in flight shows up as it progresses -- Harbor writes each trial's
result.json the moment that trial ends, so the page just needs to keep asking.
And it exposes the write endpoints that let the UI configure, start, stop and
clean up runs without anyone touching a terminal.

Standard library only: no build step, nothing to install, and it keeps working
after a dependency upgrade elsewhere breaks something.

Security
--------
The moment this server could start processes and hold an API key, "it only
listens on localhost" stopped being sufficient. A page in your browser can send
requests to 127.0.0.1, and DNS rebinding can make a hostile domain resolve
there. So every mutating request must clear four gates:

1. **A per-process token** in ``X-Harness-Arena-Token``. The token is minted at
   startup and injected into the page as it is served. A *custom* header is the
   point: cross-origin JavaScript cannot set one without a CORS preflight, and
   this server answers no preflight and sends no CORS headers.
2. **Host header validation**, which is what actually defeats DNS rebinding --
   the attacker controls DNS, not the Host string the browser sends.
3. **Origin rejection**: browsers attach Origin to cross-origin writes, so any
   Origin that is not ours is refused outright.
4. **JSON bodies only.** Form encoding is the content type that can be sent
   cross-origin without a preflight, so it is not accepted.

None of this makes the server safe to expose. It binds loopback by default and
should stay there or sit behind an authenticating proxy.
"""

from __future__ import annotations

import argparse
import json
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from bench import ROOT, subset_names
from bench import registry as registry_mod
from bench.activity import read_activity
from bench.collect import build_index
from bench.config import (
    PROVIDERS,
    REDACTED,
    Config,
    ConfigError,
    DashboardConfig,
    EndpointConfig,
    config_path,
    load,
)
from bench.probe import describe as describe_model
from bench.probe import (
    effective_label,
    list_models,
    measure_speed,
    probe,
    remember_label,
    suggest_label,
)
from bench.supervisor import (
    Supervisor,
    SupervisorError,
    active_run,
    delete_run,
    network_pool_pressure,
    orphaned_networks,
    reap_containers,
    reap_networks,
)

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"
CACHE_TTL_S = 2.0
TOKEN_HEADER = "X-Harness-Arena-Token"

#: A request body is a config blob or a run spec. Anything larger is a mistake
#: or an attempt to exhaust memory.
MAX_BODY_BYTES = 256 * 1024

LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}

#: Sentinel for "remove this value", on fields where an empty submission means
#: "leave it as it is". Not a URL and not a key, so it cannot collide with one.
CLEAR = "clear"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class _Cache:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._lock = threading.Lock()
        self._value: dict[str, Any] | None = None
        self._stamp = 0.0

    def get(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if self._value is None or now - self._stamp > CACHE_TTL_S:
                self._value = build_index(self.runs_dir)
                self._stamp = now
            return self._value

    def invalidate(self) -> None:
        with self._lock:
            self._value = None


class App:
    """Server state: configuration, the supervisor, and the results cache."""

    def __init__(
        self,
        config: Config,
        runs_dir: Path,
        token: str,
        *,
        bind_host: str = "127.0.0.1",
        read_only: bool = False,
    ) -> None:
        self.config = config
        self.runs_dir = runs_dir
        self.token = token
        self.bind_host = bind_host
        # Read-only is a hard gate checked before the token, because an empty
        # token would otherwise compare equal to an absent header and turn
        # "no control plane" into "no authentication".
        self.read_only = read_only
        self.cache = _Cache(runs_dir)
        self.supervisor = Supervisor(config)
        self.lock = threading.RLock()

    # -- state ------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Everything the UI needs to render its non-results tabs."""
        try:
            catalog = registry_mod.load()
        except (OSError, registry_mod.RegistryError) as exc:
            catalog = {"harnesses": {}, "defaults": {}, "error": str(exc)}

        subsets = subset_names()

        return {
            "read_only": self.read_only,
            # index.html is re-read from disk on every request, but this process
            # is not: a dashboard left running across an upgrade serves new
            # controls backed by old code, and a control that silently does
            # nothing is worse than one that is missing. The page checks this
            # list and disables anything the running server cannot honour. Add
            # a name here in the same commit that adds the feature.
            # "network_reaping" covered two separate things: that /api/containers
            # reports network counts, and that the reap endpoint removes them.
            # A page cannot gate on one name without gating the other, and
            # gating the reap button wholesale would have withdrawn *container*
            # reaping -- which every server has always supported -- from anyone
            # running an older process. Split so each control checks the thing
            # it actually needs.
            "capabilities": [
                "debug_capture",
                "orphan_counts",     # GET /api/containers reports networks and pool
                "network_reaping",   # POST /api/containers/reap removes them too
                "transport_faults",
            ],
            # Which checkout is actually running. An editable install pins an
            # absolute path, so a second clone launched by the console script
            # still serves the first one's runs -- and nothing on screen would
            # say so without this.
            "code_root": str(ROOT),
            # redacted(): never hand a resolved key to the browser.
            "config": self.config.redacted(),
            "config_path": str(config_path()),
            "config_exists": config_path().exists(),
            "providers": [
                {
                    "id": p.id,
                    "label": p.label,
                    "default_base_url": p.default_base_url,
                    "default_api_key_env": p.default_api_key_env,
                    "requires_api_key": p.requires_api_key,
                    "model_is_discoverable": p.model_is_discoverable,
                    "default_agent_concurrency": p.default_agent_concurrency,
                }
                for p in PROVIDERS.values()
            ],
            "api_key_set": bool(self.config.endpoint.resolve_api_key()),
            "harnesses": catalog.get("harnesses", {}),
            "defaults": catalog.get("defaults", {}),
            "editable_defaults": sorted(registry_mod.EDITABLE_DEFAULTS),
            "subsets": subsets,
            "supervisor": self.supervisor.status(),
            "runs_dir": str(self.runs_dir),
        }

    # -- config -----------------------------------------------------------

    def save_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            endpoint = self._endpoint_from(payload)
            # An explicitly empty key means "remove it", which _endpoint_from
            # cannot distinguish from "unchanged" on its own.
            if (payload.get("endpoint") or {}).get("api_key") == "":
                endpoint.api_key = ""

            dashboard_in = payload.get("dashboard") or {}
            dashboard = DashboardConfig(
                host=str(dashboard_in.get("host", self.config.dashboard.host)),
                port=int(dashboard_in.get("port", self.config.dashboard.port)),
                open_browser=bool(
                    dashboard_in.get("open_browser", self.config.dashboard.open_browser)
                ),
            )
            if not 1 <= dashboard.port <= 65535:
                raise ApiError("Port must be between 1 and 65535.")

            config = Config(
                endpoint=endpoint,
                dashboard=dashboard,
                runs_dir=str(payload.get("runs_dir", self.config.runs_dir) or "runs"),
                record_hostname=bool(
                    payload.get("record_hostname", self.config.record_hostname)
                ),
            )
            try:
                config.endpoint.validate()
            except ConfigError as exc:
                raise ApiError(str(exc)) from exc

            try:
                self.supervisor.set_config(config)
            except SupervisorError as exc:
                raise ApiError(str(exc), 409) from exc

            written = config.save()
            self.config = config
            self.cache.invalidate()

            # Pin the chosen name to the weights currently served, so it also
            # survives clearing the override and reappears for the same weights
            # later. Best-effort: the label is already saved either way.
            fingerprint = remember_label(endpoint, endpoint.label)

            return {
                "saved": str(written),
                "config": config.redacted(),
                "label_pinned_to": fingerprint,
            }

    def _endpoint_from(self, payload: dict[str, Any]) -> EndpointConfig:
        """Build an endpoint from submitted fields, falling back to the saved one.

        Used by every Setup action so the buttons answer "will *this* work"
        before anything is committed to disk.
        """
        endpoint_in = payload.get("endpoint") or {}
        current = self.config.endpoint

        supplied_key = endpoint_in.get("api_key")
        api_key = current.api_key
        if supplied_key not in (None, REDACTED):
            # The UI renders a stored key as the redaction marker; echoing that
            # back means "unchanged", not "set my key to '***'".
            api_key = supplied_key

        # Same rule for the window: an empty box means "leave it alone", while
        # an explicit 0 means "ask the server" and is a real choice.
        supplied_window = endpoint_in.get("context_window")
        context_window = current.context_window
        if supplied_window not in (None, ""):
            try:
                context_window = max(0, int(supplied_window))
            except (TypeError, ValueError):
                raise ApiError("Context window must be a whole number.") from None

        # The endpoint URL gets the same treatment, for the same reason the key
        # does: the form posts every field on every save, so a box left empty
        # would otherwise overwrite a working URL with "". That is not a benign
        # loss -- an empty base_url resolves to the *provider's default* host,
        # so the next run silently goes somewhere the user never named and
        # fails against an address that appears nowhere in their setup.
        # Empty means unchanged; the literal word "clear" removes it and
        # restores the provider default deliberately.
        supplied_base_url = endpoint_in.get("base_url")
        base_url = current.base_url
        if supplied_base_url == CLEAR:
            base_url = ""
        elif supplied_base_url not in (None, ""):
            base_url = str(supplied_base_url)

        endpoint = EndpointConfig(
            provider=str(endpoint_in.get("provider", current.provider)),
            base_url=base_url,
            api_key_env=str(endpoint_in.get("api_key_env", current.api_key_env) or ""),
            api_key=api_key or "",
            model=str(endpoint_in.get("model", current.model) or ""),
            label=str(endpoint_in.get("label", current.label) or ""),
            context_window=context_window,
        )
        if endpoint.provider not in PROVIDERS:
            raise ApiError(f"Unknown provider {endpoint.provider!r}.")
        return endpoint

    def available_models(self, payload: dict[str, Any]) -> dict[str, Any]:
        """List what the endpoint serves, so the UI can offer a choice.

        Separate from the connection test because a hosted provider cannot be
        probed until a model has been chosen -- and you cannot choose from a
        list you were never able to fetch.
        """
        endpoint = self._endpoint_from(payload)
        try:
            return list_models(endpoint)
        except RuntimeError as exc:
            raise ApiError(str(exc), 502) from exc

    def test_endpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Probe the endpoint, optionally timing it.

        Uses the *submitted* settings rather than the saved ones, so the button
        answers "will this work" before anyone commits it to disk.
        """
        endpoint = self._endpoint_from(payload)
        try:
            identity = probe(endpoint)
        except (ConfigError, RuntimeError) as exc:
            raise ApiError(str(exc), 502) from exc

        result = {
            "ok": True,
            "identity": identity.to_dict(),
            "description": describe_model(identity),
            # What the label would be if none is configured -- shown as the
            # placeholder so the field says what you get by leaving it empty.
            "suggested_label": suggest_label(identity),
            # What a run started right now would actually be labelled: an
            # override wins, then a name pinned to these weights, then the
            # derived one. "Empty" stops meaning "derived" once a pin exists.
            "effective_label": effective_label(endpoint, identity),
        }
        # Offer the catalog alongside the result so choosing a model never needs
        # a second round trip. A listing failure must not fail the test itself.
        try:
            result.update(list_models(endpoint))
        except RuntimeError:
            result.setdefault("models", [])
        if payload.get("speed"):
            try:
                result["speed"] = measure_speed(endpoint)
            except RuntimeError as exc:
                result["speed_error"] = str(exc)
        return result

    def doctor(self) -> dict[str, Any]:
        """The same checks as `harness-arena doctor`, as structured data."""
        import shutil
        import subprocess

        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, detail: str = "") -> None:
            checks.append({"name": name, "ok": bool(ok), "detail": detail})

        harbor = shutil.which("harbor")
        add("harbor on PATH", bool(harbor), harbor or "pip install -e .")
        if harbor:
            try:
                out = subprocess.run(
                    [harbor, "--version"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
                add("harbor runs", out.returncode == 0, out.stdout.strip())
            except (OSError, subprocess.SubprocessError) as exc:
                add("harbor runs", False, str(exc))

        docker = shutil.which("docker")
        add("docker on PATH", bool(docker), docker or "install Docker")
        if docker:
            try:
                out = subprocess.run(
                    [docker, "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                )
                add(
                    "docker daemon running",
                    out.returncode == 0,
                    out.stdout.strip() or "start Docker Desktop / the docker service",
                )
            except (OSError, subprocess.SubprocessError) as exc:
                add("docker daemon running", False, str(exc))

        try:
            anchor = self.runs_dir if self.runs_dir.exists() else self.runs_dir.parent
            free_gb = shutil.disk_usage(anchor).free / 1e9
            add(
                "disk space for task images",
                free_gb >= 100,
                f"{free_gb:.0f} GB free (~100 GB recommended)",
            )
        except OSError as exc:
            add("disk space", False, str(exc))

        try:
            identity = probe(self.config.endpoint)
            add("endpoint answers", True, identity.served_id)
        except (ConfigError, RuntimeError) as exc:
            add("endpoint answers", False, str(exc))

        return {"checks": checks, "ok": all(c["ok"] for c in checks)}

    # -- runs -------------------------------------------------------------

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        known = set(registry_mod.load().get("harnesses", {}))
        harnesses = payload.get("harnesses") or []
        if not isinstance(harnesses, list):
            raise ApiError("`harnesses` must be a list.")
        unknown = [h for h in harnesses if h not in known]
        if unknown:
            raise ApiError(f"Unknown harness(es): {', '.join(map(str, unknown))}.")

        subset = payload.get("subset") or None
        if subset is not None and not isinstance(subset, str):
            raise ApiError("`subset` must be a string.")
        if subset:
            # Reaches a file path in the runner; keep it to a bare name.
            if "/" in subset or "\\" in subset or subset.startswith("."):
                raise ApiError(f"Invalid subset name {subset!r}.")

        def _num(name: str, caster, low, high):
            raw = payload.get(name)
            if raw in (None, ""):
                return None
            try:
                value = caster(raw)
            except (TypeError, ValueError):
                raise ApiError(f"`{name}` must be a number.") from None
            if not low <= value <= high:
                raise ApiError(f"`{name}` must be between {low} and {high}.")
            return value

        try:
            job = self.supervisor.start(
                harnesses=harnesses,
                subset=subset,
                n_tasks=_num("n_tasks", int, 1, 1000),
                tasks=[str(t) for t in (payload.get("tasks") or [])],
                agent_timeout_multiplier=_num(
                    "agent_timeout_multiplier", float, 0.1, 100.0
                ),
                n_concurrent=_num("n_concurrent", int, 1, 32),
                n_concurrent_agents=_num("n_concurrent_agents", int, 1, 32),
                allow_hosts=bool(payload.get("allow_hosts")),
                debug_capture=bool(payload.get("debug_capture")),
                dry_run=bool(payload.get("dry_run")),
            )
        except SupervisorError as exc:
            raise ApiError(str(exc), 409) from exc
        self.cache.invalidate()
        return job.to_dict()

    def stop_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        reason = str(payload.get("reason") or "stopped from the dashboard")[:200]
        try:
            result = self.supervisor.stop(reason)
        except SupervisorError as exc:
            raise ApiError(str(exc), 409) from exc
        self.cache.invalidate()
        return result

    def start_prepull(self, payload: dict[str, Any]) -> dict[str, Any]:
        subset = payload.get("subset") or None
        if subset and ("/" in subset or "\\" in subset or subset.startswith(".")):
            raise ApiError(f"Invalid subset name {subset!r}.")
        try:
            return self.supervisor.start_prepull(subset).to_dict()
        except SupervisorError as exc:
            raise ApiError(str(exc), 409) from exc

    def remove_run(self, run_id: str) -> dict[str, Any]:
        active = self.supervisor.status().get("job") or {}
        if active.get("active"):
            raise ApiError(
                "A run is in progress. Stop it before deleting run directories.", 409
            )
        try:
            result = delete_run(self.runs_dir, run_id)
        except SupervisorError as exc:
            raise ApiError(str(exc), 404) from exc
        self.cache.invalidate()
        return result

    def export(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name") or "harness-arena-snapshot.html")
        if "/" in name or "\\" in name or not name.endswith(".html"):
            raise ApiError("Snapshot name must be a bare *.html filename.")
        out = export_snapshot(self.runs_dir, ROOT / name)
        return {"wrote": str(out), "bytes": out.stat().st_size}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def export_snapshot(runs_dir: Path, out: Path) -> Path:
    """Write a standalone copy with the current results inlined.

    The live page fetches /api/results; the snapshot pre-seeds the same payload
    so the file works with no server, offline, on any machine. It is explicitly
    *not* given a token, so a shared snapshot has no control plane in it.
    """
    index = build_index(runs_dir)
    html = INDEX_HTML.read_text(encoding="utf-8")
    seed = (
        "<script>window.__HARNESS_ARENA_SNAPSHOT__ = "
        + json.dumps(index)
        + ";window.__HARNESS_ARENA_READONLY__ = true;</script>"
    )
    marker = "</head>"
    html = html.replace(marker, f"{seed}\n{marker}", 1)
    out.write_text(html, encoding="utf-8")
    return out


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "harness-arena"
        protocol_version = "HTTP/1.1"

        #: Whether this request's body has been read. Class-level so the paths
        #: that answer without dispatching -- OPTIONS, and anything that fails
        #: before _dispatch runs -- still find it defined.
        _body_consumed = False

        # -- plumbing --------------------------------------------------

        def _drain_request_body(self) -> None:
            """Consume an unread request body before answering.

            A rejected write -- bad token, wrong Origin, unknown Host -- is
            refused before the body is ever read, and on a keep-alive
            connection that body is still sitting in the socket. Replying and
            moving on leaves it there to be parsed as the next request, and
            closing instead makes the peer's still-in-flight write fail: on
            Windows the client sees WSAECONNABORTED rather than the 403 that
            explains what it did wrong.

            An oversized body is not drained -- reading megabytes only to
            reject them is the denial-of-service this would invite -- so the
            connection is closed instead, which is the honest signal.
            """
            if self._body_consumed:
                return
            self._body_consumed = True
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self.close_connection = True
                return
            if length <= 0:
                return
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                return
            try:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 64 * 1024))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except OSError:
                self.close_connection = True

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self._drain_request_body()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            # The page is entirely self-contained; forbidding outside origins
            # limits what an injected string could reach.
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; base-uri 'none'; form-action 'none'",
            )
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(
                status,
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _host_ok(self) -> bool:
            """Reject a Host we did not expect -- this is the anti-rebinding gate."""
            host = (self.headers.get("Host") or "").strip()
            if not host:
                return False
            name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
            return name in LOOPBACK or name == app.bind_host

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if not origin:
                return True  # non-browser client; the token still gates writes
            try:
                parsed = urlparse(origin)
            except ValueError:
                return False
            return parsed.hostname in LOOPBACK or parsed.hostname == app.bind_host

        def _read_json(self) -> dict[str, Any]:
            self._body_consumed = True
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            if ctype != "application/json":
                raise ApiError("Content-Type must be application/json.", 415)
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ApiError("Bad Content-Length.", 400) from None
            if length > MAX_BODY_BYTES:
                raise ApiError("Request body too large.", 413)
            if length <= 0:
                return {}
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ApiError(f"Invalid JSON: {exc}", 400) from exc
            if not isinstance(payload, dict):
                raise ApiError("Request body must be a JSON object.", 400)
            return payload

        def _authorize_write(self) -> None:
            if app.read_only or not app.token:
                raise ApiError(
                    "This dashboard is read-only; the control plane is disabled.", 403
                )
            if not self._origin_ok():
                raise ApiError("Cross-origin request refused.", 403)
            token = self.headers.get(TOKEN_HEADER) or ""
            if not secrets.compare_digest(token, app.token):
                raise ApiError(
                    "Missing or invalid session token. Reload the dashboard.", 403
                )

        # -- routing ---------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def do_DELETE(self) -> None:  # noqa: N802
            self._dispatch("DELETE")

        def do_OPTIONS(self) -> None:  # noqa: N802
            # No CORS headers, deliberately: an unanswered preflight is what
            # stops cross-origin JavaScript from sending the token header.
            self._send(405, b"", "text/plain; charset=utf-8")

        def _dispatch(self, method: str) -> None:
            self._body_consumed = False
            try:
                if not self._host_ok():
                    raise ApiError("Unrecognized Host header.", 403)
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                query = parse_qs(parsed.query)

                if method != "GET":
                    self._authorize_write()

                handler = ROUTES.get((method, path))
                if handler is not None:
                    payload = self._read_json() if method in ("POST", "DELETE") else {}
                    self._json(200, handler(app, payload, query))
                    return

                if method == "DELETE" and path.startswith("/api/runs/"):
                    run_id = unquote(path[len("/api/runs/") :])
                    self._json(200, app.remove_run(run_id))
                    return
                if method == "DELETE" and path.startswith("/api/harness/"):
                    name = unquote(path[len("/api/harness/") :])
                    try:
                        registry_mod.delete_harness(name)
                    except registry_mod.RegistryError as exc:
                        raise ApiError(str(exc), 400) from exc
                    self._json(200, {"deleted": name})
                    return

                if method == "GET" and path in ("/", "/index.html"):
                    self._serve_index()
                    return

                self._json(404, {"error": "not found"})
            except ApiError as exc:
                self._json(exc.status, {"error": exc.message})
            except Exception as exc:  # noqa: BLE001 - never take the server down
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def _serve_index(self) -> None:
            if not INDEX_HTML.exists():
                self._send(500, b"index.html missing", "text/plain; charset=utf-8")
                return
            html = INDEX_HTML.read_text(encoding="utf-8")
            # Hand the page its token. Same-origin script can read it; a
            # cross-origin page cannot read this response at all.
            seed = (
                "<script>window.__HARNESS_ARENA_TOKEN__ = "
                + json.dumps(app.token)
                + ";</script>"
            )
            html = html.replace("</head>", f"{seed}\n</head>", 1)
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

        def log_message(self, fmt: str, *args: Any) -> None:
            # The default handler spams stderr with one line per poll.
            return

    return Handler


# --- route table -----------------------------------------------------------

Route = Callable[[App, dict[str, Any], dict[str, list[str]]], Any]


def _int_param(query: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(query.get(name, [default])[0])
    except (TypeError, ValueError):
        return default


ROUTES: dict[tuple[str, str], Route] = {
    ("GET", "/api/results"): lambda app, _p, _q: app.cache.get(),
    ("GET", "/api/activity"): lambda app, _p, _q: read_activity(app.runs_dir),
    ("GET", "/api/health"): lambda _app, _p, _q: {"ok": True},
    ("GET", "/api/state"): lambda app, _p, _q: app.state(),
    ("GET", "/api/doctor"): lambda app, _p, _q: app.doctor(),
    ("GET", "/api/run/log"): lambda app, _p, q: {
        "lines": app.supervisor.log(_int_param(q, "limit", 400)),
        "supervisor": app.supervisor.status(),
    },
    ("GET", "/api/containers"): lambda app, _p, _q: {
        # Supervisor-aware: the module-level list cannot tell a leftover from a
        # trial of the run currently in flight.
        "orphaned": app.supervisor.orphaned_containers(),
        # Networks litter the same way containers do, and are easier to miss:
        # they cost nothing visible until the subnet pool runs dry.
        "orphaned_networks": orphaned_networks(),
        "network_pool": network_pool_pressure(),
        # These lists are "every Harbor-shaped container", which during a run
        # is the run's own trials. Orphanhood cannot be read off a name, so it
        # is reported rather than guessed: the caller has to know a benchmark
        # owns them before offering to remove them.
        # Not just this process's run: a benchmark started from a terminal
        # leaves a marker, and its trials are just as live.
        "run_active": bool(
            app.supervisor.is_active() or active_run(app.config.resolved_runs_dir())
        ),
    },
    ("POST", "/api/config"): lambda app, p, _q: app.save_config(p),
    ("POST", "/api/config/test"): lambda app, p, _q: app.test_endpoint(p),
    ("POST", "/api/config/models"): lambda app, p, _q: app.available_models(p),
    ("POST", "/api/run"): lambda app, p, _q: app.start_run(p),
    ("POST", "/api/run/stop"): lambda app, p, _q: app.stop_run(p),
    ("POST", "/api/prepull"): lambda app, p, _q: app.start_prepull(p),
    ("POST", "/api/export"): lambda app, p, _q: app.export(p),
    ("POST", "/api/containers/reap"): lambda app, p, _q: {
        **reap_containers(p.get("names") or app.supervisor.orphaned_containers()),
        # Containers first: a network cannot be removed while one is attached.
        "networks": reap_networks(orphaned_networks()),
    },
    ("POST", "/api/harness"): lambda _app, p, _q: _upsert_harness(p),
    ("POST", "/api/defaults"): lambda _app, p, _q: _update_defaults(p),
}


def _upsert_harness(payload: dict[str, Any]) -> dict[str, Any]:
    harness_id = str(payload.get("id") or "").strip()
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        raise ApiError("`spec` must be an object.")

    # The editor renders six fields; the catalog holds more. A save posts only
    # what the form knows, and an upsert *replaces* the entry -- so editing a
    # label through the UI silently deleted anything the form does not render.
    # host_env and min_context_window are both load-bearing: dropping the
    # latter from hermes restores a run that fails identically 89 times.
    #
    # Keyed off the stored entry rather than a list of field names, so a field
    # added later is preserved without anyone remembering to come back here.
    # A field the form *does* render still overrides, and delete_harness is
    # still how you remove one.
    try:
        stored = registry_mod.load()["harnesses"].get(harness_id) or {}
    except (OSError, registry_mod.RegistryError):
        stored = {}
    merged = {**stored, **spec}

    try:
        saved = registry_mod.upsert_harness(harness_id, merged)
    except registry_mod.RegistryError as exc:
        raise ApiError(str(exc)) from exc
    return {"id": harness_id, "spec": saved}


def _update_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"defaults": registry_mod.update_defaults(payload)}
    except registry_mod.RegistryError as exc:
        raise ApiError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Write a self-contained snapshot HTML and exit",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Serve results without the control plane (no start/stop/config)",
    )
    args = parser.parse_args(argv)

    config = load()
    runs_dir = args.runs_dir or config.resolved_runs_dir()

    if args.export:
        out = export_snapshot(runs_dir, args.export)
        print(f"wrote {out}")
        return 0

    host = args.host or config.dashboard.host
    port = args.port or config.dashboard.port
    open_browser = config.dashboard.open_browser and not args.no_browser

    # A fresh token per process: restarting the server invalidates every page
    # left open, which is the correct blast radius for a local control plane.
    token = "" if args.read_only else secrets.token_urlsafe(32)

    app = App(config, runs_dir, token, bind_host=host, read_only=args.read_only)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}/"
    print(f"harness-arena dashboard: {url}")
    print(f"code:     {ROOT}")
    print(f"config:   {config_path() if config_path().exists() else '(defaults)'}")
    print(f"watching: {runs_dir}")
    # The trap this prevents: an editable install pins an absolute path, so
    # launching from a second clone still serves the first one's runs.
    cwd = Path.cwd().resolve()
    if cwd != ROOT and (cwd / "bench" / "__init__.py").exists():
        print(
            f"  [!] You launched from {cwd}, but the running code is {ROOT}.\n"
            f"      `cd` does not change import resolution. Reinstall here with "
            f"`python -m pip install -e .`,\n"
            f"      or run this checkout with `python -m bench dash`."
        )
    if args.read_only:
        print("read-only: the control plane is disabled")
    if host not in LOOPBACK:
        print(
            f"  [!] Listening on {host}, not loopback. This server can start and "
            f"stop benchmarks and read your run output.\n"
            f"      Put it behind an authenticating proxy or bind 127.0.0.1."
        )
    print("Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
        if app.supervisor.is_active():
            print("  terminating the running benchmark")
            try:
                app.supervisor.stop("dashboard shut down")
            except SupervisorError:
                pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
