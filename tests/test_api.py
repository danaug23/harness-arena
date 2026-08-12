"""Guard the control plane: authentication, input validation, and key redaction.

The dashboard can start processes, delete directories and hold an API key, and
it listens on a port any page in your browser can reach. That combination is
why these are tests rather than review comments -- every check here is a gate
that fails open silently if it regresses. Nothing would look broken; the server
would simply also do what an attacker asked.

Needs no model server and no Docker: everything exercised here is refused,
validated, or redacted before it would reach either.

    python tests/test_api.py
"""

import json
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import REGISTRY_PATH  # noqa: E402
from bench import registry as registry_mod  # noqa: E402
from bench.config import REDACTED, Config, EndpointConfig  # noqa: E402
from dashboard import server as server_mod  # noqa: E402

TOKEN = "test-token-abc"
KEY = "sk-should-never-be-served-0123456789"

failures: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<52} got={got!r} want={want!r}")
    if not ok:
        failures.append(label)


class Client:
    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}"

    def __call__(
        self, method, path, body=None, *, token=TOKEN, host=None,
        ctype="application/json", origin=None,
    ):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None and ctype:
            req.add_header("Content-Type", ctype)
        if token is not None:
            req.add_header(server_mod.TOKEN_HEADER, token)
        if host:
            req.add_header("Host", host)
        if origin:
            req.add_header("Origin", origin)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status, json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read().decode())
            except (ValueError, OSError):
                return exc.code, {}


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    runs = tmp / "runs"
    runs.mkdir()

    # Work against a copy of the catalog: these tests write to it, and the real
    # registry.yaml is a source file.
    registry_copy = tmp / "registry.yaml"
    shutil.copy2(REGISTRY_PATH, registry_copy)
    registry_mod.REGISTRY_PATH = registry_copy

    config = Config(
        endpoint=EndpointConfig(base_url="http://127.0.0.1:9/v1"), runs_dir=str(runs)
    )
    app = server_mod.App(config, runs, TOKEN, bind_host="127.0.0.1")

    server = ThreadingHTTPServer(("127.0.0.1", 0), server_mod.make_handler(app))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.3)
    call = Client(port)

    try:
        # -- authentication -------------------------------------------------
        # A GET is readable without a token; anything that writes is not.
        check("GET needs no token", call("GET", "/api/health", token=None)[0], 200)
        check("POST without a token", call("POST", "/api/run", {}, token=None)[0], 403)
        check("POST with a wrong token", call("POST", "/api/run", {}, token="no")[0], 403)

        # DNS rebinding: the attacker controls name resolution, not the Host
        # string the browser sends, so this is the check that actually holds.
        check(
            "foreign Host header",
            call("GET", "/api/health", host="attacker.example")[0],
            403,
        )
        check(
            "cross-origin write",
            call("POST", "/api/export", {}, origin="https://attacker.example")[0],
            403,
        )
        # Form encoding is the one content type sendable cross-origin without a
        # preflight, so it must not be accepted.
        check(
            "form-encoded body",
            call("POST", "/api/export", {}, ctype="application/x-www-form-urlencoded")[0],
            415,
        )
        check("preflight is unanswered", call("OPTIONS", "/api/run")[0], 405)

        # -- read-only mode -------------------------------------------------
        # An empty token must not compare equal to an absent header.
        ro_app = server_mod.App(config, runs, "", bind_host="127.0.0.1", read_only=True)
        ro = ThreadingHTTPServer(("127.0.0.1", 0), server_mod.make_handler(ro_app))
        ro_port = ro.server_address[1]
        threading.Thread(target=ro.serve_forever, daemon=True).start()
        time.sleep(0.2)
        check(
            "read-only refuses writes",
            Client(ro_port)("POST", "/api/run", {}, token="")[0],
            403,
        )
        ro.shutdown()

        # -- the page carries its token -------------------------------------
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=20) as resp:
            page = resp.read().decode()
        found = re.search(r'__HARNESS_ARENA_TOKEN__ = "([^"]+)"', page)
        check("token injected into the page", bool(found and found[1] == TOKEN), True)

        # -- key redaction ---------------------------------------------------
        app.config.endpoint.api_key = KEY
        status, state = call("GET", "/api/state")
        check("state responds", status, 200)
        check("key never sent to the browser", KEY in json.dumps(state), False)
        check(
            "redaction marker sent instead",
            state["config"]["endpoint"]["api_key"],
            REDACTED,
        )
        check("but the UI is told one is set", state["api_key_set"], True)
        app.config.endpoint.api_key = ""

        # -- input validation -------------------------------------------------
        check(
            "unknown harness",
            call("POST", "/api/run", {"harnesses": ["nope"]})[0],
            400,
        )
        check(
            "subset path traversal",
            call("POST", "/api/run", {"harnesses": [], "subset": "../../etc"})[0],
            400,
        )
        check(
            "out-of-range multiplier",
            call("POST", "/api/run", {"agent_timeout_multiplier": 9999})[0],
            400,
        )
        check(
            "non-numeric multiplier",
            call("POST", "/api/run", {"agent_timeout_multiplier": "abc"})[0],
            400,
        )
        # Deleting a run takes an id straight from the URL; without the guard in
        # delete_run this is an arbitrary-delete endpoint on the host.
        check(
            "run id traversal",
            call("DELETE", "/api/runs/..%2F..%2Fetc", {})[0],
            404,
        )
        check(
            "snapshot name traversal",
            call("POST", "/api/export", {"name": "../evil.html"})[0],
            400,
        )
        check(
            "unknown provider",
            call("POST", "/api/config", {"endpoint": {"provider": "bogus"}})[0],
            400,
        )
        check("unknown route", call("GET", "/api/nope")[0], 404)
        check("stop with nothing running", call("POST", "/api/run/stop", {})[0], 409)

        # -- an empty base_url must not silently repoint the endpoint --------
        #
        # The Setup form posts every field on every save, so a blank URL box
        # arrives as "" rather than as an absent key. Written through, it
        # resolves to the *provider default* -- and the next run fails against
        # a host the user never typed, which reads as stale config rather than
        # as data loss. Exercised on _endpoint_from directly: going through
        # POST /api/config would write over the real config.yaml.
        stored = "http://127.0.0.1:9/v1"
        check(
            "empty base_url keeps the stored one",
            app._endpoint_from({"endpoint": {"base_url": ""}}).base_url,
            stored,
        )
        check(
            "  and does not fall back to the provider default",
            app._endpoint_from({"endpoint": {"base_url": ""}}).resolved_base_url(),
            stored,
        )
        check(
            "absent base_url keeps the stored one",
            app._endpoint_from({"endpoint": {}}).base_url,
            stored,
        )
        check(
            "a supplied base_url still wins",
            app._endpoint_from({"endpoint": {"base_url": "http://h:1/v1"}}).base_url,
            "http://h:1/v1",
        )
        # Without a sentinel, "unchanged" would make the field unclearable.
        check(
            "'clear' removes it",
            app._endpoint_from({"endpoint": {"base_url": "clear"}}).base_url,
            "",
        )

        # -- model listing ------------------------------------------------------
        #
        # Listing must not require a model to already be chosen: a hosted
        # provider serves hundreds, and you cannot pick from a list you were
        # never able to fetch. So this path validates the URL and the key only.
        status, body = call("POST", "/api/config/models", {})
        check("unreachable endpoint reports an error", status >= 400, True)
        check(
            "  and says where it tried",
            "9" in body.get("error", "") or "models" in body.get("error", ""),
            True,
        )
        status, body = call(
            "POST", "/api/config/models",
            {"endpoint": {"provider": "openrouter", "model": ""}},
        )
        check("hosted provider without a key is refused", status >= 400, True)
        check("  and names the missing key", "API key" in body.get("error", ""), True)
        check(
            "listing rejects an unknown provider",
            call("POST", "/api/config/models", {"endpoint": {"provider": "nope"}})[0],
            400,
        )

        # -- registry write guards ---------------------------------------------
        # registry.yaml is committed, so a credential must never land in it.
        status, body = call(
            "POST", "/api/harness",
            {"id": "leaky", "spec": {
                "agent": "x:Y", "agent_kwargs": {"api_key": "sk-abcdefghijklmnop"}}},
        )
        check("credential in the catalog", status, 400)
        check("  and the error explains why", "committed" in body.get("error", ""), True)

        check(
            "unknown placeholder",
            call("POST", "/api/harness",
                 {"id": "typo", "spec": {"agent": "x:Y",
                                         "agent_kwargs": {"base_url": "{base_urls}"}}})[0],
            400,
        )
        check(
            "invalid harness id",
            call("POST", "/api/harness", {"id": "Bad Id!", "spec": {"agent": "x:Y"}})[0],
            400,
        )
        check(
            "missing agent",
            call("POST", "/api/harness", {"id": "noagent", "spec": {"label": "x"}})[0],
            400,
        )
        check(
            "unknown harness field",
            call("POST", "/api/harness",
                 {"id": "weird", "spec": {"agent": "x:Y", "bogus": 1}})[0],
            400,
        )

        # A valid one must actually be written, or the guards above prove nothing.
        status, body = call(
            "POST", "/api/harness",
            {"id": "mycli", "spec": {
                "label": "My CLI", "agent": "harnesses.mycli:MyCli",
                "agent_kwargs": {"base_url": "{base_url}", "api_key": "{api_key}"}}},
        )
        check("valid harness accepted", status, 200)
        check("  and persisted", "mycli" in registry_mod.load()["harnesses"], True)
        check("  placeholders preserved",
              registry_mod.load()["harnesses"]["mycli"]["agent_kwargs"]["api_key"],
              "{api_key}")
        # A save posts only the fields the editor renders, and an upsert
        # replaces the entry -- so editing a label through the UI used to
        # delete everything the form does not show. min_context_window is the
        # dangerous one: dropping it from hermes restores a run that fails
        # identically on every task.
        call("POST", "/api/harness",
             {"id": "mycli", "spec": {
                 "label": "My CLI", "agent": "harnesses.mycli:MyCli",
                 "min_context_window": 64000,
                 "host_env": {"SOME_VAR": "1"},
                 "agent_kwargs": {"base_url": "{base_url}"}}})
        call("POST", "/api/harness",
             {"id": "mycli", "spec": {          # what the form actually sends
                 "label": "Renamed", "agent": "harnesses.mycli:MyCli",
                 "agent_kwargs": {"base_url": "{base_url}"}}})
        saved = registry_mod.load()["harnesses"]["mycli"]
        check("an edit applies the field it changed", saved["label"], "Renamed")
        check("  and keeps a field the form omits", saved.get("min_context_window"), 64000)
        check("  including nested ones", (saved.get("host_env") or {}).get("SOME_VAR"), "1")

        check("delete a harness",
              call("DELETE", "/api/harness/mycli", {})[0], 200)
        check("  and it is gone", "mycli" in registry_mod.load()["harnesses"], False)
        check("delete an unknown harness",
              call("DELETE", "/api/harness/ghost", {})[0], 400)

        # -- defaults bounds ----------------------------------------------------
        check("default above its bound",
              call("POST", "/api/defaults", {"n_concurrent": 999})[0], 400)
        check("default below its bound",
              call("POST", "/api/defaults", {"agent_timeout_multiplier": 0})[0], 400)
        check("non-editable default",
              call("POST", "/api/defaults", {"secret_knob": 1})[0], 400)
        # Harbor rejects this combination outright, and it is an easy slider slip.
        check("more agents than trials",
              call("POST", "/api/defaults",
                   {"n_concurrent": 1, "n_concurrent_agents": 4})[0], 400)
        status, body = call("POST", "/api/defaults", {"agent_timeout_multiplier": 8})
        check("valid default accepted", status, 200)
        check("  and applied", body["defaults"]["agent_timeout_multiplier"], 8.0)

        # -- run directory deletion --------------------------------------------
        target = runs / "hermes__model__20260101T000000Z"
        target.mkdir()
        (target / "harness-bench.json").write_text("{}", encoding="utf-8")
        check("delete a real run", call("DELETE", f"/api/runs/{target.name}", {})[0], 200)
        check("  and it is gone", target.exists(), False)

    finally:
        server.shutdown()
        registry_mod.REGISTRY_PATH = REGISTRY_PATH
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("FAILED: " + ", ".join(failures) if failures else "all checks passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
