"""HTTP server: stdlib `http.server`, read-only GET API under /api/v1 and static assets under /."""

from __future__ import annotations

import json
import mimetypes
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit

from .data import API_VERSION, ApiError, SiteData, national_csv, simulate_csv

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_URL_LENGTH = 2048
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin", "X-Frame-Options": "DENY",
}
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8"}


def resolve_static(root: Path, url_path: str) -> "Path | None":
    """Map a URL path to a file inside `root`; None for anything that escapes it, is hidden, or is not a regular file."""
    rel = unquote(url_path)
    if "\x00" in rel or "\\" in rel or ":" in rel:
        return None
    parts = [p for p in rel.split("/") if p]
    if any(p in (".", "..") or p.startswith(".") for p in parts):
        return None
    target = (root.joinpath(*parts) if parts else root / "index.html")
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def make_handler(site: SiteData, static_dir: Path = STATIC_DIR, quiet: bool = False) -> "type[BaseHTTPRequestHandler]":
    api: dict[str, Callable[[dict[str, list[str]]], "tuple[str, bytes, dict[str, str]]"]] = {}

    def json_body(obj: Any) -> bytes:
        return (json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")

    def jsonify(fn: Callable[[dict[str, list[str]]], Any]):
        return lambda q: ("application/json; charset=utf-8", json_body(fn(q)), {})

    def csv_route(compute: Callable[[dict[str, list[str]]], Any], render: Callable[[Any], str], stem: str):
        def run(q: dict[str, list[str]]):
            res = compute(q)
            name = f"{stem}_{res['request']['start'][:10]}_{res['request']['end'][:10]}{'_DEMO-SYNTHETIC' if res['synthetic'] else ''}.csv"
            return ("text/csv; charset=utf-8", render(res).encode("utf-8"), {"Content-Disposition": f'attachment; filename="{name}"'})
        return run

    def only_none(fn: Callable[[], Any]):
        def run(q: dict[str, list[str]]):
            if q:
                raise ApiError(400, "unknown_parameter", f"This endpoint takes no parameters (got '{next(iter(q))}').", next(iter(q)))
            return fn()
        return run

    p = f"/api/{API_VERSION}"
    api[f"{p}/meta"] = jsonify(only_none(site.meta))
    api[f"{p}/archetypes"] = jsonify(only_none(site.archetype_list))
    api[f"{p}/simulate"] = jsonify(site.simulate)
    api[f"{p}/simulate.csv"] = csv_route(site.simulate, simulate_csv, "wind_simulation")
    api[f"{p}/national"] = jsonify(site.national_series)
    api[f"{p}/national.csv"] = csv_route(site.national_series, national_csv, "national_wind")

    class Handler(BaseHTTPRequestHandler):
        server_version = "windpower-site"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # never log query strings (coordinates) or errors verbosely
            if not quiet:
                sys.stderr.write(f"{self.address_string()} {self.command} {urlsplit(self.path).path} -> {args[1] if len(args) > 1 else ''}\n")

        # -- response helpers
        def _send(self, status: int, ctype: str, body: bytes, extra: "dict[str, str] | None" = None, cache: str = "no-store") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            for k, v in {**SECURITY_HEADERS, **(extra or {})}.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _error(self, err: ApiError) -> None:
            self._send(err.status, "application/json; charset=utf-8", json_body({**err.as_dict(), "data_label": site.label_block()}))

        def _handle(self) -> None:
            if len(self.path) > MAX_URL_LENGTH:
                return self._error(ApiError(414, "uri_too_long", "Request URL is too long."))
            try:
                parts = urlsplit(self.path)
                path = parts.path
                if path.startswith("/api/"):
                    route = api.get(path)
                    if route is None:
                        return self._error(ApiError(404, "not_found", f"Unknown API endpoint. Available: {', '.join(sorted(api))}."))
                    query = parse_qs(parts.query, keep_blank_values=True, strict_parsing=False, max_num_fields=20)
                    ctype, body, extra = route(query)
                    return self._send(200, ctype, body, extra)
                file = resolve_static(static_dir, path)
                if file is None:
                    return self._error(ApiError(404, "not_found", "No such file."))
                ctype = CONTENT_TYPES.get(file.suffix.lower()) or mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                self._send(200, ctype, file.read_bytes(), cache="no-cache")
            except ApiError as err:
                self._error(err)
            except ValueError as err:  # e.g. parse_qs max_num_fields
                self._error(ApiError(400, "bad_request", "Malformed request."))
            except Exception:  # noqa: BLE001 - never leak a traceback to the client
                sys.stderr.write("internal error while handling a request\n")
                self._error(ApiError(500, "internal_error", "Internal server error."))

        def do_GET(self) -> None:
            self._handle()

        do_HEAD = do_GET

        def _not_allowed(self) -> None:
            self._send(HTTPStatus.METHOD_NOT_ALLOWED, "application/json; charset=utf-8",
                       json_body({"error": {"code": "method_not_allowed", "message": "This server is read-only: use GET.", "param": None}}),
                       {"Allow": "GET, HEAD"})

        do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _not_allowed

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False   # on Windows SO_REUSEADDR would let a second process bind the same port silently


def make_server(site: SiteData, host: str = "127.0.0.1", port: int = 8000, *, static_dir: Path = STATIC_DIR, quiet: bool = False) -> ThreadingHTTPServer:
    return _Server((host, port), make_handler(site, static_dir, quiet))


def start_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    return t


def url_for(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/"


def run(site_dir: "str | Path", host: str = "127.0.0.1", port: int = 8000, say: Callable[[str], None] = print) -> int:
    """Serve until interrupted. Prints the URL and the data label first."""
    site = SiteData.load(site_dir)
    server = make_server(site, host, port)
    say(f"Serving {site.manifest['label']} from {site.site_dir}")
    say(f"Open {url_for(server)}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        say("Stopping.")
    finally:
        server.server_close()
    return 0
