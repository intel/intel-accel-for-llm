# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlencode
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List, Callable, Union
import urllib.request
import urllib.error

from ..envs import envs

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = envs.IAXL_API_TIMEOUT


_get_routes: Dict[str, Callable[[dict], dict]] = {}
_post_routes: Dict[str, Callable[[dict], dict]] = {}


def register(method: str, path: str, handler: Callable) -> None:
    m = method.upper()
    if m == "GET":
        _get_routes[path] = handler
    elif m == "POST":
        _post_routes[path] = handler
    else:
        raise ValueError(f"Unsupported method: {method!r} (expected GET/POST)")
    logger.debug("Registered %s %s", m, path)


def start_mgmt_server(
    role: str, rank: int = 0, num_workers: int = 1
) -> Union["MgmtServer", "ControllerServer"]:

    if role == "controller":
        port = envs.IAXL_API_CONTROLLER_PORT
        worker_base_port = envs.IAXL_API_WORKER_BASE_PORT
        server = ControllerServer(
            port=port,
            num_workers=num_workers,
            worker_base_port=worker_base_port,
        )
        server.start()
        logger.info(
            "Controller server started on port %d (workers: %d, base_port: %d)",
            port,
            num_workers,
            worker_base_port,
        )
        return server
    elif role == "worker":
        base_port = envs.IAXL_API_WORKER_BASE_PORT
        port = base_port + rank
        server = MgmtServer(port=port, name=f"worker-{rank}")
        server.start()
        logger.info("Worker server started on port %d (rank %d)", port, rank)
        return server
    else:
        raise ValueError(f"Unknown role: {role!r} (expected 'controller' or 'worker')")


def stop_mgmt_server(owner) -> None:
    server = getattr(owner, "_mgmt_server", None)
    if server is not None:
        server.stop()
        owner._mgmt_server = None
        logger.info("Management server stopped")


def _make_handler(server_name: str):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

            handler = _get_routes.get(path)
            if handler is not None:
                try:
                    result = handler(params)
                    self._json_ok(result)
                except Exception as e:
                    logger.exception("Handler error: GET %s", path)
                    self._json_err(str(e), 500)
            else:
                self._json_err("Not found", 404)

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")

            content_len = int(self.headers.get("Content-Length", 0))
            body: dict = {}
            if content_len > 0:
                body = json.loads(self.rfile.read(content_len))

            handler = _post_routes.get(path)
            if handler is not None:
                try:
                    result = handler(body)
                    self._json_ok(result)
                except Exception as e:
                    logger.exception("Handler error: POST %s", path)
                    self._json_err(str(e), 500)
            else:
                self._json_err("Not found", 404)

        def _json_ok(self, data: Any, status: int = 200):
            self._send_json(data, status)

        def _json_err(self, message: str, status: int = 400):
            self._send_json({"error": message}, status)

        def _send_json(self, data: Any, status: int):
            body_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, fmt, *args):
            logger.debug("%s %s", server_name, fmt % args)

    return _Handler


class MgmtServer:
    def __init__(self, port: int, name: str = "mgmt"):
        self.port = port
        self.name = name
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        handler_class = _make_handler(self.name)
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), handler_class)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name=f"mgmt-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


class _WorkerClient:
    def __init__(
        self, rank: int, host: str, port: int, timeout: int = _DEFAULT_TIMEOUT
    ):
        self.rank = rank
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def get(self, path: str) -> dict:
        url = self.base_url + path
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError) as e:
            return {"error": str(e), "rank": self.rank}

    def post(self, path: str, body: dict) -> dict:
        url = self.base_url + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError) as e:
            return {"error": str(e), "rank": self.rank}


def _parse_rank_targets(query_params: dict) -> Optional[List[int]]:
    if "rank" in query_params:
        return [int(query_params["rank"])]
    if "ranks" in query_params:
        return [int(r) for r in query_params["ranks"].split(",")]
    return None


def _strip_controller_params(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params.pop("rank", None)
    params.pop("ranks", None)
    if params:
        query = urlencode({k: v[0] for k, v in params.items()})
        return f"{parsed.path}?{query}"
    return parsed.path


def _make_controller_handler(controller: "ControllerServer"):
    class _ControllerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            ranks = _parse_rank_targets(params)
            forward_path = _strip_controller_params(self.path)
            results = controller.broadcast(
                lambda c, p=forward_path: c.get(p), ranks=ranks
            )
            self._json_ok({"workers": results})

        def do_POST(self):
            parsed = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            ranks = _parse_rank_targets(params)
            content_len = int(self.headers.get("Content-Length", 0))
            body: dict = {}
            if content_len > 0:
                body = json.loads(self.rfile.read(content_len))
            forward_path = _strip_controller_params(self.path)
            results = controller.broadcast(
                lambda c, p=forward_path, b=body: c.post(p, b), ranks=ranks
            )
            self._json_ok({"workers": results})

        def _json_ok(self, data: Any, status: int = 200):
            self._send_json(data, status)

        def _json_err(self, message: str, status: int = 400):
            self._send_json({"error": message}, status)

        def _send_json(self, data: Any, status: int):
            body_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, fmt, *args):
            logger.debug("Controller %s", fmt % args)

    return _ControllerHandler


class ControllerServer:
    def __init__(
        self,
        port: int,
        num_workers: int,
        worker_base_port: int,
        worker_host: str = "127.0.0.1",
    ):
        self.port = port
        self.num_workers = num_workers
        self.worker_base_port = worker_base_port

        self._workers: Dict[int, _WorkerClient] = {}
        for rank in range(num_workers):
            self._workers[rank] = _WorkerClient(
                rank=rank,
                host=worker_host,
                port=worker_base_port + rank,
            )

        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="mgmt-ctrl"
        )
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        handler_class = _make_controller_handler(self)
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), handler_class)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="mgmt-controller",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._executor.shutdown(wait=False)

    def broadcast(
        self, fn: Callable, ranks: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        target_ranks = ranks if ranks is not None else list(self._workers.keys())
        futures = {}
        for rank in target_ranks:
            client = self._workers.get(rank)
            if client is not None:
                futures[rank] = self._executor.submit(fn, client)
            else:
                futures[rank] = None

        results: Dict[str, Any] = {}
        for rank in target_ranks:
            future = futures.get(rank)
            if future is None:
                results[str(rank)] = {"error": f"Unknown rank {rank}"}
            else:
                try:
                    results[str(rank)] = future.result(timeout=_DEFAULT_TIMEOUT)
                except Exception as e:
                    results[str(rank)] = {"error": str(e), "rank": rank}
        return results
