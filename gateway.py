#!/usr/bin/env python3
"""
Qwen3.8-27B concurrency gateway for Apple Silicon (oMLX / MLX).

An OpenAI + Anthropic-compatible reverse proxy that sits in front of a single
oMLX server and makes N client requests behave well on a memory-bandwidth-bound
Mac. It does the one thing that actually ports from the SGLang/DGX-Spark recipe
to Metal: **cap the number of requests running inside the model at once, and
FIFO-queue the rest** — so 5-6 clients all succeed instead of thrashing the
memory ceiling.

Why a gate helps on a Mac (measured on an M-series, dense 27B, 8-bit):
    concurrent   aggregate out-tok/s
        1              5.3
        2              7.9
        4             10.6   <- peak
        6              7.6   <- collapses: hits oMLX memory ceiling + cache thrash
Decode is bandwidth-bound, so running more than ~4 at once buys nothing and
starts to lose. The gateway admits ADMIT (default 4) into oMLX and queues the
rest; clients see a short wait, never an error or a stalled machine.

Stdlib only — no pip installs. Streams responses (SSE and JSON) straight
through with chunked transfer-encoding, so token streaming still feels live.

    python3 gateway.py                 # listen :8080, upstream :8000, admit 4
    PROXY_PORT=8080 UPSTREAM=http://127.0.0.1:8000 ADMIT=4 python3 gateway.py

Point any OpenAI/Anthropic client at http://127.0.0.1:8080 instead of :8000.
GET /health and GET /metrics are served locally (JSON); everything else is
proxied to the upstream oMLX server unchanged.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------- config
PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8080"))
UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
ADMIT = int(os.environ.get("ADMIT", "4"))            # requests allowed inside oMLX at once
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "32"))   # waiting slots before we shed load
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "3600"))
CHUNK = 65536

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
}


# --------------------------------------------------------------------------- FIFO admission gate
class FifoGate:
    """A fair (first-come-first-served) concurrency limiter with a bounded queue.

    Semaphores aren't ordered, so under load a late request can jump the line.
    This hands out monotonic tickets and only lets a waiter proceed when it is
    both at the head of the line and a slot is free — true FIFO, which keeps
    tail latency sane when 6 clients hit a 4-wide gate.
    """

    def __init__(self, limit: int, max_queue: int):
        self.limit = limit
        self.max_queue = max_queue
        self._cv = threading.Condition()
        self._next = 0            # next ticket to hand out
        self._serving = 0         # ticket at the head that may enter next
        self._running = 0         # currently inside the model
        self._waiting = deque()   # tickets waiting
        # metrics
        self.admitted = 0
        self.completed = 0
        self.errors = 0
        self.rejected = 0
        self.peak_running = 0
        self.peak_waiting = 0
        self._wait_samples: deque[float] = deque(maxlen=200)

    def acquire(self) -> tuple[bool, float]:
        t0 = time.time()
        with self._cv:
            if len(self._waiting) >= self.max_queue:
                self.rejected += 1
                return False, 0.0
            ticket = self._next
            self._next += 1
            self._waiting.append(ticket)
            self.peak_waiting = max(self.peak_waiting, len(self._waiting))
            while not (self._waiting and self._waiting[0] == ticket and self._running < self.limit):
                self._cv.wait()
            self._waiting.popleft()
            self._running += 1
            self.admitted += 1
            self.peak_running = max(self.peak_running, self._running)
            waited = time.time() - t0
            self._wait_samples.append(waited)
            return True, waited

    def release(self, ok: bool):
        with self._cv:
            self._running -= 1
            if ok:
                self.completed += 1
            else:
                self.errors += 1
            self._cv.notify_all()

    def snapshot(self) -> dict:
        with self._cv:
            waits = sorted(self._wait_samples)
            p50 = waits[len(waits) // 2] if waits else 0.0
            p95 = waits[int(len(waits) * 0.95)] if waits else 0.0
            return {
                "admit_limit": self.limit,
                "max_queue": self.max_queue,
                "running": self._running,
                "waiting": len(self._waiting),
                "admitted": self.admitted,
                "completed": self.completed,
                "errors": self.errors,
                "rejected_queue_full": self.rejected,
                "peak_running": self.peak_running,
                "peak_waiting": self.peak_waiting,
                "wait_seconds_p50": round(p50, 3),
                "wait_seconds_p95": round(p95, 3),
            }


GATE = FifoGate(ADMIT, MAX_QUEUE)


# --------------------------------------------------------------------------- handler
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen38-gateway/1.0"

    # keep the console quiet; we log our own one-liners
    def log_message(self, *_):
        pass

    # -- local endpoints ----------------------------------------------------
    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            up = _upstream_alive()
            return self._json(200 if up else 503, {"gateway": "ok", "upstream": up})
        if self.path == "/metrics":
            return self._json(200, GATE.snapshot())
        return self._proxy("GET")

    def do_POST(self):
        return self._proxy("POST")

    def do_DELETE(self):
        return self._proxy("DELETE")

    # -- proxy with admission gate -----------------------------------------
    def _proxy(self, method: str):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None

        got, waited = GATE.acquire()
        if not got:
            retry = 5
            payload = json.dumps({
                "error": {
                    "type": "overloaded_error",
                    "message": f"Gateway queue full ({MAX_QUEUE} waiting). Retry shortly.",
                }
            }).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", str(retry))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        ok = False
        t0 = time.time()
        try:
            ok = self._forward(method, body)
        finally:
            GATE.release(ok)
            snap = GATE.snapshot()
            sys.stderr.write(
                f"[gw] {method} {self.path.split('?')[0]} wait={waited:5.2f}s "
                f"serve={time.time()-t0:6.2f}s ok={ok} "
                f"running={snap['running']} waiting={snap['waiting']}\n"
            )

    def _forward(self, method: str, body: bytes | None) -> bool:
        url = UPSTREAM + self.path
        fwd_headers = {
            k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP
        }
        req = urllib.request.Request(url, data=body, method=method, headers=fwd_headers)
        try:
            resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
        except urllib.error.HTTPError as e:
            # upstream returned a non-2xx — relay it faithfully
            data = e.read()
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return 200 <= e.code < 400
        except Exception as e:
            payload = json.dumps({"error": {"type": "upstream_error", "message": str(e)[:200]}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return False

        # Stream the upstream response back with chunked encoding so token
        # streaming (SSE) stays live and we never buffer a whole reply.
        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                self.wfile.write(f"{len(buf):X}\r\n".encode())
                self.wfile.write(buf)
                self.wfile.write(b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False
        finally:
            resp.close()


def _upstream_alive() -> bool:
    # Generous timeout: a busy oMLX (mid-decode) can be slow to answer /v1/models
    # even though it is perfectly healthy — a short probe would false-negative.
    try:
        with urllib.request.urlopen(UPSTREAM + "/v1/models", timeout=15) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    srv = ThreadingHTTPServer((PROXY_HOST, PROXY_PORT), Handler)
    srv.daemon_threads = True
    sys.stderr.write(
        f"qwen3.8 gateway → listening {PROXY_HOST}:{PROXY_PORT}  upstream={UPSTREAM}  "
        f"admit={ADMIT} max_queue={MAX_QUEUE}\n"
        f"  clients: point OpenAI/Anthropic base URL at http://{PROXY_HOST}:{PROXY_PORT}\n"
        f"  GET /health  GET /metrics\n"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
