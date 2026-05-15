"""Minimal LSP client for resolvers.

Speaks just enough of the Language Server Protocol to let resolvers
ask a language server "where is the definition of the name at this
position?" That's the one capability we need; everything else (hover,
diagnostics, completion) is the IDE's job, not ours.

Spec reference: https://microsoft.github.io/language-server-protocol/specification

Wire format: each message is `Content-Length: N\r\n\r\n<N bytes of JSON>`.
Requests have an `id`; the server's response carries the matching `id`.
Notifications (didOpen, exit) have no `id` and expect no response.
Servers also send unprompted notifications and log messages; we read
incoming frames until we see one with our id and drop the rest.

Designed for short-lived synchronous use: launch the server, ask N
definition queries, shut down. Long-lived editor-style sessions are
out of scope.
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LspError(RuntimeError):
    """Raised on protocol or transport failures.

    Caller-friendly: __str__ returns the descriptive message, but the
    code + data fields preserve the LSP error structure for callers
    that want to react programmatically.
    """

    message: str
    code: int | None = None
    data: Any = None

    def __str__(self) -> str:
        return self.message


class LspClient:
    """Single-process LSP client over stdin/stdout.

    Usage:
        with LspClient(["typescript-language-server", "--stdio"]) as c:
            c.initialize(root_uri="file:///path/to/repo")
            c.notify("initialized", {})
            c.did_open(uri, language_id, text)
            defs = c.definition(uri, line, character)

    Threading: a reader thread drains stdout and routes frames into
    self._responses (indexed by request id) and self._notifications
    (kept for debugging — most callers ignore them).
    """

    def __init__(
        self,
        args: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.args = args
        self.cwd = str(cwd) if cwd else None
        self.env = env
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._next_id = 1
        self._responses: dict[int, dict[str, Any]] = {}
        self._response_event = threading.Event()
        self._stopped = threading.Event()
        # Diagnostic surface — callers can introspect after a session
        # to debug what the server told us.
        self.notifications: list[dict[str, Any]] = []
        self.stderr_tail: list[bytes] = []

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            env=self.env,
            bufsize=0,
        )
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="lsp-reader",
        )
        self._reader_thread.start()
        # Stderr is drained on a separate thread so a chatty server
        # doesn't fill its pipe buffer and block.
        stderr_thread = threading.Thread(
            target=self._stderr_loop, daemon=True, name="lsp-stderr",
        )
        stderr_thread.start()

    def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            try:
                self._send({"jsonrpc": "2.0", "id": self._take_id(), "method": "shutdown"})
                self._send({"jsonrpc": "2.0", "method": "exit"})
            except OSError:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            self._stopped.set()
            self._proc = None

    def __enter__(self) -> LspClient:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    # ── public LSP methods we use ────────────────────────────────────

    def initialize(self, *, root_uri: str, capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
        """`initialize` request — must come first.

        We send the minimum capabilities a definition-only client needs;
        servers tolerate a thin set and don't gate definition behind
        anything unusual.
        """
        params = {
            "processId": None,
            "rootUri": root_uri,
            "capabilities": capabilities or {
                "textDocument": {
                    "definition": {"linkSupport": False},
                    "synchronization": {"didSave": False},
                },
            },
            "trace": "off",
        }
        return self.request("initialize", params)

    def did_open(self, *, uri: str, language_id: str, text: str, version: int = 1) -> None:
        self.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": version,
                "text": text,
            },
        })

    def did_close(self, *, uri: str) -> None:
        self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})

    def definition(
        self, *, uri: str, line: int, character: int,
    ) -> list[dict[str, Any]]:
        """`textDocument/definition` — returns 0..n target locations.

        LSP allows three response shapes (single Location, list of
        Locations, or list of LocationLinks). We normalize to a list of
        dicts with `uri` and `range`; callers index into range.start
        for (line, character).
        """
        result = self.request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })
        if result is None:
            return []
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            # LocationLink has `targetUri` + `targetRange` — flatten to Location shape.
            out: list[dict[str, Any]] = []
            for item in result:
                if not isinstance(item, dict):
                    continue
                if "targetUri" in item:
                    out.append({
                        "uri": item["targetUri"],
                        "range": item.get("targetRange") or item.get("targetSelectionRange") or {},
                    })
                else:
                    out.append(item)
            return out
        return []

    # ── generic plumbing ─────────────────────────────────────────────

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        msg_id = self._take_id()
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})
        return self._await_response(msg_id)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # ── internals ────────────────────────────────────────────────────

    def _take_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, msg: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise LspError("LSP client not started")
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._proc.stdin.write(header)
        self._proc.stdin.write(body)
        self._proc.stdin.flush()

    def _await_response(self, msg_id: int) -> Any:
        deadline = self.timeout
        # Polling on the event with a short timeout lets us yield and
        # re-check the response map between wakeups (multiple responses
        # may arrive in any order).
        end = threading.Event()

        def _watchdog():
            end.wait(deadline)

        watcher = threading.Thread(target=_watchdog, daemon=True)
        watcher.start()

        while not end.is_set():
            self._response_event.wait(timeout=0.1)
            self._response_event.clear()
            if msg_id in self._responses:
                resp = self._responses.pop(msg_id)
                if "error" in resp:
                    err = resp["error"]
                    raise LspError(
                        message=err.get("message", "LSP error"),
                        code=err.get("code"),
                        data=err.get("data"),
                    )
                return resp.get("result")
            if self._stopped.is_set():
                raise LspError("LSP server exited before responding")
        raise LspError(f"LSP request {msg_id!r} timed out after {self.timeout}s")

    def _reader_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for frame in _read_frames(proc.stdout):
                self._dispatch(frame)
        except Exception:
            pass
        finally:
            self._stopped.set()
            self._response_event.set()

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self.stderr_tail.append(line)
                # Keep only the last 200 lines so we don't accumulate
                # forever if a server is chatty.
                if len(self.stderr_tail) > 200:
                    self.stderr_tail.pop(0)
        except Exception:
            pass

    def _dispatch(self, frame: dict[str, Any]) -> None:
        if "id" in frame and ("result" in frame or "error" in frame):
            try:
                msg_id = int(frame["id"])
            except (TypeError, ValueError):
                return
            self._responses[msg_id] = frame
            self._response_event.set()
            return
        # Notification or server-initiated request. We don't act on
        # those (we're a definition-only client), but stash them so
        # tests can introspect.
        self.notifications.append(frame)


def _read_frames(stream) -> Iterator[dict[str, Any]]:
    """Yield JSON dicts from an LSP `Content-Length:`-framed byte stream."""
    while True:
        header_bytes = bytearray()
        # Read header lines until the blank-line separator.
        while True:
            chunk = stream.readline()
            if not chunk:
                return  # EOF
            header_bytes += chunk
            if chunk == b"\r\n":
                break
        header_text = header_bytes.decode("ascii", errors="replace")
        length: int | None = None
        for line in header_text.splitlines():
            if not line:
                continue
            if line.lower().startswith("content-length:"):
                try:
                    length = int(line.split(":", 1)[1].strip())
                except ValueError:
                    length = None
        if length is None:
            continue  # malformed header — keep reading
        body = b""
        while len(body) < length:
            piece = stream.read(length - len(body))
            if not piece:
                return
            body += piece
        try:
            yield json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue  # skip malformed bodies
