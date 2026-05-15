"""LspClient tests — drive the client with a mock JSON-RPC server.

These don't require any LSP server binary; we spin up a tiny Python
subprocess that speaks the Content-Length framing and returns
canned responses keyed by method name. Validates framing, request/
response correlation by id, and basic error handling.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from otter_docs.resolvers.lsp import LspClient, LspError

_MOCK_SERVER_SCRIPT = textwrap.dedent("""\
import json, sys

def read_frame():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line == b'\\r\\n':
            break
        k, _, v = line.decode('ascii').partition(':')
        headers[k.strip().lower()] = v.strip()
    length = int(headers['content-length'])
    return json.loads(sys.stdin.buffer.read(length).decode('utf-8'))

def send_frame(obj):
    body = json.dumps(obj).encode('utf-8')
    sys.stdout.buffer.write(f'Content-Length: {len(body)}\\r\\n\\r\\n'.encode('ascii'))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()

# Send one unsolicited notification first to verify we ignore it.
send_frame({'jsonrpc': '2.0', 'method': 'window/logMessage',
            'params': {'type': 3, 'message': 'hello'}})

while True:
    frame = read_frame()
    if frame is None:
        break
    method = frame.get('method')
    if 'id' not in frame:
        # Notification; ignore unless it's `exit` which terminates.
        if method == 'exit':
            break
        continue
    msg_id = frame['id']
    if method == 'initialize':
        send_frame({'jsonrpc': '2.0', 'id': msg_id,
                    'result': {'capabilities': {}, 'serverInfo': {'name': 'mock'}}})
    elif method == 'shutdown':
        send_frame({'jsonrpc': '2.0', 'id': msg_id, 'result': None})
    elif method == 'echo':
        send_frame({'jsonrpc': '2.0', 'id': msg_id,
                    'result': frame.get('params')})
    elif method == 'fail':
        send_frame({'jsonrpc': '2.0', 'id': msg_id,
                    'error': {'code': -32603, 'message': 'mock failure'}})
    elif method == 'textDocument/definition':
        # Return a canned location for any request.
        send_frame({'jsonrpc': '2.0', 'id': msg_id, 'result': [
            {'uri': 'file:///tmp/target.ts',
             'range': {'start': {'line': 5, 'character': 9},
                       'end':   {'line': 5, 'character': 15}}}
        ]})
    else:
        send_frame({'jsonrpc': '2.0', 'id': msg_id,
                    'error': {'code': -32601, 'message': f'unknown {method}'}})
""")


@pytest.fixture
def mock_server_args(tmp_path):
    script = tmp_path / "mock_lsp.py"
    script.write_text(_MOCK_SERVER_SCRIPT)
    return [sys.executable, str(script)]


def test_initialize_and_request(mock_server_args):
    with LspClient(mock_server_args, timeout=5.0) as c:
        init = c.initialize(root_uri="file:///tmp")
        assert init["serverInfo"]["name"] == "mock"


def test_request_response_correlation(mock_server_args):
    with LspClient(mock_server_args, timeout=5.0) as c:
        c.initialize(root_uri="file:///tmp")
        assert c.request("echo", {"a": 1}) == {"a": 1}
        assert c.request("echo", {"b": 2}) == {"b": 2}


def test_server_error_raises_lsp_error(mock_server_args):
    with LspClient(mock_server_args, timeout=5.0) as c:
        c.initialize(root_uri="file:///tmp")
        with pytest.raises(LspError) as info:
            c.request("fail")
        assert info.value.code == -32603
        assert "mock failure" in str(info.value)


def test_definition_normalizes_to_list(mock_server_args):
    with LspClient(mock_server_args, timeout=5.0) as c:
        c.initialize(root_uri="file:///tmp")
        defs = c.definition(uri="file:///tmp/x.ts", line=0, character=0)
        assert isinstance(defs, list)
        assert defs[0]["uri"] == "file:///tmp/target.ts"
        assert defs[0]["range"]["start"]["line"] == 5


def test_unsolicited_notifications_are_collected(mock_server_args):
    """The mock server sends a `window/logMessage` notification at start.

    The client should ignore it for request/response purposes but stash
    it so callers can introspect.
    """
    with LspClient(mock_server_args, timeout=5.0) as c:
        c.initialize(root_uri="file:///tmp")
        # Give the reader a moment to pick up the early notification.
        import time
        time.sleep(0.1)
        # `notifications` should contain at least the logMessage.
        kinds = [n.get("method") for n in c.notifications]
        assert "window/logMessage" in kinds
