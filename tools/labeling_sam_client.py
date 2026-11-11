"""Local-only requests to the dashboard-owned SAM worker."""
import json
import socket


def request(path, **message):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(5)
        sock.connect(str(path))
        sock.sendall(json.dumps(message).encode()+b'\n')
        with sock.makefile('rb') as stream:
            value = json.loads(stream.readline(65536))
    if value.get('error'):
        raise RuntimeError(value['error'])
    return value
