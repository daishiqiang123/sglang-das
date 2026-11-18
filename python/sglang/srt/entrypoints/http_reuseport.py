"""Per-worker TCP listeners for multi-tokenizer Uvicorn serving."""

import importlib
import logging
import socket

from uvicorn import Config
from uvicorn.supervisors import multiprocess


def _bind_reuseport_socket(family, address):
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(address)
    except BaseException:
        sock.close()
        raise
    return sock


class _ReusePortProcess(multiprocess.Process):
    def target(self, sockets=None):
        # The parent reserves the port without listening. Each child has its
        # own accept queue, and the kernel distributes incoming TCP connections.
        reserved_socket = sockets[0]
        family, address = reserved_socket.family, reserved_socket.getsockname()
        reserved_socket.close()
        with _bind_reuseport_socket(family, address) as worker_socket:
            super().target(sockets=[worker_socket])


def run_uvicorn_with_reuseport(app, **kwargs):
    config = Config(app, **kwargs)
    family = socket.AF_INET6 if config.host and ":" in config.host else socket.AF_INET
    uvicorn_main = importlib.import_module("uvicorn.main")
    with _bind_reuseport_socket(family, (config.host, config.port)) as reserved_socket:
        logging.getLogger("uvicorn.error").info(
            "Uvicorn SO_REUSEPORT enabled on %s",
            reserved_socket.getsockname(),
        )
        original_process = multiprocess.Process
        multiprocess.Process = _ReusePortProcess
        try:
            uvicorn_main.Multiprocess(config, sockets=[reserved_socket]).run()
        finally:
            multiprocess.Process = original_process
