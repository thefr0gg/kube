"""Blocking launcher for the localhost kube web viewer.

Picks a free port if none is given, opens the default browser once the server
is accepting connections, and runs uvicorn in the foreground until the user
closes the browser tab (which calls `/api/shutdown`) or interrupts with Ctrl+C.
"""

from __future__ import annotations

import socket
import threading
import webbrowser

import uvicorn

from kube.web.server import create_app


def _free_port() -> int:
    """Ask the OS for an unused localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def serve(
    kube,
    *,
    read_only: bool = False,
    slice_axis: int | None = None,
    slice_index: int | None = None,
    lod: int = 1,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> None:
    """Run the FastAPI viewer for `kube` until the browser tab is closed."""
    if port == 0:
        port = _free_port()

    app = create_app(
        kube,
        read_only=read_only,
        slice_axis=slice_axis,
        slice_index=slice_index,
        lod=lod,
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    if open_browser:
        def _open():
            webbrowser.open(f"http://{host}:{port}/")

        threading.Timer(0.5, _open).start()

    print(f"kube viewer running at http://{host}:{port}/  (Ctrl+C to stop)")
    server.run()
