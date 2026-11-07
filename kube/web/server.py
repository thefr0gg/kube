"""Localhost FastAPI app serving the WebGL kube viewer/editor.

Replaces the old native raylib editor (`kube_run` in `src/kube_renderer.c`).
The server wraps one open `Kube` instance: REST endpoints return mesh/state
data, and a WebSocket applies click-to-edit face mutations back through the
existing `kube_storage`-backed `Kube.set_face`/`flush`.

This app is meant for `127.0.0.1` only — there is no auth, and it is never
intended to be exposed beyond the local machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kube.web.mesh import build_shell_mesh, build_slice_mesh

STATIC_DIR = Path(__file__).parent / "static"


class EditMessage(BaseModel):
    x: int
    y: int
    z: int
    face: int
    value: int


class ViewerState:
    """Holds the open `Kube` plus the current view configuration.

    Exists so REST/WebSocket handlers share one source of truth for which
    slice/LOD/visibility settings are currently active, instead of threading
    view params through every request.
    """

    def __init__(self, kube, *, read_only: bool, slice_axis: int | None, slice_index: int | None, lod: int):
        self.kube = kube
        self.read_only = read_only
        self.slice_axis = slice_axis
        self.slice_index = slice_index
        self.lod = max(1, int(lod))
        self.view_mode: Literal["shell", "slice"] = "slice" if slice_axis is not None else "shell"
        self.visible_faces = [True] * 6


def create_app(kube, *, read_only: bool = False, slice_axis: int | None = None, slice_index: int | None = None, lod: int = 1) -> FastAPI:
    """Build the FastAPI app bound to one open `Kube` instance."""
    state = ViewerState(kube, read_only=read_only, slice_axis=slice_axis, slice_index=slice_index, lod=lod)
    app = FastAPI()

    @app.get("/api/state")
    def get_state():
        return {
            "shape": [kube.W, kube.H, kube.D],
            "states": kube.S,
            "file_label": kube.file_label,
            "storage_label": kube.storage_label,
            "read_only": state.read_only,
            "view_mode": state.view_mode,
            "slice_axis": state.slice_axis,
            "slice_index": state.slice_index,
            "lod": state.lod,
        }

    @app.post("/api/view")
    def set_view(view_mode: str | None = None, slice_axis: int | None = None, slice_index: int | None = None, lod: int | None = None):
        if view_mode is not None:
            state.view_mode = "slice" if view_mode == "slice" else "shell"
        if slice_axis is not None:
            state.slice_axis = slice_axis
        if slice_index is not None:
            state.slice_index = slice_index
        if lod is not None:
            state.lod = max(1, int(lod))
        return get_state()

    @app.get("/api/mesh")
    def get_mesh():
        faces = kube.faces_to_numpy()
        if state.view_mode == "slice" and state.slice_axis is not None:
            index = state.slice_index if state.slice_index is not None else 0
            mesh = build_slice_mesh(faces, slice_axis=state.slice_axis, slice_index=index, lod=state.lod)
        else:
            mesh = build_shell_mesh(faces, lod=state.lod, visible_faces=tuple(state.visible_faces))
        return {
            "positions": mesh["positions"].ravel().tolist(),
            "colors": mesh["colors"].ravel().tolist(),
            "indices": mesh["indices"].tolist(),
            "states": mesh["states"],
            "vertex_count": int(mesh["positions"].shape[0]),
        }

    @app.get("/api/cell")
    def get_cell(x: int, y: int, z: int):
        cell = kube.get_cell(x, y, z)
        return {"x": x, "y": y, "z": z, "faces": list(cell.faces), "packed": kube.get_packed(x, y, z)}

    @app.websocket("/ws/edit")
    async def edit(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                payload = await websocket.receive_json()
                msg = EditMessage(**payload)
                if state.read_only:
                    await websocket.send_json({"ok": False, "error": "viewer is read-only"})
                    continue
                try:
                    kube.set_face(msg.x, msg.y, msg.z, msg.face, msg.value)
                    kube.flush()
                except Exception as exc:  # noqa: BLE001 - surfaced to the client, not swallowed
                    await websocket.send_json({"ok": False, "error": str(exc)})
                    continue
                await websocket.send_json(
                    {"ok": True, "x": msg.x, "y": msg.y, "z": msg.z, "face": msg.face, "value": msg.value}
                )
        except WebSocketDisconnect:
            pass

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app
