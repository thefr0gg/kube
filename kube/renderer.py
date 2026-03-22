"""Python-facing wrapper around the native Kube editor.

This module exists to translate a high-level `Kube` object into the primitive
values that the native renderer understands: raw buffer pointers, scalar shape
metadata, UTF-8 labels, and callback functions for mutation, flushing, and the
built-in Python debug console.
"""

import contextlib
import ctypes
import io
import json
import math
import pprint
import re
import traceback
from rlcompleter import Completer

from kube._dll import FLUSH_CB, RUN_CODE_CB, SET_FACE_CB, load_dll


_CONSOLE_COMPLETE_PREFIX = "\x1dCOMPLETE\x1d"


class Renderer:
    """Thin bridge object that launches the native editor for one kube.

    The class exists to keep renderer-specific setup out of `Kube` itself. It
    stores launch parameters, prepares callback wrappers, and invokes the
    exported `kube_run(...)` function with the exact signature expected by the
    native code.

    Attributes:
        kube: The `Kube` instance being displayed or edited.
        window_size: Desired viewer size as a `(width, height)` pair.
        read_only: Whether the native editor should block mutation callbacks.
        slice_axis: Optional initial slice axis. If set, the viewer boots in
            slice mode instead of shell mode.
        slice_index: Optional initial slice index for slice mode.
        lod: Initial level-of-detail sampling factor used by the renderer.
    """

    def __init__(
        self,
        kube,
        window_size=(1024, 768),
        *,
        read_only: bool = False,
        slice_axis: int | None = None,
        slice_index: int | None = None,
        lod: int = 1,
    ):
        """Store launch-time renderer configuration for one kube instance.

        The constructor exists so viewer configuration can be assembled in
        Python first and then handed off to the native library in one step.
        """
        self.kube = kube
        self.window_size = window_size
        self.read_only = read_only
        self.slice_axis = slice_axis
        self.slice_index = slice_index
        self.lod = max(1, int(lod))
        self._set_face_callback = None
        self._flush_callback = None
        self._run_code_callback = None
        self._file_label = None
        self._storage_label = None
        self._console_globals = {
            "__builtins__": __builtins__,
            "json": json,
            "math": math,
            "kube": self.kube,
        }

        try:
            import polars as pl  # type: ignore
        except Exception:
            pl = None
        self._console_globals["pl"] = pl
        self._console_globals["cell"] = lambda x, y, z: self.kube.get_cell(x, y, z)
        self._console_globals["packed"] = lambda x, y, z: self.kube.get_packed(x, y, z)
        self._console_globals["face"] = lambda x, y, z, face_idx: self.kube.get_face(x, y, z, face_idx)
        self._console_globals["selection_bounds"] = self._selection_bounds
        self._console_globals["inspect_selection"] = self._inspect_selection
        self._console_globals["frame"] = self._frame
        self._console_globals["histogram"] = self._histogram
        self._console_globals["set_selected"] = self._set_selected
        self._console_globals["console_api"] = self._console_api

    def _current_selection(self) -> dict:
        """Return the latest selection dict stored in the console namespace.

        Console helpers need live access to the most recent editor selection,
        and this small accessor keeps that lookup in one place.
        """
        selection = self._console_globals.get("selection")
        if not selection:
            raise ValueError("no selected cell in the editor")
        return selection

    def _selection_bounds(self, radius: int = 0):
        """Return cube bounds centered on the current selection.

        This helper exists because region-oriented debugging usually starts from
        "show me a neighborhood around the selected cell" rather than manual
        bound construction.
        """
        selection = self._current_selection()
        radius = max(0, int(radius))
        x = selection["x"]
        y = selection["y"]
        z = selection["z"]
        return (
            (max(0, x - radius), min(self.kube.W, x + radius + 1)),
            (max(0, y - radius), min(self.kube.H, y + radius + 1)),
            (max(0, z - radius), min(self.kube.D, z + radius + 1)),
        )

    def _inspect_selection(self) -> dict:
        """Return a richer Python dict for the current selected cell.

        The raw `selected_cell` object is helpful, but debug sessions often need
        packed value, coordinates, and the active face value together in one
        structure.
        """
        selection = self._current_selection()
        x = selection["x"]
        y = selection["y"]
        z = selection["z"]
        face_idx = selection["face"]
        cell = self.kube.get_cell(x, y, z)
        return {
            "selection": dict(selection),
            "packed": self.kube.get_packed(x, y, z),
            "faces": list(cell.faces),
            "active_face_value": self.kube.get_face(x, y, z, face_idx),
        }

    def _frame(self, bounds=None, *, radius: int | None = None, include_packed: bool = True):
        """Return a debug DataFrame for a region or selection neighborhood.

        This helper exists so console users can quickly materialize a Polars
        frame without remembering the exact `to_polars(...)` argument shape.
        """
        if bounds is None and radius is not None:
            bounds = self._selection_bounds(radius)
        if bounds is None:
            bounds = self.kube.full_bounds()
        return self.kube.to_polars(bounds, include_packed=include_packed)

    def _histogram(self, bounds=None, *, radius: int | None = None, normalize: bool = False):
        """Return face-value counts for a region or selection neighborhood.

        The helper exists because histogram checks are a common debugging move
        when verifying import, mutation, or class-balance behavior.
        """
        if bounds is None and radius is not None:
            bounds = self._selection_bounds(radius)
        return self.kube.face_histogram(bounds=bounds, normalize=normalize)

    def _set_selected(self, value: int, *, face_idx: int | None = None) -> dict:
        """Write one value to the selected cell's active face and flush it.

        This exists so debug-console mutations can target the current selection
        in one call without manually unpacking coordinates.
        """
        selection = self._current_selection()
        if face_idx is None:
            face_idx = selection["face"]
        self.kube.set_face(selection["x"], selection["y"], selection["z"], int(face_idx), int(value))
        self.kube.flush()
        return self._inspect_selection()

    def _console_api(self) -> dict:
        """Return a short description of the built-in console helper surface.

        This helper exists so the editor can self-document its Python namespace
        from inside the console itself.
        """
        return {
            "objects": [
                "kube",
                "selection",
                "hover",
                "selected_cell",
                "hover_cell",
                "selected_face_value",
                "hover_face_value",
                "selected_packed",
                "hover_packed",
                "_",
            ],
            "helpers": [
                "cell(x, y, z)",
                "packed(x, y, z)",
                "face(x, y, z, face_idx)",
                "selection_bounds(radius=0)",
                "inspect_selection()",
                "frame(bounds=None, radius=None, include_packed=True)",
                "histogram(bounds=None, radius=None, normalize=False)",
                "set_selected(value, face_idx=None)",
            ],
        }

    def _write_console_output(self, output_buffer, output_cap: int, text: str) -> None:
        """Write interpreter output into the native renderer's mutable buffer.

        The C editor allocates a fixed char buffer for command results. This
        helper exists to handle truncation and null termination correctly from
        Python.
        """
        if output_cap <= 0 or not output_buffer:
            return
        max_bytes = max(0, int(output_cap) - 1)
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) > max_bytes:
            suffix = b"... [truncated]"
            encoded = encoded[: max(0, max_bytes - len(suffix))] + suffix
        if hasattr(output_buffer, "contents"):
            address = ctypes.addressof(output_buffer.contents)
        else:
            address = ctypes.addressof(output_buffer)
        if encoded:
            ctypes.memmove(address, encoded, len(encoded))
        ctypes.memset(address + len(encoded), 0, 1)

    def _format_console_value(self, value) -> str:
        """Render a console result in a compact but readable debug format.

        Plain `repr(...)` works for simple scalars, but larger dict/list-shaped
        debugging values become hard to scan quickly. This helper exists to use
        a wider pretty-printer for common container types while leaving custom
        object reprs intact.
        """
        if isinstance(value, (dict, list, tuple, set)):
            return pprint.pformat(value, width=88, sort_dicts=False, compact=False)
        return repr(value)

    def _format_console_success(self, captured_text: str, result) -> str:
        """Assemble labeled console sections for stdout and expression values.

        The editor console should read like a small terminal transcript rather
        than one unstructured blob. This helper exists to add stable section
        labels so users can tell printed output apart from the final result.
        """
        blocks: list[str] = []
        text = captured_text.strip()
        if text:
            blocks.append(f"output\n------\n{text}")
        if result is not None:
            blocks.append(f"result\n------\n{self._format_console_value(result)}")
        if not blocks:
            return "ok"
        return "\n\n".join(blocks)

    def _complete_console_input(self, source: str, namespace: dict) -> tuple[str, str]:
        """Return an updated input line plus a small completion summary.

        The native editor only owns the text box, so Python has to provide the
        actual namespace-aware completion logic. This helper exists to power
        `Tab` completion for names like `kube`, `selection`, and dotted object
        attributes exposed through the live console namespace.
        """
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\.?)$", source)
        if not match:
            return source, "tab completion works for Python names and dotted attributes"

        token = match.group(1)
        base = source[: match.start(1)]
        completer = Completer(namespace)
        matches: list[str] = []
        seen: set[str] = set()
        state = 0

        while True:
            candidate = completer.complete(token, state)
            if candidate is None:
                break
            if candidate not in seen:
                seen.add(candidate)
                matches.append(candidate)
            state += 1

        if not matches:
            return source, f"no matches for {token!r}"

        shared = matches[0]
        for candidate in matches[1:]:
            while shared and not candidate.startswith(shared):
                shared = shared[:-1]
        if len(shared) < len(token):
            shared = token

        completed = base + shared
        preview = matches[:12]
        if len(matches) == 1:
            return completed, f"completed: {matches[0]}"
        summary = ", ".join(preview)
        if len(matches) > len(preview):
            summary += f", ... +{len(matches) - len(preview)} more"
        return completed, f"{len(matches)} matches\n{summary}"

    def _console_context(
        self,
        has_selection: int,
        sel_x: int,
        sel_y: int,
        sel_z: int,
        sel_face: int,
        has_hover: int,
        hover_x: int,
        hover_y: int,
        hover_z: int,
        hover_face: int,
    ) -> dict:
        """Build the dynamic interpreter context for the current editor state.

        The built-in console is most useful when it exposes the live editor
        selection and hover target directly, so this helper packages those
        values into convenient Python objects.
        """
        selection = None
        hover = None
        selected_cell = None
        hover_cell = None
        selected_face_value = None
        hover_face_value = None
        selected_packed = None
        hover_packed = None

        if has_selection:
            selection = {"x": sel_x, "y": sel_y, "z": sel_z, "face": sel_face}
            selected_cell = self.kube.get_cell(sel_x, sel_y, sel_z)
            selected_face_value = self.kube.get_face(sel_x, sel_y, sel_z, sel_face)
            selected_packed = self.kube.get_packed(sel_x, sel_y, sel_z)
        if has_hover:
            hover = {"x": hover_x, "y": hover_y, "z": hover_z, "face": hover_face}
            hover_cell = self.kube.get_cell(hover_x, hover_y, hover_z)
            hover_face_value = self.kube.get_face(hover_x, hover_y, hover_z, hover_face)
            hover_packed = self.kube.get_packed(hover_x, hover_y, hover_z)

        return {
            "selection": selection,
            "hover": hover,
            "selected_cell": selected_cell,
            "hover_cell": hover_cell,
            "selected_face_value": selected_face_value,
            "hover_face_value": hover_face_value,
            "selected_packed": selected_packed,
            "hover_packed": hover_packed,
            "view": {
                "read_only": self.read_only,
                "slice_axis": self.slice_axis,
                "slice_index": self.slice_index,
                "lod": self.lod,
                "window_size": self.window_size,
            },
        }

    def _execute_console_code(
        self,
        code_text,
        has_selection: int,
        sel_x: int,
        sel_y: int,
        sel_z: int,
        sel_face: int,
        has_hover: int,
        hover_x: int,
        hover_y: int,
        hover_z: int,
        hover_face: int,
        output_buffer,
        output_cap: int,
    ) -> int:
        """Execute one console snippet from the native editor.

        The callback exists to power the editor's built-in Python interpreter.
        It keeps a persistent namespace across commands, captures stdout/stderr,
        and returns either the expression value or a traceback string.
        """
        namespace = self._console_globals
        namespace.update(
            self._console_context(
                has_selection,
                sel_x,
                sel_y,
                sel_z,
                sel_face,
                has_hover,
                hover_x,
                hover_y,
                hover_z,
                hover_face,
            )
        )
        code = code_text.decode("utf-8", errors="replace")
        capture = io.StringIO()

        try:
            if code.startswith(_CONSOLE_COMPLETE_PREFIX):
                updated_input, completion_text = self._complete_console_input(
                    code[len(_CONSOLE_COMPLETE_PREFIX) :],
                    namespace,
                )
                payload = updated_input if not completion_text else f"{updated_input}\n{completion_text}"
                self._write_console_output(output_buffer, output_cap, payload)
                return 1

            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                try:
                    compiled = compile(code, "<kube-editor>", "eval")
                except SyntaxError:
                    compiled = compile(code, "<kube-editor>", "exec")
                    exec(compiled, namespace, namespace)
                    result = None
                else:
                    result = eval(compiled, namespace, namespace)
                    namespace["_"] = result
            output = self._format_console_success(capture.getvalue(), result)
            try:
                self.kube.flush()
            except Exception:
                pass
            self._write_console_output(output_buffer, output_cap, output)
            return 1
        except Exception:
            self._write_console_output(
                output_buffer,
                output_cap,
                f"error\n-----\n{traceback.format_exc(limit=6).strip()}",
            )
            return 0

    def _make_callbacks(self):
        """Create stable `ctypes` callback objects for the C renderer.

        The native renderer needs callable function pointers with C-compatible
        signatures. This method exists to wrap `Kube.set_face(...)`,
        `Kube.flush()`, and the built-in debug console executor in small
        adapters that return `1` on success and `0` on failure.
        """

        def set_face(x, y, z, face_idx, value):
            try:
                self.kube.set_face(x, y, z, face_idx, value)
            except Exception:
                return 0
            return 1

        def flush():
            try:
                self.kube.flush()
            except Exception:
                return 0
            return 1

        self._set_face_callback = SET_FACE_CB(set_face)
        self._flush_callback = FLUSH_CB(flush)
        self._run_code_callback = RUN_CODE_CB(self._execute_console_code)
        self._file_label = self.kube.file_label.encode("utf-8")
        self._storage_label = self.kube.storage_label.encode("utf-8")

    def run(self):
        """Launch the native editor with the current renderer configuration.

        This method exists as the single handoff point from Python into the
        C/raylib viewer. It converts Python-side state into `ctypes` values and
        passes the raw mapped buffer address plus callbacks to the shared
        library.
        """
        self._make_callbacks()
        view_mode = 1 if self.slice_axis is not None else 0
        slice_axis = 0 if self.slice_axis is None else int(self.slice_axis)
        slice_index = 0 if self.slice_index is None else int(self.slice_index)
        window_width = max(960, int(self.window_size[0]))
        window_height = max(640, int(self.window_size[1]))

        load_dll().kube_run(
            self.kube.buffer_address(),
            ctypes.c_int(self.kube.W),
            ctypes.c_int(self.kube.H),
            ctypes.c_int(self.kube.D),
            ctypes.c_int(self.kube.S),
            ctypes.c_int(self.kube.itemsize),
            ctypes.c_int(view_mode),
            ctypes.c_int(slice_axis),
            ctypes.c_int(slice_index),
            ctypes.c_int(self.lod),
            ctypes.c_int(window_width),
            ctypes.c_int(window_height),
            ctypes.c_int(1 if self.read_only else 0),
            ctypes.c_char_p(self._file_label),
            ctypes.c_char_p(self._storage_label),
            self._set_face_callback,
            self._flush_callback,
            self._run_code_callback,
        )
