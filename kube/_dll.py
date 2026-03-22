"""Native library loading helpers for the renderer bridge.

This module exists to keep `ctypes` setup isolated from the higher-level
renderer code. It resolves the platform-specific shared library name, loads the
editor DLL/shared object, and defines the callback signatures that the C
renderer expects when it calls back into Python.
"""

import ctypes
import os
import sys
from pathlib import Path

_dll_cache = None
_storage_dll_cache = False

SET_FACE_CB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)
FLUSH_CB = ctypes.CFUNCTYPE(ctypes.c_int)
RUN_CODE_CB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_int,
)


def _renderer_lib_names() -> tuple[str, ...]:
    """Return the platform-specific native renderer filename candidates.

    Different build tools may or may not prefix shared libraries with `lib` on
    Windows. This helper exists so the Python bridge can accept the names that
    CMake actually emits in local builds and packaged installs.
    """
    if sys.platform == "win32":
        return ("kube_renderer.dll", "libkube_renderer.dll")
    if sys.platform == "darwin":
        return ("libkube_renderer.dylib",)
    return ("libkube_renderer.so",)


def _storage_lib_names() -> tuple[str, ...]:
    """Return the platform-specific storage-accelerator filename candidates."""
    if sys.platform == "win32":
        return ("kube_storage.dll", "libkube_storage.dll")
    if sys.platform == "darwin":
        return ("libkube_storage.dylib",)
    return ("libkube_storage.so",)


def _find_library_path(candidates: tuple[str, ...]) -> Path | None:
    """Resolve the first matching native library path from the package directory."""
    for candidate in candidates:
        candidate_path = Path(__file__).parent / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def _prepare_windows_dll_directory(path: Path) -> None:
    """Add the package directory to Windows DLL resolution when available."""
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(path.parent))


def load_dll() -> ctypes.CDLL:
    """Load and cache the native renderer library.

    This function exists so the rest of the codebase can ask for the renderer
    once and rely on a configured `ctypes.CDLL` object with the correct
    argument types and return type for `kube_run(...)`.
    """
    global _dll_cache
    if _dll_cache is not None:
        return _dll_cache

    path = _find_library_path(_renderer_lib_names())
    if path is None:
        raise FileNotFoundError(
            f"{_renderer_lib_names()[0]} not found in {Path(__file__).parent}\n"
            "Build it first:\n"
            "  python build.py\n"
            "\n"
            "Windows (MinGW64):  python build.py --generator \"MinGW Makefiles\"\n"
            "Linux:              python3 build.py"
        )

    _prepare_windows_dll_directory(path)

    lib = ctypes.CDLL(str(path))
    fn = lib.kube_run
    fn.restype = None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        SET_FACE_CB,
        FLUSH_CB,
        RUN_CODE_CB,
    ]
    _dll_cache = lib
    return lib


def load_storage_dll(*, optional: bool = False) -> ctypes.CDLL | None:
    """Load and cache the native storage accelerator.

    Kube's storage runtime now requires the storage library so the packed-cell
    engine always uses the native hot path instead of silently falling back to
    the slower Python implementation.
    """
    global _storage_dll_cache
    if _storage_dll_cache is not False:
        return _storage_dll_cache

    path = _find_library_path(_storage_lib_names())
    if path is None:
        raise FileNotFoundError(
            f"{_storage_lib_names()[0]} not found in {Path(__file__).parent}\n"
            "Build it first:\n"
            "  python build.py\n"
            "\n"
            "Windows (MinGW64):  python build.py --generator \"MinGW Makefiles\"\n"
            "Linux:              python3 build.py\n"
        )

    try:
        _prepare_windows_dll_directory(path)
        lib = ctypes.CDLL(str(path))
    except OSError:
        raise

    lib.kube_storage_get_packed.restype = ctypes.c_uint64
    lib.kube_storage_get_packed.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    lib.kube_storage_set_packed.restype = None
    lib.kube_storage_set_packed.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_uint64,
    ]
    lib.kube_storage_set_packed_batch.restype = None
    lib.kube_storage_set_packed_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.kube_storage_get_face.restype = ctypes.c_uint32
    lib.kube_storage_get_face.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.kube_storage_get_faces_batch.restype = None
    lib.kube_storage_get_faces_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.kube_storage_set_face.restype = None
    lib.kube_storage_set_face.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    lib.kube_storage_set_faces_batch.restype = None
    lib.kube_storage_set_faces_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.kube_storage_zero_bytes.restype = None
    lib.kube_storage_zero_bytes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    _storage_dll_cache = lib
    return lib
