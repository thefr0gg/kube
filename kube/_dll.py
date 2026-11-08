"""Native storage library loading helpers.

This module exists to keep `ctypes` setup isolated from the higher-level
storage model code. It resolves the platform-specific shared library name and
loads the `kube_storage` accelerator used by the packed-cell runtime.
"""

import ctypes
import os
import sys
from pathlib import Path

_storage_dll_cache = False


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


def load_storage_dll(*, optional: bool = False) -> ctypes.CDLL | None:
    """Load and cache the native storage accelerator.

    Kube's storage runtime always uses the native storage library so the
    packed-cell engine always uses the native hot path instead of silently
    falling back to a slower Python implementation.
    """
    global _storage_dll_cache
    if _storage_dll_cache is not False:
        return _storage_dll_cache

    path = _find_library_path(_storage_lib_names())
    if path is None:
        raise FileNotFoundError(
            f"{_storage_lib_names()[0]} not found in {Path(__file__).parent}\n"
            "Build it first:\n"
            "  python scripts/build.py\n"
            "\n"
            "Windows (MinGW64):  python scripts/build.py --generator \"MinGW Makefiles\"\n"
            "Linux:              python3 scripts/build.py\n"
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
