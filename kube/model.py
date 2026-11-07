"""Core storage model for Kube.

This module exists to define the canonical `.kube` file format, the in-memory
representation used by Python code, and the conversion paths between packed
storage, raw byte streams, and Polars tables.

The key pieces are:

- `PackedBuffer`: a thin abstraction over bytearray or mmap-backed storage
- `Cell`: the logical six-face value object
- `Kube`: the main storage, IO, analytics, and viewer-facing API
"""

from __future__ import annotations

import ctypes
import io
import json
import math
import mmap
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import polars as pl

from kube._dll import load_storage_dll


MAGIC = b"KUBE"
FORMAT_VERSION = 1
HEADER_SIZE = 4096
HEADER_STRUCT = struct.Struct("<4sHHQQQIIQQQI")

TYPE_INFO = {
    1: {"code": "B", "name": "UInt8"},
    2: {"code": "H", "name": "UInt16"},
    4: {"code": "I", "name": "UInt32"},
    8: {"code": "Q", "name": "UInt64"},
}


class _StorageAccelerator:
    """Thin adapter around the optional native storage hot-path library.

    This private helper exists so the rest of the model code can call a small,
    Pythonic interface instead of reaching into raw `ctypes` symbols.
    """

    def __init__(self, library):
        self._library = library
        self._get_packed = library.kube_storage_get_packed
        self._set_packed = library.kube_storage_set_packed
        self._set_packed_batch = library.kube_storage_set_packed_batch
        self._get_face = library.kube_storage_get_face
        self._get_faces_batch = library.kube_storage_get_faces_batch
        self._set_face = library.kube_storage_set_face
        self._set_faces_batch = library.kube_storage_set_faces_batch
        self._zero_bytes = library.kube_storage_zero_bytes

    def get_packed(self, address: int, itemsize: int, index: int) -> int:
        """Read one packed value through the native accelerator."""
        return int(self._get_packed(address, itemsize, index))

    def set_packed(self, address: int, itemsize: int, index: int, value: int) -> None:
        """Write one packed value through the native accelerator."""
        self._set_packed(address, itemsize, index, int(value))

    def set_packed_batch(
        self,
        address: int,
        itemsize: int,
        start_index: int,
        values,
        count: int,
    ) -> None:
        """Write a contiguous packed-value range in one native call."""
        self._set_packed_batch(address, itemsize, start_index, values, count)

    def get_face(
        self,
        address: int,
        itemsize: int,
        index: int,
        bits_per_face: int,
        face_idx: int,
    ) -> int:
        """Read one decoded face through the native accelerator."""
        return int(self._get_face(address, itemsize, index, bits_per_face, face_idx))

    def set_face(
        self,
        address: int,
        itemsize: int,
        index: int,
        bits_per_face: int,
        face_idx: int,
        value: int,
    ) -> None:
        """Write one decoded face through the native accelerator."""
        self._set_face(address, itemsize, index, bits_per_face, face_idx, int(value))

    def zero_bytes(self, address: int, byte_offset: int, byte_count: int) -> None:
        """Clear a contiguous byte range through the native accelerator."""
        self._zero_bytes(address, byte_offset, byte_count)

    def get_faces_batch(
        self,
        address: int,
        itemsize: int,
        indices,
        face_indices,
        count: int,
        bits_per_face: int,
        out_values,
    ) -> None:
        """Decode many faces in one native call to amortize bridge overhead."""
        self._get_faces_batch(
            address,
            itemsize,
            indices,
            face_indices,
            count,
            bits_per_face,
            out_values,
        )

    def set_faces_batch(
        self,
        address: int,
        itemsize: int,
        indices,
        face_indices,
        values,
        count: int,
        bits_per_face: int,
    ) -> None:
        """Encode many face writes in one native call to amortize bridge overhead."""
        self._set_faces_batch(
            address,
            itemsize,
            indices,
            face_indices,
            values,
            count,
            bits_per_face,
        )


@dataclass(slots=True)
class CompiledFaceBatch:
    """Precomputed random face operations for repeated batch execution.

    This helper exists because batched native calls only pay off when Python
    avoids rebuilding coordinate and ctypes-array state for every timing loop.

    Attributes:
        indices: Flat cell indices as a contiguous native-compatible array.
        faces: Face indices as a contiguous native-compatible array.
        values: Face-state payload values aligned with `indices`/`faces`.
        count: Operation count cached so callers can skip repeated `len(...)`.
    """

    indices: ctypes.Array
    faces: ctypes.Array
    values: ctypes.Array
    count: int

    def __repr__(self) -> str:
        return f"CompiledFaceBatch(count={self.count:,})"


def _load_storage_accelerator() -> _StorageAccelerator:
    """Load the required native storage engine once for the whole process."""
    return _StorageAccelerator(load_storage_dll(optional=False))


_STORAGE_ACCELERATOR = _load_storage_accelerator()


def _validate_states(states: int) -> None:
    """Enforce the face-state invariant used throughout the storage format.

    Kube packs face values by bit width, so `states` must be a power of two.
    This helper exists to keep that rule centralized and consistently checked.
    """
    if states < 2 or (states & (states - 1)) != 0:
        raise ValueError(f"states must be a power of 2 >= 2, got {states}")


def _itemsize_for_states(states: int) -> int:
    """Choose the smallest integer width that can hold one packed cell.

    This function exists because each cell stores six faces in one packed
    integer, and the correct on-disk/in-memory dtype depends on `states`.
    """
    bits = 6 * int(math.log2(states))
    if bits <= 8:
        return 1
    if bits <= 16:
        return 2
    if bits <= 32:
        return 4
    return 8


def _dtype_name(itemsize: int) -> str:
    """Translate a packed item width into a human-readable dtype label."""
    return TYPE_INFO[itemsize]["name"]


def _coerce_shape(shape) -> tuple[int, int, int]:
    """Normalize shape input into a strict `(W, H, D)` tuple.

    The helper exists so callers can provide either one scalar side length or
    an explicit three-dimensional shape.
    """
    if isinstance(shape, int):
        shape = (shape, shape, shape)
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError("shape must be an int or a (W, H, D) tuple")
    dims = tuple(int(value) for value in shape)
    if any(value <= 0 for value in dims):
        raise ValueError(f"shape values must be positive, got {dims}")
    return dims


def _shape_for_bytes(byte_count: int, states: int) -> tuple[int, int, int]:
    """Estimate a cube shape large enough to hold a byte payload.

    This exists for convenience constructors such as `from_bytes(...)` and
    `import_file(...)` when the caller does not specify an explicit target
    shape.
    """
    bits_per_cell = 6 * int(math.log2(states))
    cells_needed = max(1, math.ceil(byte_count * 8 / bits_per_cell))
    side = max(1, math.ceil(cells_needed ** (1 / 3)))
    while side ** 3 < cells_needed:
        side += 1
    return (side, side, side)


def _normalize_bounds(
    bounds, shape: tuple[int, int, int]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Validate and normalize a region-bounds triple.

    Many table and iteration APIs accept `(start, stop)` bounds per axis. This
    helper exists to keep that validation logic in one place.
    """
    if len(bounds) != 3:
        raise ValueError("bounds must contain three (start, stop) pairs")
    normalized = []
    for axis, (start, stop) in enumerate(bounds):
        start = int(start)
        stop = int(stop)
        limit = shape[axis]
        if start < 0 or stop < 0 or start > stop or stop > limit:
            raise ValueError(f"invalid bounds {bounds} for shape {shape}")
        normalized.append((start, stop))
    return tuple(normalized)


def _normalize_axis(axis: int | str) -> int:
    """Convert axis names like `'x'` into the integer axis convention.

    The method exists because the public API allows both numeric and string
    axis references for readability.
    """
    if isinstance(axis, str):
        axis_name = axis.strip().lower()
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis_name not in axis_map:
            raise ValueError("axis must be one of 'x', 'y', or 'z'")
        return axis_map[axis_name]
    axis = int(axis)
    if axis < 0 or axis > 2:
        raise ValueError("axis must be 0, 1, or 2")
    return axis


class PackedBuffer:
    """Typed flat storage over either in-memory bytes or a mapped file.

    This class exists so higher-level code can read and write packed cell
    values without caring whether the underlying storage is a Python bytearray
    or an `mmap`.

    Attributes:
        shape: Logical `(W, H, D)` shape represented by the flat storage.
        itemsize: Bytes per packed cell.
        offset: Byte offset into the backing object where the payload begins.
        code: `memoryview.cast(...)` type code for the selected integer width.
        name: Human-readable dtype name used in metadata and inspection output.
        count: Total number of packed cells.
        nbytes: Total payload bytes exposed by the buffer.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, int, int],
        itemsize: int,
        backing,
        offset: int = 0,
        file_handle=None,
    ):
        self.shape = shape
        self.itemsize = itemsize
        self.offset = offset
        self.code = TYPE_INFO[itemsize]["code"]
        self.name = TYPE_INFO[itemsize]["name"]
        self.count = shape[0] * shape[1] * shape[2]
        self.nbytes = self.count * itemsize
        self._backing = backing
        self._file_handle = file_handle
        self._full_view = memoryview(backing)
        self._data_view = self._full_view[offset:offset + self.nbytes].cast(self.code)
        self._byte_view = self._full_view[offset:offset + self.nbytes]
        self._is_mapped = isinstance(backing, mmap.mmap)
        self._address_cache = None

    @property
    def is_mapped(self) -> bool:
        """Report whether the payload is backed by an `mmap` object."""
        return self._is_mapped

    def get_flat(self, index: int) -> int:
        """Read one packed cell by flat index.

        This method exists because the rest of the storage layer reasons in
        terms of packed integers, not raw bytes.
        """
        return int(self._data_view[index])

    def set_flat(self, index: int, value: int) -> None:
        """Write one packed cell by flat index."""
        self._data_view[index] = int(value)

    def clear(self, *, accelerator: _StorageAccelerator | None = None) -> None:
        """Zero the entire payload.

        This exists for initialization and full dataset reset workflows.
        """
        self.clear_from(0, accelerator=accelerator)

    def clear_from(self, start: int, *, accelerator: _StorageAccelerator | None = None) -> None:
        """Zero the payload from one flat index to the end.

        The method exists so imports can clear trailing unused cells after a
        shorter logical payload has been written.
        """
        start = int(start)
        if start <= 0:
            start = 0
        if start >= self.count:
            return
        byte_offset = start * self.itemsize
        byte_count = self.nbytes - byte_offset
        address = self.try_address()
        if address is not None:
            if accelerator is not None:
                accelerator.zero_bytes(address, byte_offset, byte_count)
                return
            ctypes.memset(address + byte_offset, 0, byte_count)
            return
        for index in range(start, self.count):
            self._data_view[index] = 0

    def address(self) -> int:
        """Expose the payload address for the native renderer bridge."""
        address = self.try_address()
        if address is None:
            raise BufferError("packed buffer does not expose a writable native address")
        return address

    def try_address(self) -> int | None:
        """Return the payload address when the backing object exports one.

        The storage accelerator and renderer both benefit from direct memory
        access, but some buffer implementations may refuse `from_buffer(...)`.
        This helper keeps that capability optional instead of making every
        caller handle `TypeError` or `BufferError`.
        """
        if self._address_cache is not None:
            return self._address_cache
        try:
            self._address_cache = ctypes.addressof(
                ctypes.c_char.from_buffer(self._backing, self.offset)
            )
        except (TypeError, BufferError):
            self._address_cache = None
        return self._address_cache

    def flush(self) -> None:
        """Flush the payload when it is backed by a mapped file."""
        if self._is_mapped:
            self._backing.flush()

    def close(self) -> None:
        """Release views and close the underlying mapping/file if needed."""
        self._data_view.release()
        self._byte_view.release()
        self._full_view.release()
        if self._is_mapped:
            self._backing.close()
        if self._file_handle is not None:
            self._file_handle.close()

    def __repr__(self) -> str:
        mode = "mapped" if self._is_mapped else "memory"
        return (
            f"PackedBuffer(shape={self.shape}, itemsize={self.itemsize}, "
            f"nbytes={self.nbytes:,}, mode={mode!r})"
        )


def _pack_stream_to_values(data: bytes, bits_per_face: int) -> list[int]:
    """Pack a logical byte stream into Kube packed-cell integers.

    This helper exists so arbitrary byte payloads can be stored in the Kube
    face-state format without first decoding them into explicit cells.
    """
    bits_per_cell = 6 * bits_per_face
    mask = (1 << bits_per_face) - 1
    bit_buffer = 0
    bits_in_buffer = 0
    packed_values: list[int] = []

    for byte in data:
        bit_buffer = (bit_buffer << 8) | byte
        bits_in_buffer += 8
        while bits_in_buffer >= bits_per_cell:
            shift = bits_in_buffer - bits_per_cell
            chunk = bit_buffer >> shift
            packed = 0
            for face_index in range(6):
                stream_shift = bits_per_cell - ((face_index + 1) * bits_per_face)
                face_value = (chunk >> stream_shift) & mask
                packed |= face_value << (face_index * bits_per_face)
            packed_values.append(packed)
            bit_buffer &= (1 << shift) - 1 if shift > 0 else 0
            bits_in_buffer = shift

    if bits_in_buffer:
        chunk = bit_buffer << (bits_per_cell - bits_in_buffer)
        packed = 0
        for face_index in range(6):
            stream_shift = bits_per_cell - ((face_index + 1) * bits_per_face)
            face_value = (chunk >> stream_shift) & mask
            packed |= face_value << (face_index * bits_per_face)
        packed_values.append(packed)

    return packed_values


def _unpack_values_to_bytes(
    values: list[int],
    bits_per_face: int,
    byte_limit: int,
) -> bytes:
    """Reconstruct a byte stream from packed cell values.

    The function exists for export and decode operations that need to recover
    the original logical byte payload.
    """
    bits_per_cell = 6 * bits_per_face
    mask = (1 << bits_per_face) - 1
    bit_buffer = 0
    bits_in_buffer = 0
    output = bytearray()

    for packed in values:
        chunk = 0
        for face_index in range(6):
            face_value = (int(packed) >> (face_index * bits_per_face)) & mask
            stream_shift = bits_per_cell - ((face_index + 1) * bits_per_face)
            chunk |= face_value << stream_shift
        bit_buffer = (bit_buffer << bits_per_cell) | chunk
        bits_in_buffer += bits_per_cell

        while bits_in_buffer >= 8 and len(output) < byte_limit:
            shift = bits_in_buffer - 8
            output.append((bit_buffer >> shift) & 0xFF)
            bit_buffer &= (1 << shift) - 1 if shift > 0 else 0
            bits_in_buffer = shift

        if len(output) >= byte_limit:
            break

    return bytes(output[:byte_limit])


def _packed_value_to_stream_chunk(packed: int, bits_per_face: int) -> tuple[int, int]:
    """Convert one packed cell into its stream-oriented bit chunk.

    This exists as a low-level helper for chunked decode/export code.
    """
    bits_per_cell = 6 * bits_per_face
    mask = (1 << bits_per_face) - 1
    chunk = 0
    for face_index in range(6):
        face_value = (int(packed) >> (face_index * bits_per_face)) & mask
        stream_shift = bits_per_cell - ((face_index + 1) * bits_per_face)
        chunk |= face_value << stream_shift
    return chunk, bits_per_cell


class Cell:
    """Logical six-face cell object.

    The class exists because many users and APIs want to reason in terms of
    explicit face values rather than packed integers.

    Attributes:
        states: Number of allowed values per face.
        faces: The six decoded face values in fixed index order.
    """

    def __init__(self, states: int, face_values=None):
        """Create one validated six-face cell."""
        _validate_states(states)
        if face_values is None:
            face_values = [0] * 6
        if len(face_values) != 6:
            raise ValueError("face_values must have exactly 6 entries")
        self.states = states
        self.faces = [int(value) for value in face_values]
        for value in self.faces:
            if value < 0 or value >= states:
                raise ValueError(f"face value {value} out of range [0, {states - 1}]")

    @property
    def bits_per_face(self) -> int:
        """Return the bit width needed to represent one face."""
        return int(math.log2(self.states))

    def get_face(self, index: int) -> int:
        """Read one face value by index."""
        return self.faces[index]

    def set_face(self, index: int, value: int) -> None:
        """Update one face value after validating its range."""
        value = int(value)
        if value < 0 or value >= self.states:
            raise ValueError(f"face value {value} out of range [0, {self.states - 1}]")
        self.faces[index] = value

    def to_packed(self) -> int:
        """Pack the six face values into one storage integer."""
        packed = 0
        bits_per_face = self.bits_per_face
        for index, value in enumerate(self.faces):
            packed |= value << (index * bits_per_face)
        return packed

    @classmethod
    def from_packed(cls, packed: int, states: int) -> "Cell":
        """Decode a packed cell integer into a `Cell` instance."""
        bits_per_face = int(math.log2(states))
        mask = states - 1
        faces = [(int(packed) >> (index * bits_per_face)) & mask for index in range(6)]
        return cls(states, faces)

    def __repr__(self) -> str:
        return f"Cell(states={self.states}, faces={self.faces})"

    def __str__(self) -> str:
        face_str = " ".join(f"f{i}={v}" for i, v in enumerate(self.faces))
        return f"Cell(S={self.states} | {face_str})"


class Kube:
    """Primary cube storage object for Kube.

    The class exists to unify file-format handling, packed storage access,
    byte-stream import/export, table conversion, and renderer launching behind
    one object.

    Attributes:
        W, H, D: Grid dimensions.
        S: Number of states per face.
        B: Bits per face.
        bits_per_cell: Total packed bits for one cell.
        itemsize: Bytes per packed cell in storage.
        dtype: Human-readable name of the packed cell dtype.
        path: Backing file path for mapped kubes, if any.
        payload_offset: Byte offset where packed cells begin in a file.
        source_byte_count: Logical byte length of the imported payload.
        metadata: User and pipeline metadata stored in the file header.
        mode: Access mode such as in-memory, read-only, or read-write.
        data: The underlying `PackedBuffer`.
    """

    def __init__(
        self,
        W: int,
        H: int,
        D: int,
        states: int = 4,
        storage: PackedBuffer | None = None,
        *,
        path: str | Path | None = None,
        payload_offset: int = HEADER_SIZE,
        source_byte_count: int = 0,
        metadata: dict | None = None,
        mode: str = "memory",
    ):
        _validate_states(states)
        self.W = int(W)
        self.H = int(H)
        self.D = int(D)
        self.S = int(states)
        self.B = int(math.log2(states))
        self.bits_per_cell = 6 * self.B
        self.itemsize = _itemsize_for_states(states)
        self.dtype = _dtype_name(self.itemsize)
        self._x_stride = int(H) * int(D)
        self._y_stride = int(D)
        self._face_mask = self.S - 1
        self.path = Path(path) if path is not None else None
        self.payload_offset = int(payload_offset)
        self.source_byte_count = int(source_byte_count)
        self.metadata = dict(metadata or {})
        self.mode = mode

        if storage is None:
            storage = PackedBuffer(
                shape=(self.W, self.H, self.D),
                itemsize=self.itemsize,
                backing=bytearray(self.W * self.H * self.D * self.itemsize),
            )
        self.data = storage
        self._payload_address = self.data.try_address()
        if self._payload_address is None:
            raise RuntimeError(
                "Kube requires a native-addressable backing buffer because storage is now native-only"
            )
        self._storage_accel = _STORAGE_ACCELERATOR
        self._acceleration_mode = "native"

    @property
    def shape(self) -> tuple[int, int, int]:
        """Expose the grid dimensions as one tuple."""
        return (self.W, self.H, self.D)

    @property
    def face_columns(self) -> tuple[str, ...]:
        """Return the standard decoded face-column names.

        This exists so table-producing methods can stay consistent.
        """
        return tuple(f"face_{index}" for index in range(6))

    @property
    def is_mapped(self) -> bool:
        """Report whether storage is backed by a mapped file."""
        return self.data.is_mapped

    @property
    def cell_count(self) -> int:
        """Return the total number of cells in the grid."""
        return self.W * self.H * self.D

    def capacity_bytes(self) -> int:
        """Return the logical byte capacity of the cube payload."""
        return (self.cell_count * self.bits_per_cell) // 8

    @property
    def byte_length(self) -> int:
        """Return the logical byte length currently represented by the kube."""
        return self.source_byte_count or self.capacity_bytes()

    def full_bounds(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Return bounds covering the full cube."""
        return ((0, self.W), (0, self.H), (0, self.D))

    def slice_bounds(self, axis: int | str, index: int):
        """Return bounds describing one single-cell-thick axis slice.

        The method exists so slice-oriented analytics can reuse the same region
        APIs as full-cube analytics.
        """
        axis_index = _normalize_axis(axis)
        limits = (self.W, self.H, self.D)
        index = int(index)
        if index < 0 or index >= limits[axis_index]:
            raise IndexError(f"slice index {index} is out of bounds for axis {axis_index}")
        bounds = list(self.full_bounds())
        bounds[axis_index] = (index, index + 1)
        return tuple(bounds)

    def _ensure_writable(self) -> None:
        """Reject mutating operations when the kube is opened read-only."""
        if self.mode == "r":
            raise PermissionError("kube is opened in read-only mode")

    def _flat_index(self, x: int, y: int, z: int) -> int:
        """Translate `(x, y, z)` coordinates into flat packed-buffer index."""
        x = int(x)
        y = int(y)
        z = int(z)
        if x < 0 or y < 0 or z < 0 or x >= self.W or y >= self.H or z >= self.D:
            raise IndexError(f"cell {(x, y, z)} is out of bounds for shape {self.shape}")
        return x * self._x_stride + y * self._y_stride + z

    def memory_stats(self) -> dict:
        """Summarize packed-vs-logical storage characteristics.

        This method exists for debugging, reporting, and CLI inspection.
        """
        packed_bytes = self.data.nbytes
        unpacked_bytes = self.cell_count * 6
        return {
            "packed_bytes": packed_bytes,
            "unpacked_bytes": unpacked_bytes,
            "ratio": unpacked_bytes / packed_bytes if packed_bytes else 0.0,
            "capacity_bytes": self.capacity_bytes(),
            "dtype": self.dtype,
            "states_per_face": self.S,
            "bits_per_face": self.B,
            "grid_shape": self.shape,
            "storage": "mapped" if self.is_mapped else "memory",
            "acceleration": self._acceleration_mode,
            "path": str(self.path) if self.path else None,
            "source_byte_count": self.source_byte_count,
        }

    def inspect(self) -> dict:
        """Return a richer inspection payload including user metadata."""
        info = self.memory_stats()
        info["metadata"] = dict(self.metadata)
        return info

    @property
    def storage_label(self) -> str:
        """Return a short human-readable storage mode label."""
        return "mapped" if self.is_mapped else "memory"

    @property
    def file_label(self) -> str:
        """Return a short label suitable for UI display."""
        return self.path.name if self.path is not None else "in-memory kube"

    def _write_header(self) -> None:
        """Rewrite the fixed file header for mapped kube files.

        The header stores shape, format info, source byte length, and metadata,
        so mapped writes need this helper to keep on-disk metadata in sync.
        """
        if self.path is None:
            return
        metadata_blob = json.dumps(self.metadata, sort_keys=True).encode("utf-8")
        prefix_size = HEADER_STRUCT.size
        if prefix_size + len(metadata_blob) > HEADER_SIZE:
            raise ValueError("metadata does not fit in the fixed kube header")
        header = bytearray(HEADER_SIZE)
        header[:prefix_size] = HEADER_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION,
            0,
            self.W,
            self.H,
            self.D,
            self.S,
            self.itemsize,
            self.payload_offset,
            self.cell_count,
            self.source_byte_count,
            len(metadata_blob),
        )
        header[prefix_size:prefix_size + len(metadata_blob)] = metadata_blob
        with self.path.open("r+b") as handle:
            handle.seek(0)
            handle.write(header)

    def flush(self) -> None:
        """Flush payload and header changes to disk when writable."""
        if self.mode == "r":
            return
        self.data.flush()
        if self.is_mapped:
            self._write_header()

    def update_metadata(self, **metadata) -> None:
        """Merge metadata keys into the kube header state.

        This exists so pipelines can attach provenance without re-creating the
        file.
        """
        self._ensure_writable()
        self.metadata.update(metadata)
        if self.is_mapped:
            self._write_header()

    def close(self) -> None:
        """Close underlying storage resources."""
        self.data.close()

    def __enter__(self) -> "Kube":
        """Support `with` statements for automatic cleanup."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close storage at the end of a context-manager block."""
        self.close()

    @classmethod
    def create(
        cls,
        path: str | Path,
        shape,
        *,
        states: int = 4,
        mode: str = "w+",
        overwrite: bool = False,
        source_byte_count: int = 0,
        metadata: dict | None = None,
    ) -> "Kube":
        """Create a new mapped `.kube` file on disk.

        This classmethod exists for the persistent storage case where a kube
        should be memory-mapped and reusable across processes or sessions.
        """
        if mode not in {"w+", "r+"}:
            raise ValueError("create mode must be 'w+' or 'r+'")
        W, H, D = _coerce_shape(shape)
        itemsize = _itemsize_for_states(states)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        total_bytes = HEADER_SIZE + (W * H * D * itemsize)
        create_mode = "wb" if overwrite else "xb"
        with path.open(create_mode) as handle:
            if total_bytes > 0:
                handle.seek(total_bytes - 1)
                handle.write(b"\0")

        file_handle = path.open("r+b")
        mapped = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_WRITE)
        storage = PackedBuffer(
            shape=(W, H, D),
            itemsize=itemsize,
            backing=mapped,
            offset=HEADER_SIZE,
            file_handle=file_handle,
        )
        kube = cls(
            W,
            H,
            D,
            states,
            storage=storage,
            path=path,
            payload_offset=HEADER_SIZE,
            source_byte_count=source_byte_count,
            metadata=metadata,
            mode=mode,
        )
        kube.flush()
        return kube

    @classmethod
    def open(cls, path: str | Path, mode: str = "r+") -> "Kube":
        """Open an existing `.kube` file and map its payload.

        The method exists as the canonical entry point for disk-backed kube
        storage.
        """
        path = Path(path)
        with path.open("rb") as handle:
            header = handle.read(HEADER_SIZE)
        (
            magic,
            version,
            _reserved,
            W,
            H,
            D,
            states,
            itemsize,
            payload_offset,
            cell_count,
            source_byte_count,
            metadata_len,
        ) = HEADER_STRUCT.unpack(header[:HEADER_STRUCT.size])
        if magic != MAGIC:
            raise ValueError(f"{path} is not a kube file")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported kube version {version}")
        if itemsize not in TYPE_INFO:
            raise ValueError(f"unsupported itemsize {itemsize} in {path}")
        if cell_count != W * H * D:
            raise ValueError(f"corrupt kube header in {path}")
        metadata_blob = header[HEADER_STRUCT.size:HEADER_STRUCT.size + metadata_len]
        metadata = json.loads(metadata_blob.decode("utf-8")) if metadata_blob else {}
        file_mode = "r+b" if mode != "r" else "rb"
        file_handle = path.open(file_mode)
        access = mmap.ACCESS_COPY if mode == "r" else mmap.ACCESS_WRITE
        mapped = mmap.mmap(file_handle.fileno(), 0, access=access)
        storage = PackedBuffer(
            shape=(W, H, D),
            itemsize=itemsize,
            backing=mapped,
            offset=payload_offset,
            file_handle=file_handle,
        )
        return cls(
            W,
            H,
            D,
            states,
            storage=storage,
            path=path,
            payload_offset=payload_offset,
            source_byte_count=source_byte_count,
            metadata=metadata,
            mode=mode,
        )

    @classmethod
    def from_bytes(cls, data: bytes, *, states: int = 4, shape=None) -> "Kube":
        """Create an in-memory kube from raw bytes."""
        if shape is None:
            shape = _shape_for_bytes(len(data), states)
        W, H, D = _coerce_shape(shape)
        kube = cls(W, H, D, states)
        kube.encode(data)
        return kube

    @classmethod
    def from_file(cls, path: str | Path, *, states: int = 4, shape=None) -> "Kube":
        """Read a file eagerly and store it in an in-memory kube."""
        with Path(path).open("rb") as handle:
            return cls.from_bytes(handle.read(), states=states, shape=shape)

    @classmethod
    def from_polars(
        cls,
        frame: pl.DataFrame,
        *,
        shape,
        states: int = 4,
        path: str | Path | None = None,
        metadata: dict | None = None,
        x_col: str = "x",
        y_col: str = "y",
        z_col: str = "z",
        packed_col: str = "packed",
        face_columns: tuple[str, str, str, str, str, str] = (
            "face_0",
            "face_1",
            "face_2",
            "face_3",
            "face_4",
            "face_5",
        ),
    ) -> "Kube":
        """Build a kube from a Polars frame of packed or decoded values.

        This method exists to close the loop with table-first workflows where
        data has already been transformed in Polars.
        """
        W, H, D = _coerce_shape(shape)
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("frame must be a polars.DataFrame")

        available = set(frame.columns)
        required_coords = {x_col, y_col, z_col}
        if not required_coords.issubset(available):
            raise ValueError(f"frame must include coordinate columns {sorted(required_coords)}")

        use_packed = packed_col in available
        use_faces = all(name in available for name in face_columns)
        if not use_packed and not use_faces:
            raise ValueError(
                f"frame must include either '{packed_col}' or all face columns {list(face_columns)}"
            )

        kube = (
            cls.create(path, (W, H, D), states=states, metadata=metadata)
            if path is not None
            else cls(W, H, D, states=states, metadata=metadata)
        )
        select_columns = [x_col, y_col, z_col, packed_col] if use_packed else [x_col, y_col, z_col, *face_columns]
        for row in frame.select(select_columns).iter_rows(named=True):
            x = int(row[x_col])
            y = int(row[y_col])
            z = int(row[z_col])
            if use_packed:
                kube.data.set_flat(kube._flat_index(x, y, z), int(row[packed_col]))
            else:
                kube.set_cell(x, y, z, Cell(states, [int(row[name]) for name in face_columns]))
        kube.flush()
        return kube

    @classmethod
    def import_file(
        cls,
        src_path: str | Path,
        dst_path: str | Path,
        *,
        states: int = 4,
        shape=None,
        chunk_size: int = 1 << 20,
    ) -> "Kube":
        """Stream a normal file into a new mapped kube container."""
        src_path = Path(src_path)
        byte_count = src_path.stat().st_size
        if shape is None:
            shape = _shape_for_bytes(byte_count, states)
        metadata = {"source_name": src_path.name}
        kube = cls.create(
            dst_path,
            shape,
            states=states,
            source_byte_count=byte_count,
            metadata=metadata,
        )
        kube._stream_import(src_path, chunk_size=chunk_size)
        kube.flush()
        return kube

    def encode(self, data: bytes) -> None:
        """Replace the kube payload with a packed encoding of raw bytes."""
        self._ensure_writable()
        packed_values = _pack_stream_to_values(data, self.B)
        if len(packed_values) > self.cell_count:
            raise ValueError(
                f"data too large: {len(data)} bytes > capacity {self.capacity_bytes()} bytes"
            )
        self._write_packed_values(0, packed_values)
        self.data.clear_from(len(packed_values), accelerator=self._storage_accel)
        self.source_byte_count = len(data)
        self.flush()

    def _stream_import(self, src_path: Path, *, chunk_size: int) -> None:
        """Import a source file incrementally instead of reading it all at once.

        This helper exists so very large files can be loaded into mapped kube
        storage without requiring the full source payload in RAM.
        """
        carry = bytearray()
        offset = 0
        bits_per_cell = self.bits_per_cell
        block_bytes = max(1, bits_per_cell // math.gcd(bits_per_cell, 8))
        cells_per_block = (block_bytes * 8) // bits_per_cell

        with src_path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                carry.extend(chunk)
                usable = (len(carry) // block_bytes) * block_bytes
                if usable == 0:
                    continue
                packed_values = _pack_stream_to_values(bytes(carry[:usable]), self.B)
                if offset + len(packed_values) > self.cell_count:
                    raise ValueError("shape is too small for the imported file")
                self._write_packed_values(offset, packed_values)
                offset += len(packed_values)
                del carry[:usable]

        if carry:
            packed_values = _pack_stream_to_values(bytes(carry), self.B)
            if offset + len(packed_values) > self.cell_count:
                raise ValueError("shape is too small for the imported file")
            self._write_packed_values(offset, packed_values)
            offset += len(packed_values)

        self.data.clear_from(offset, accelerator=self._storage_accel)

    def _iter_byte_chunks(self, *, chunk_cells: int = 1 << 18) -> Iterator[bytes]:
        """Yield decoded byte chunks from the packed cell payload.

        The iterator exists to support streaming export and decode operations.
        """
        byte_limit = self.source_byte_count or self.capacity_bytes()
        bits_per_face = self.B
        bit_buffer = 0
        bits_in_buffer = 0
        produced = 0
        output = bytearray()

        for start in range(0, self.cell_count, chunk_cells):
            end = min(start + chunk_cells, self.cell_count)
            for index in range(start, end):
                if produced >= byte_limit:
                    break
                chunk, chunk_bits = _packed_value_to_stream_chunk(self.data.get_flat(index), bits_per_face)
                bit_buffer = (bit_buffer << chunk_bits) | chunk
                bits_in_buffer += chunk_bits

                while bits_in_buffer >= 8 and produced < byte_limit:
                    shift = bits_in_buffer - 8
                    output.append((bit_buffer >> shift) & 0xFF)
                    produced += 1
                    bit_buffer &= (1 << shift) - 1 if shift > 0 else 0
                    bits_in_buffer = shift

                if len(output) >= 65536:
                    yield bytes(output)
                    output.clear()
            if produced >= byte_limit:
                break

        if output:
            yield bytes(output)

    def export_file(self, dst_path: str | Path, *, chunk_cells: int = 1 << 18) -> None:
        """Write the logical byte payload back out to a file."""
        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with dst_path.open("wb") as handle:
            for chunk in self._iter_byte_chunks(chunk_cells=chunk_cells):
                handle.write(chunk)

    def read_byte_range(self, start: int = 0, length: int | None = None) -> bytes:
        """Read a logical byte range from the decoded payload.

        This exists for partial inspection and token access helpers.
        """
        start = int(start)
        if start < 0 or start > self.byte_length:
            raise ValueError(f"start must be in [0, {self.byte_length}]")
        if length is None:
            stop = self.byte_length
        else:
            length = int(length)
            if length < 0:
                raise ValueError("length must be non-negative")
            stop = min(self.byte_length, start + length)
        if stop <= start:
            return b""
        return self.to_bytes()[start:stop]

    def decode(self, *, chunk_cells: int = 1 << 18) -> bytes:
        """Decode the full logical payload into a byte string."""
        buffer = io.BytesIO()
        for chunk in self._iter_byte_chunks(chunk_cells=chunk_cells):
            buffer.write(chunk)
        return buffer.getvalue()

    def to_bytes(self) -> bytes:
        """Alias for `decode()` for callers that prefer the `to_*` naming style."""
        return self.decode()

    def to_file(self, path: str | Path) -> None:
        """Alias for `export_file()` for API symmetry."""
        self.export_file(path)

    def token_count(self, token_bytes: int = 4) -> int:
        """Return how many fixed-width tokens fit in the logical payload.

        This exists for language-model style use cases where the payload should
        be treated as token IDs instead of arbitrary bytes.
        """
        token_bytes = int(token_bytes)
        if token_bytes <= 0:
            raise ValueError("token_bytes must be positive")
        return self.byte_length // token_bytes

    def read_token_ids(
        self,
        *,
        token_bytes: int = 4,
        start: int = 0,
        count: int | None = None,
        signed: bool = False,
    ) -> list[int]:
        """Read token IDs from the logical payload.

        The method exists so `.kube` files can double as compact token storage
        for sharded dataset workflows.
        """
        token_bytes = int(token_bytes)
        start = int(start)
        if token_bytes not in {1, 2, 4, 8}:
            raise ValueError("token_bytes must be one of 1, 2, 4, or 8")
        if start < 0:
            raise ValueError("start must be non-negative")
        available = self.token_count(token_bytes)
        if start > available:
            raise ValueError(f"start must be <= available token count ({available})")
        if count is None:
            count = available - start
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        raw = self.read_byte_range(start * token_bytes, count * token_bytes)
        return [
            int.from_bytes(raw[offset:offset + token_bytes], "little", signed=signed)
            for offset in range(0, len(raw), token_bytes)
        ]

    def encode_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        token_bytes: int = 4,
        signed: bool = False,
    ) -> None:
        """Encode a token sequence into the kube payload.

        This helper exists as the inverse of `read_token_ids(...)` for
        dataset-ingest code.
        """
        token_bytes = int(token_bytes)
        if token_bytes not in {1, 2, 4, 8}:
            raise ValueError("token_bytes must be one of 1, 2, 4, or 8")
        data = bytearray()
        for token_id in token_ids:
            data.extend(int(token_id).to_bytes(token_bytes, "little", signed=signed))
        self.metadata["token_bytes"] = token_bytes
        self.metadata["token_count"] = len(token_ids)
        self.encode(bytes(data))

    def buffer_address(self) -> int:
        """Expose the raw payload address for the native renderer."""
        return self.data.address()

    def _write_packed_values(self, start_index: int, packed_values: Sequence[int]) -> None:
        """Write a contiguous run of packed cell values through the native engine.

        This helper exists so encode and dataset-ingest paths can batch packed
        writes instead of crossing the Python/native bridge once per cell.
        """
        start_index = int(start_index)
        count = len(packed_values)
        if count == 0:
            return
        if start_index < 0 or start_index + count > self.cell_count:
            raise ValueError("packed write range exceeds kube capacity")
        values_type = ctypes.c_uint64 * count
        values = values_type(*(int(value) for value in packed_values))
        self._storage_accel.set_packed_batch(
            self._payload_address,
            self.itemsize,
            start_index,
            values,
            count,
        )

    def compile_face_batch(
        self,
        ops: Sequence[tuple[int, int, int, int, int]],
    ) -> CompiledFaceBatch:
        """Precompute a batch of face operations for repeated execution.

        This method exists so higher-level random-IO workloads can convert
        `(x, y, z, face, value)` tuples into flat native-friendly arrays once,
        then reuse that representation across many timing or processing loops.
        """
        count = len(ops)
        indices_type = ctypes.c_size_t * count
        faces_type = ctypes.c_ubyte * count
        values_type = ctypes.c_uint32 * count
        indices = indices_type()
        faces = faces_type()
        values = values_type()

        for offset, (x, y, z, face_idx, value) in enumerate(ops):
            face_idx = int(face_idx)
            value = int(value)
            if face_idx < 0 or face_idx >= 6:
                raise ValueError("face_idx must be in [0, 5]")
            if value < 0 or value >= self.S:
                raise ValueError(f"value {value} out of range [0, {self.S - 1}]")
            indices[offset] = self._flat_index(x, y, z)
            faces[offset] = face_idx
            values[offset] = value
        return CompiledFaceBatch(indices=indices, faces=faces, values=values, count=count)

    def get_packed(self, x: int, y: int, z: int) -> int:
        """Return the packed integer stored for one cell."""
        index = self._flat_index(x, y, z)
        return self._storage_accel.get_packed(self._payload_address, self.itemsize, index)

    def get_face(self, x: int, y: int, z: int, face_idx: int) -> int:
        """Read one decoded face value from a cell."""
        face_idx = int(face_idx)
        if face_idx < 0 or face_idx >= 6:
            raise ValueError("face_idx must be in [0, 5]")
        index = self._flat_index(x, y, z)
        return self._storage_accel.get_face(
            self._payload_address,
            self.itemsize,
            index,
            self.B,
            face_idx,
        )

    def get_faces_batch(
        self,
        ops: Sequence[tuple[int, int, int, int, int]] | CompiledFaceBatch,
    ) -> list[int]:
        """Read many decoded face values in one logical call.

        The method exists because random inspection and benchmark workloads are
        often bottlenecked more by Python call overhead than by bit packing
        itself. The native path can service a whole batch with one bridge call.
        """
        compiled = ops if isinstance(ops, CompiledFaceBatch) else self.compile_face_batch(ops)
        if compiled.count == 0:
            return []
        out_type = ctypes.c_uint32 * compiled.count
        out_values = out_type()
        self._storage_accel.get_faces_batch(
            self._payload_address,
            self.itemsize,
            compiled.indices,
            compiled.faces,
            compiled.count,
            self.B,
            out_values,
        )
        return [int(out_values[index]) for index in range(compiled.count)]

    def set_face(self, x: int, y: int, z: int, face_idx: int, value: int) -> None:
        """Update one face value in-place without unpacking the full cube.

        This method exists because random face-level writes are one of Kube's
        core design goals.
        """
        self._ensure_writable()
        value = int(value)
        if value < 0 or value >= self.S:
            raise ValueError(f"value {value} out of range [0, {self.S - 1}]")
        if face_idx < 0 or face_idx >= 6:
            raise ValueError("face_idx must be in [0, 5]")
        index = self._flat_index(x, y, z)
        self._storage_accel.set_face(
            self._payload_address,
            self.itemsize,
            index,
            self.B,
            face_idx,
            value,
        )

    def set_faces_batch(
        self,
        ops: Sequence[tuple[int, int, int, int, int]] | CompiledFaceBatch,
    ) -> None:
        """Apply many random face writes in one logical call.

        This method exists so sparse-edit workloads can batch through the
        native accelerator and amortize the Python-to-C transition cost.
        """
        self._ensure_writable()
        compiled = ops if isinstance(ops, CompiledFaceBatch) else self.compile_face_batch(ops)
        if compiled.count == 0:
            return
        self._storage_accel.set_faces_batch(
            self._payload_address,
            self.itemsize,
            compiled.indices,
            compiled.faces,
            compiled.values,
            compiled.count,
            self.B,
        )

    def get_cell(self, x: int, y: int, z: int) -> Cell:
        """Decode and return one logical cell object."""
        return Cell.from_packed(self.data.get_flat(self._flat_index(x, y, z)), self.S)

    def set_cell(self, x: int, y: int, z: int, cell: Cell) -> None:
        """Replace one cell from a decoded `Cell` instance."""
        self._ensure_writable()
        if cell.states != self.S:
            raise ValueError("cell state count does not match kube")
        index = self._flat_index(x, y, z)
        packed = cell.to_packed()
        self._storage_accel.set_packed(self._payload_address, self.itemsize, index, packed)

    def _iter_cells(
        self,
        bounds=None,
    ) -> Iterator[tuple[int, int, int, int, Cell]]:
        """Yield decoded cells over the full cube or a bounded region.

        This private generator exists as the common backbone for the table and
        analytics helpers.
        """
        (x0, x1), (y0, y1), (z0, z1) = (
            self.full_bounds() if bounds is None else _normalize_bounds(bounds, self.shape)
        )
        for x in range(x0, x1):
            for y in range(y0, y1):
                for z in range(z0, z1):
                    packed = self.data.get_flat(self._flat_index(x, y, z))
                    yield x, y, z, packed, Cell.from_packed(packed, self.S)

    def to_polars(
        self,
        bounds=None,
        *,
        include_coords: bool = True,
        include_packed: bool = True,
        include_faces: bool = True,
    ) -> pl.DataFrame:
        """Materialize cube contents as a Polars DataFrame.

        This method exists to make Kube easy to analyze with column-oriented
        data tools and ML preprocessing pipelines.
        """
        if not any((include_coords, include_packed, include_faces)):
            raise ValueError("at least one output group must be enabled")

        rows: dict[str, list[int]] = {}
        if include_coords:
            rows.update({"x": [], "y": [], "z": []})
        if include_packed:
            rows["packed"] = []
        if include_faces:
            for name in self.face_columns:
                rows[name] = []

        for x, y, z, packed, cell in self._iter_cells(bounds):
            if include_coords:
                rows["x"].append(x)
                rows["y"].append(y)
                rows["z"].append(z)
            if include_packed:
                rows["packed"].append(packed)
            if include_faces:
                for name, value in zip(self.face_columns, cell.faces):
                    rows[name].append(value)
        return pl.DataFrame(rows)

    def feature_frame(
        self,
        bounds=None,
        *,
        include_coords: bool = True,
        include_packed: bool = False,
    ) -> pl.DataFrame:
        """Return the face-centric feature table used by many ML workflows."""
        return self.to_polars(
            bounds,
            include_coords=include_coords,
            include_packed=include_packed,
            include_faces=True,
        )

    def supervised_frame(
        self,
        target_face: int,
        bounds=None,
        *,
        include_coords: bool = True,
        include_packed: bool = False,
        target_name: str = "target",
        drop_target_face: bool = True,
    ) -> pl.DataFrame:
        """Return a supervised-learning table with one face as the target.

        The method exists because a common training pattern is "predict face N
        from the remaining features".
        """
        target_face = int(target_face)
        if target_face < 0 or target_face >= 6:
            raise ValueError("target_face must be in [0, 5]")

        frame = self.feature_frame(
            bounds,
            include_coords=include_coords,
            include_packed=include_packed,
        )
        source_name = f"face_{target_face}"
        rename_map = {source_name: target_name}
        frame = frame.rename(rename_map)
        if drop_target_face:
            return frame
        restored = frame.rename({target_name: source_name})
        ordered = []
        if include_coords:
            ordered.extend(["x", "y", "z"])
        if include_packed:
            ordered.append("packed")
        ordered.extend(self.face_columns)
        ordered.append(target_name)
        return restored.with_columns(pl.col(source_name).alias(target_name)).select(ordered)

    def face_histogram(self, bounds=None, *, normalize: bool = False) -> pl.DataFrame:
        """Compute per-face value counts, optionally normalized.

        This exists for inspection, drift checks, and dataset balancing tasks.
        """
        counts = {(face_index, value): 0 for face_index in range(6) for value in range(self.S)}
        total = 0
        for _, _, _, _, cell in self._iter_cells(bounds):
            total += 1
            for face_index, value in enumerate(cell.faces):
                counts[(face_index, value)] += 1

        rows = {
            "face": [],
            "value": [],
            "count": [],
        }
        if normalize:
            rows["probability"] = []

        denominator = float(total) if total else 1.0
        for face_index in range(6):
            for value in range(self.S):
                count = counts[(face_index, value)]
                rows["face"].append(face_index)
                rows["value"].append(value)
                rows["count"].append(count)
                if normalize:
                    rows["probability"].append(count / denominator)
        return pl.DataFrame(rows)

    def read_region(self, bounds) -> pl.DataFrame:
        """Materialize one bounded region as a DataFrame."""
        return self.to_polars(bounds)

    def iter_region(self, bounds, tile_shape: tuple[int, int, int] = (32, 32, 32)):
        """Yield tiled subregions across a bounded area.

        The iterator exists for spatial workflows that need deterministic
        sub-cube traversal rather than simple row-count batching.
        """
        (x0, x1), (y0, y1), (z0, z1) = _normalize_bounds(bounds, self.shape)
        tx, ty, tz = tile_shape
        for x in range(x0, x1, tx):
            for y in range(y0, y1, ty):
                for z in range(z0, z1, tz):
                    region_bounds = (
                        (x, min(x + tx, x1)),
                        (y, min(y + ty, y1)),
                        (z, min(z + tz, z1)),
                    )
                    yield (x, y, z), self.read_region(region_bounds)

    def iter_batches(
        self,
        batch_size: int,
        bounds=None,
        *,
        include_coords: bool = True,
        include_packed: bool = True,
        include_faces: bool = True,
    ) -> Iterator[pl.DataFrame]:
        """Yield sequential DataFrame batches across the cube.

        This exists for preprocessing and training loops that should stream
        manageable chunks instead of materializing the whole table at once.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not any((include_coords, include_packed, include_faces)):
            raise ValueError("at least one output group must be enabled")

        rows: dict[str, list[int]] = {}
        if include_coords:
            rows.update({"x": [], "y": [], "z": []})
        if include_packed:
            rows["packed"] = []
        if include_faces:
            for name in self.face_columns:
                rows[name] = []

        size = 0
        for x, y, z, packed, cell in self._iter_cells(bounds):
            if include_coords:
                rows["x"].append(x)
                rows["y"].append(y)
                rows["z"].append(z)
            if include_packed:
                rows["packed"].append(packed)
            if include_faces:
                for name, value in zip(self.face_columns, cell.faces):
                    rows[name].append(value)
            size += 1
            if size >= batch_size:
                yield pl.DataFrame({name: values[:] for name, values in rows.items()})
                for values in rows.values():
                    values.clear()
                size = 0

        if size:
            yield pl.DataFrame({name: values[:] for name, values in rows.items()})

    def iter_supervised_batches(
        self,
        batch_size: int,
        target_face: int,
        bounds=None,
        *,
        include_coords: bool = True,
        include_packed: bool = False,
        target_name: str = "target",
        drop_target_face: bool = True,
    ) -> Iterator[pl.DataFrame]:
        """Yield supervised-learning batches with one face renamed as target."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        target_face = int(target_face)
        if target_face < 0 or target_face >= 6:
            raise ValueError("target_face must be in [0, 5]")

        for batch in self.iter_batches(
            batch_size,
            bounds,
            include_coords=include_coords,
            include_packed=include_packed,
            include_faces=True,
        ):
            source_name = f"face_{target_face}"
            renamed = batch.rename({source_name: target_name})
            if drop_target_face:
                yield renamed
            else:
                restored = renamed.rename({target_name: source_name})
                ordered = []
                if include_coords:
                    ordered.extend(["x", "y", "z"])
                if include_packed:
                    ordered.append("packed")
                ordered.extend(self.face_columns)
                ordered.append(target_name)
                yield restored.with_columns(pl.col(source_name).alias(target_name)).select(ordered)

    def __getitem__(self, pos) -> Cell:
        """Provide `kube[x, y, z]` syntax for reading one cell."""
        x, y, z = pos
        return self.get_cell(x, y, z)

    def __setitem__(self, pos, cell: Cell) -> None:
        """Provide `kube[x, y, z] = cell` syntax for writing one cell."""
        x, y, z = pos
        self.set_cell(x, y, z, cell)

    def __len__(self) -> int:
        """Return the number of cells for container-style APIs."""
        return self.cell_count

    def __repr__(self) -> str:
        """Summarize kube size and storage characteristics for debugging."""
        stats = self.memory_stats()
        return (
            f"Kube({self.W}x{self.H}x{self.D}, S={self.S}, "
            f"packed={stats['packed_bytes']:,}B, "
            f"logical={stats['unpacked_bytes']:,}B, "
            f"ratio={stats['ratio']:.2f}x, "
            f"storage={stats['storage']})"
        )

    def __str__(self) -> str:
        """Return a concise human-readable summary of this kube."""
        path_part = f", path={self.path.name!r}" if self.path is not None else ""
        return (
            f"Kube {self.W}×{self.H}×{self.D}  "
            f"S={self.S} states  {self.B}b/face  "
            f"{self.cell_count:,} cells  "
            f"{self.data.nbytes:,}B {self.dtype}{path_part}"
        )

    def view(
        self,
        *,
        read_only: bool = False,
        slice_axis: int | None = None,
        slice_index: int | None = None,
        lod: int = 1,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
    ) -> None:
        """Launch the localhost web viewer/editor for this kube.

        Starts a FastAPI/uvicorn server serving the WebGL viewer and blocks
        until the browser tab is closed (via `/api/shutdown`) or the process
        receives Ctrl+C. The method exists so inspection and editing stay a
        first-class part of the storage object rather than requiring callers
        to wire up the web server manually.
        """
        from kube.web.serve import serve

        serve(
            self,
            read_only=read_only,
            slice_axis=slice_axis,
            slice_index=slice_index,
            lod=lod,
            host=host,
            port=port,
            open_browser=open_browser,
        )

    # ------------------------------------------------------------------
    # Interop: Arrow / Parquet / NumPy / Pandas / sklearn
    # ------------------------------------------------------------------

    def schema(self, mode: str = "wide_faces") -> dict:
        """Return ``{column_name: dtype_str}`` for the given schema mode.

        The method exists so downstream tools can introspect expected column
        layouts without materialising any data.
        """
        from kube.schemas import schema_columns
        return schema_columns(mode, self.S)

    def to_arrow(self, bounds=None, *, schema_mode: str = "wide_faces"):
        """Convert the kube (or a bounded region) to a PyArrow Table.

        Routing: ``to_polars(bounds)`` → ``polars.DataFrame.to_arrow()``.
        Near-zero-copy because Polars uses Arrow-native memory internally.

        Args:
            bounds: Optional ``((x0,x1),(y0,y1),(z0,z1))`` region; ``None``
                means the full cube.
            schema_mode: One of ``"wide_faces"``, ``"long_faces"``,
                ``"cell_packed"``.  Determines which columns are exported.

        Returns:
            A ``pyarrow.Table`` with the requested column layout.
        """
        try:
            import pyarrow as pa  # noqa: F401
        except ImportError as exc:
            raise ImportError("pyarrow is required: pip install pyarrow") from exc

        from kube.schemas import (
            SCHEMA_CELL_PACKED,
            SCHEMA_LONG_FACES,
            SCHEMA_WIDE_FACES,
        )

        if schema_mode == SCHEMA_WIDE_FACES:
            frame = self.to_polars(bounds, include_coords=True, include_packed=False, include_faces=True)
        elif schema_mode == SCHEMA_CELL_PACKED:
            frame = self.to_polars(bounds, include_coords=True, include_packed=True, include_faces=False)
        elif schema_mode == SCHEMA_LONG_FACES:
            wide = self.to_polars(bounds, include_coords=True, include_packed=False, include_faces=True)
            frame = wide.unpivot(
                index=["x", "y", "z"],
                on=list(self.face_columns),
                variable_name="face",
                value_name="value",
            ).with_columns(
                pl.col("face").str.extract(r"(\d+)$").cast(pl.UInt8).alias("face")
            ).sort(["x", "y", "z", "face"])
        else:
            raise ValueError(f"unknown schema_mode {schema_mode!r}")

        return frame.to_arrow()

    def region_to_arrow(self, bounds):
        """Thin alias for ``to_arrow(bounds)``."""
        return self.to_arrow(bounds)

    @classmethod
    def from_arrow(
        cls,
        table,
        *,
        shape,
        states: int = 4,
        path=None,
        metadata: dict | None = None,
    ) -> "Kube":
        """Reconstruct a kube from a PyArrow Table.

        The classmethod exists to close the loop for Arrow-first pipelines.
        It delegates to ``from_polars`` via ``polars.from_arrow``.
        """
        try:
            pass  # pyarrow is imported implicitly by polars
        except ImportError:
            pass
        frame = pl.from_arrow(table)
        return cls.from_polars(frame, shape=shape, states=states, path=path, metadata=metadata)

    def to_numpy(self, bounds=None, *, packed: bool = False):
        """Convert the kube to a 2-D NumPy array.

        Args:
            bounds: Optional region bounds.  ``None`` means the full cube.
            packed: If ``True``, return a 1-D array of packed cell integers
                instead of a ``(N, 6)`` face matrix.

        Returns:
            ``np.ndarray`` of shape ``(N,)`` (packed) or ``(N, 6)`` (faces).
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required: pip install numpy") from exc

        if packed:
            frame = self.to_polars(bounds, include_coords=False, include_packed=True, include_faces=False)
            return frame["packed"].to_numpy()

        frame = self.to_polars(bounds, include_coords=False, include_packed=False, include_faces=True)
        return np.stack([frame[col].to_numpy() for col in self.face_columns], axis=1)

    def faces_to_numpy(self, order: str = "xyzf"):
        """Return a ``(W, H, D, 6)`` array of face values.

        The method iterates all cells in x,y,z order and builds a contiguous
        numpy array, then reshapes it.  Not zero-copy due to bit-packed storage.

        Args:
            order: Axis order string.  Only ``"xyzf"`` is supported currently.

        Returns:
            ``np.ndarray`` of shape ``(W, H, D, 6)``, dtype ``uint8`` when
            ``states <= 256``, ``uint16`` when ``states <= 65536``, otherwise
            ``uint32``.
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required: pip install numpy") from exc

        if self.S <= 256:
            dtype = np.uint8
        elif self.S <= 65536:
            dtype = np.uint16
        else:
            dtype = np.uint32

        arr = np.empty((self.W * self.H * self.D, 6), dtype=dtype)
        frame = self.to_polars(None, include_coords=False, include_packed=False, include_faces=True)
        for i, col in enumerate(self.face_columns):
            arr[:, i] = frame[col].to_numpy().astype(dtype)
        return arr.reshape(self.W, self.H, self.D, 6)

    def region_to_numpy(self, bounds):
        """Return a ``(N, 6)`` face array for the cells in *bounds*."""
        return self.to_numpy(bounds, packed=False)

    def __array__(self, dtype=None, copy=None):
        """Support ``np.array(kube)`` by delegating to ``faces_to_numpy()``.

        Note: this always involves a copy because cell data is bit-packed.
        The *copy* parameter is accepted for NumPy 2.0 compatibility but is
        ignored — a copy is always produced.
        """
        arr = self.faces_to_numpy()
        if dtype is not None:
            try:
                import numpy as np
            except ImportError as exc:
                raise ImportError("numpy is required: pip install numpy") from exc
            return arr.astype(dtype)
        return arr

    def to_pandas(self, bounds=None):
        """Convert the kube (or a region) to a Pandas DataFrame.

        Delegates to ``to_polars(...).to_pandas()``; requires ``pandas``.
        """
        return self.to_polars(bounds, include_coords=True, include_packed=False, include_faces=True).to_pandas()

    @classmethod
    def from_pandas(
        cls,
        df,
        *,
        shape,
        states: int = 4,
        path=None,
        metadata: dict | None = None,
    ) -> "Kube":
        """Build a kube from a Pandas DataFrame.

        Delegates to ``from_polars(pl.from_pandas(df))``.
        """
        frame = pl.from_pandas(df)
        return cls.from_polars(frame, shape=shape, states=states, path=path, metadata=metadata)

    def to_parquet(self, path, bounds=None, *, schema_mode: str = "wide_faces") -> Path:
        """Write the kube (or a region) to a Parquet file.

        Args:
            path: Destination file path.
            bounds: Optional region bounds.
            schema_mode: Column layout; see ``SCHEMA_*`` constants.

        Returns:
            The resolved ``Path`` of the written file.
        """
        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        table = self.to_arrow(bounds, schema_mode=schema_mode)
        import pyarrow.parquet as pq
        pq.write_table(table, str(dst))
        return dst

    def region_to_parquet(self, bounds, path) -> Path:
        """Thin alias for ``to_parquet(path, bounds)``."""
        return self.to_parquet(path, bounds)

    def to_sklearn_xy(self, target_face: int, bounds=None):
        """Return ``(X, y)`` numpy arrays ready for sklearn estimators.

        The method delegates to ``supervised_frame()`` and converts the
        resulting Polars columns to numpy.

        Args:
            target_face: Index (0–5) of the face to use as the label column.
            bounds: Optional region bounds.

        Returns:
            Tuple ``(X, y)`` where ``X`` has shape ``(N, 5)`` and ``y`` has
            shape ``(N,)``.
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required: pip install numpy") from exc

        frame = self.supervised_frame(
            target_face,
            bounds,
            include_coords=False,
            include_packed=False,
            target_name="target",
            drop_target_face=True,
        )
        feature_cols = [c for c in frame.columns if c != "target"]
        X = np.stack([frame[col].to_numpy() for col in feature_cols], axis=1)
        y = frame["target"].to_numpy()
        return X, y
