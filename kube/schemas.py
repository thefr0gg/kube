"""Schema mode constants and column definitions for Kube export formats.

This module exists to give downstream code stable string constants for the
schema mode names used in table-export and interop methods, without reaching
into implementation details of the storage model.
"""

from __future__ import annotations

SCHEMA_WIDE_FACES = "wide_faces"
"""N rows × (x, y, z, face_0 … face_5) — the default analytical layout."""

SCHEMA_LONG_FACES = "long_faces"
"""N×6 rows × (x, y, z, face, value) — useful for melted / tidy workflows."""

SCHEMA_CELL_PACKED = "cell_packed"
"""N rows × (x, y, z, packed) — compact single-integer representation."""

SCHEMA_SUPERVISED = "supervised"
"""wide_faces layout extended with a 'target' column for ML pipelines."""

SCHEMA_TOKEN_SEQUENCES = "token_sequences"
"""input_ids, target_ids, and metadata columns for LM dataset exports."""

KUBE_SCHEMA_VERSION = 1
"""Monotone schema format version, incremented on incompatible layout changes."""

_FACE_COLUMNS = tuple(f"face_{i}" for i in range(6))


def schema_columns(mode: str, states: int) -> dict[str, str]:
    """Return ``{column_name: dtype_str}`` for the requested schema mode.

    The function exists so code that generates or validates table layouts can
    stay independent of the concrete DataFrame implementation.

    Args:
        mode: One of the ``SCHEMA_*`` constants defined in this module.
        states: Number of distinct face values (e.g. 4, 256, 65536).

    Returns:
        Ordered mapping of column name to dtype string.

    Raises:
        ValueError: If *mode* is not a recognised schema constant.
    """
    if states <= 256:
        face_dtype = "UInt8"
    elif states <= 65536:
        face_dtype = "UInt16"
    else:
        face_dtype = "UInt32"

    coord_cols = {"x": "UInt32", "y": "UInt32", "z": "UInt32"}

    if mode == SCHEMA_WIDE_FACES:
        return {**coord_cols, **{col: face_dtype for col in _FACE_COLUMNS}}

    if mode == SCHEMA_LONG_FACES:
        return {**coord_cols, "face": "UInt8", "value": face_dtype}

    if mode == SCHEMA_CELL_PACKED:
        packed_dtype = "UInt8" if states ** 6 <= 256 else "UInt32"
        return {**coord_cols, "packed": packed_dtype}

    if mode == SCHEMA_SUPERVISED:
        return {**coord_cols, **{col: face_dtype for col in _FACE_COLUMNS}, "target": face_dtype}

    if mode == SCHEMA_TOKEN_SEQUENCES:
        return {
            "input_ids": "List(UInt32)",
            "target_ids": "List(UInt32)",
            "shard_index": "UInt32",
            "token_offset": "UInt64",
        }

    raise ValueError(
        f"unknown schema mode {mode!r}; expected one of "
        f"{SCHEMA_WIDE_FACES!r}, {SCHEMA_LONG_FACES!r}, {SCHEMA_CELL_PACKED!r}, "
        f"{SCHEMA_SUPERVISED!r}, {SCHEMA_TOKEN_SEQUENCES!r}"
    )
