"""Backend fixtures for the Kube benchmark suite.

This module builds all in-memory objects (Kube grid, binary grid, numpy
arrays, compiled batches) that workload functions read from.  Everything is
constructed once and shared across workloads so allocation cost does not
contaminate timings.
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

from kube.model import Kube


class BinaryOccupancyGrid:
    """Tiny one-bit-per-cell baseline for comparison.

    This baseline represents the classic binary system: one undirected
    occupancy bit per cell, no face orientation, no multi-state measurements.
    """

    def __init__(self, width: int, height: int, depth: int) -> None:
        self.W = int(width)
        self.H = int(height)
        self.D = int(depth)
        self.cell_count = self.W * self.H * self.D
        self.data = bytearray((self.cell_count + 7) // 8)

    def _flat_index(self, x: int, y: int, z: int) -> int:
        return x * (self.H * self.D) + y * self.D + z

    def set_cell(self, x: int, y: int, z: int, value: int) -> None:
        index = self._flat_index(x, y, z)
        byte_index = index >> 3
        bit_index = index & 7
        mask = 1 << bit_index
        if value:
            self.data[byte_index] |= mask
        else:
            self.data[byte_index] &= 0xFF ^ mask

    def get_cell(self, x: int, y: int, z: int) -> int:
        index = self._flat_index(x, y, z)
        return (self.data[index >> 3] >> (index & 7)) & 1

    @property
    def nbytes(self) -> int:
        return len(self.data)


def build_fixtures(
    W: int,
    H: int,
    D: int,
    S: int,
    ops: list[tuple[int, int, int, int, int]],
    seed: int,
    *,
    tmp_dir: str | None = None,
) -> dict[str, Any]:
    """Build all in-memory fixtures shared across workloads.

    All expensive allocations (Kube, numpy arrays, compiled batches) happen
    here so workload timings only measure pure IO/compute, not setup.
    """
    kube_instance = Kube(W, H, D, states=S)
    compiled_batch = kube_instance.compile_face_batch(ops)

    # Populate the kube with the ops so reads have non-trivial data
    kube_instance.set_faces_batch(compiled_batch)

    # Coordinate and value arrays for numpy workloads
    xs = None
    ys = None
    zs = None
    face_idxs = None
    values = None
    numpy_arr = None

    if _HAS_NUMPY:
        xs = np.array([op[0] for op in ops], dtype=np.int32)
        ys = np.array([op[1] for op in ops], dtype=np.int32)
        zs = np.array([op[2] for op in ops], dtype=np.int32)
        face_idxs = np.array([op[3] for op in ops], dtype=np.int32)
        values = np.array([op[4] for op in ops], dtype=np.int32)
        # Full (W, H, D, 6) numpy face array for array-based workloads
        numpy_arr = kube_instance.faces_to_numpy()

    # Neighbour scan: for each op, gather all 6 face-direction neighbours
    _DIRS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    neighbor_ops: list[tuple[int, int, int, int, int]] = []
    for x, y, z, _f, _v in ops[:10_000]:
        for dx, dy, dz in _DIRS:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D:
                neighbor_ops.append((nx, ny, nz, 0, 0))
    n_neighbor_cells = len(neighbor_ops)
    compiled_neighbor_batch = kube_instance.compile_face_batch(neighbor_ops) if neighbor_ops else compiled_batch

    if _HAS_NUMPY and neighbor_ops:
        nb_xs = np.array([op[0] for op in neighbor_ops], dtype=np.int32)
        nb_ys = np.array([op[1] for op in neighbor_ops], dtype=np.int32)
        nb_zs = np.array([op[2] for op in neighbor_ops], dtype=np.int32)
    else:
        nb_xs = nb_ys = nb_zs = None

    # Region bounds: lower octant of the cube
    region_bounds = (
        (0, max(1, W // 2)),
        (0, max(1, H // 2)),
        (0, max(1, D // 2)),
    )

    # Binary grid for baseline comparisons
    binary_grid = BinaryOccupancyGrid(W, H, D)
    for x, y, z, _f, v in ops:
        binary_grid.set_cell(x, y, z, v & 1)

    # Temporary path for export workloads
    import tempfile
    _tmp = tempfile.mkdtemp(dir=tmp_dir)
    parquet_path = f"{_tmp}/bench_export.parquet"
    arrow_path = f"{_tmp}/bench_export.arrow"

    return {
        "W": W,
        "H": H,
        "D": D,
        "S": S,
        "ops": ops,
        "kube_instance": kube_instance,
        "compiled_batch": compiled_batch,
        "compiled_neighbor_batch": compiled_neighbor_batch,
        "n_neighbor_cells": n_neighbor_cells,
        "numpy_arr": numpy_arr,
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "face_idxs": face_idxs,
        "values": values,
        "nb_xs": nb_xs,
        "nb_ys": nb_ys,
        "nb_zs": nb_zs,
        "binary_grid": binary_grid,
        "region_bounds": region_bounds,
        "parquet_path": parquet_path,
        "arrow_path": arrow_path,
    }
