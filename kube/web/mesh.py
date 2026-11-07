"""Shell/slice mesh construction for the web viewer.

Builds the same visible-face geometry the old native raylib editor built in
`src/kube_renderer.c`, but as vectorized numpy over a `(W, H, D, 6)` face-value
array instead of a C loop over the packed buffer. Output is a flat vertex/color/
index triplet ready to upload as a WebGL buffer.
"""

from __future__ import annotations

import numpy as np

# Per-face quad vertices, matching the old native FACE_VERTS table.
# Order: top(+y), bottom(-y), left(-x), right(+x), front(+z), back(-z).
_FACE_VERTS = np.array(
    [
        [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]],
        [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]],
        [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]],
        [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]],
        [[-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]],
    ],
    dtype=np.float32,
)

# (axis, is_min) per face index — which boundary makes that face shell-visible.
_FACE_BOUND = [(1, 0), (1, 1), (0, 1), (0, 0), (2, 0), (2, 1)]

_PALETTE_8 = np.array(
    [
        [40, 40, 40],
        [217, 60, 70],
        [70, 160, 217],
        [90, 200, 120],
        [230, 180, 60],
        [170, 90, 220],
        [230, 230, 230],
        [120, 120, 120],
    ],
    dtype=np.uint8,
)


def _color_for_states(values: np.ndarray, states: int) -> np.ndarray:
    """Map an array of face state values to RGB colors, mirroring `color_for_state`."""
    if states <= 8:
        return _PALETTE_8[values]
    hue = (values.astype(np.float32) / states) * 360.0
    return _hsv_to_rgb(hue, 0.85, 0.95)


def _hsv_to_rgb(hue: np.ndarray, saturation: float, value: float) -> np.ndarray:
    """Vectorized HSV->RGB for hue arrays in degrees; s/v are scalars."""
    h = (hue / 60.0) % 6
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = value * (1 - saturation)
    q = value * (1 - saturation * f)
    t = value * (1 - saturation * (1 - f))

    r = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [value, q, p, p, t, value],
        default=value,
    )
    g = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [t, value, value, q, p, p],
        default=value,
    )
    b = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [p, p, t, value, value, q],
        default=value,
    )
    return np.stack([r, g, b], axis=-1 if r.ndim else 0) * 255


def _sample_mask(coord: np.ndarray, limit: int, lod: int) -> np.ndarray:
    """Boolean LOD mask: always keep boundary coords, else sample every `lod`-th."""
    if lod <= 1:
        return np.ones_like(coord, dtype=bool)
    return (coord == 0) | (coord == limit) | (coord % lod == 0)


def build_shell_mesh(faces: np.ndarray, *, lod: int = 1, visible_faces: tuple[bool, ...] = (True,) * 6):
    """Build shell-mode geometry: only boundary faces of the grid.

    Args:
        faces: `(W, H, D, 6)` array of per-face state values (from `Kube.faces_to_numpy()`).
        lod: sampling stride; cells not on a stride boundary are skipped (edges always kept).
        visible_faces: per-face-index visibility toggle (camera-facing culling).

    Returns:
        dict with flat `positions` (float32 Nx3), `colors` (uint8 Nx3), `indices`
        (uint32 Mx3), and `states` for the client's color legend.
    """
    W, H, D, _ = faces.shape
    states = int(faces.max()) + 1 if faces.size else 2

    xs, ys, zs = np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing="ij")
    lod_x = _sample_mask(xs, W - 1, lod)
    lod_y = _sample_mask(ys, H - 1, lod)
    lod_z = _sample_mask(zs, D - 1, lod)

    all_positions = []
    all_colors = []
    all_indices = []
    vertex_offset = 0

    for face_index, (axis, is_min) in enumerate(_FACE_BOUND):
        if not visible_faces[face_index]:
            continue
        coord = (xs, ys, zs)[axis]
        limit = (W - 1, H - 1, D - 1)[axis]
        boundary_mask = coord == (0 if is_min else limit)

        if axis == 0:
            sample_mask = lod_y & lod_z
        elif axis == 1:
            sample_mask = lod_x & lod_z
        else:
            sample_mask = lod_x & lod_y

        mask = boundary_mask & sample_mask
        if not mask.any():
            continue

        cell_x = xs[mask].astype(np.float32) - W / 2.0 + 0.5
        cell_y = ys[mask].astype(np.float32) - H / 2.0 + 0.5
        cell_z = zs[mask].astype(np.float32) - D / 2.0 + 0.5
        centers = np.stack([cell_x, cell_y, cell_z], axis=1)
        face_values = faces[..., face_index][mask]

        n = centers.shape[0]
        quad = _FACE_VERTS[face_index]
        positions = centers[:, None, :] + quad[None, :, :]
        all_positions.append(positions.reshape(-1, 3))

        colors = _color_for_states(face_values, states).astype(np.uint8)
        all_colors.append(np.repeat(colors, 4, axis=0))

        base = (np.arange(n, dtype=np.uint32) * 4)[:, None] + vertex_offset
        tris = np.stack(
            [base[:, 0], base[:, 0] + 1, base[:, 0] + 2, base[:, 0], base[:, 0] + 2, base[:, 0] + 3],
            axis=1,
        )
        all_indices.append(tris.reshape(-1))
        vertex_offset += n * 4

    if not all_positions:
        return {
            "positions": np.zeros((0, 3), dtype=np.float32),
            "colors": np.zeros((0, 3), dtype=np.uint8),
            "indices": np.zeros((0,), dtype=np.uint32),
            "states": states,
        }

    return {
        "positions": np.concatenate(all_positions, axis=0),
        "colors": np.concatenate(all_colors, axis=0),
        "indices": np.concatenate(all_indices, axis=0),
        "states": states,
    }


def build_slice_mesh(faces: np.ndarray, *, slice_axis: int, slice_index: int, lod: int = 1):
    """Build slice-mode geometry: all six faces of every cell on one axis-aligned plane."""
    W, H, D, _ = faces.shape
    limits = (W - 1, H - 1, D - 1)
    sizes = (W, H, D)
    slice_index = max(0, min(sizes[slice_axis] - 1, int(slice_index)))

    other_axes = [axis for axis in range(3) if axis != slice_axis]
    size_a = sizes[other_axes[0]]
    size_b = sizes[other_axes[1]]
    coord_a, coord_b = np.meshgrid(np.arange(size_a), np.arange(size_b), indexing="ij")
    mask = _sample_mask(coord_a, size_a - 1, lod) & _sample_mask(coord_b, size_b - 1, lod)

    coord_a = coord_a[mask]
    coord_b = coord_b[mask]
    coords = [None, None, None]
    coords[slice_axis] = np.full_like(coord_a, slice_index)
    coords[other_axes[0]] = coord_a
    coords[other_axes[1]] = coord_b
    xs, ys, zs = coords

    states = int(faces.max()) + 1 if faces.size else 2
    cell_x = xs.astype(np.float32) - W / 2.0 + 0.5
    cell_y = ys.astype(np.float32) - H / 2.0 + 0.5
    cell_z = zs.astype(np.float32) - D / 2.0 + 0.5
    centers = np.stack([cell_x, cell_y, cell_z], axis=1)
    n = centers.shape[0]

    all_positions = []
    all_colors = []
    all_indices = []
    vertex_offset = 0
    for face_index in range(6):
        quad = _FACE_VERTS[face_index]
        positions = centers[:, None, :] + quad[None, :, :]
        all_positions.append(positions.reshape(-1, 3))

        face_values = faces[xs, ys, zs, face_index]
        colors = _color_for_states(face_values, states).astype(np.uint8)
        all_colors.append(np.repeat(colors, 4, axis=0))

        base = (np.arange(n, dtype=np.uint32) * 4)[:, None] + vertex_offset
        tris = np.stack(
            [base[:, 0], base[:, 0] + 1, base[:, 0] + 2, base[:, 0], base[:, 0] + 2, base[:, 0] + 3],
            axis=1,
        )
        all_indices.append(tris.reshape(-1))
        vertex_offset += n * 4

    if not all_positions or n == 0:
        return {
            "positions": np.zeros((0, 3), dtype=np.float32),
            "colors": np.zeros((0, 3), dtype=np.uint8),
            "indices": np.zeros((0,), dtype=np.uint32),
            "states": states,
        }

    return {
        "positions": np.concatenate(all_positions, axis=0),
        "colors": np.concatenate(all_colors, axis=0),
        "indices": np.concatenate(all_indices, axis=0),
        "states": states,
    }
