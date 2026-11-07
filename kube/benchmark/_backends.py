"""Fixture setup for the Kube benchmark suite.

Each scenario builds its own fixtures (source files, populated kubes,
sharded datasets) once, up front, so workload timings only measure the
operation under test, not setup cost.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from kube.model import Kube


def _xorshift32(state: int) -> int:
    """Return the next deterministic pseudo-random 32-bit integer."""
    state &= 0xFFFF_FFFF
    state ^= (state << 13) & 0xFFFF_FFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFF_FFFF
    return state & 0xFFFF_FFFF


def _random_bytes(n: int, seed: int) -> bytes:
    """Build n deterministic pseudo-random bytes without needing numpy."""
    state = int(seed) or 1
    out = bytearray(n)
    for i in range(0, n, 4):
        state = _xorshift32(state)
        chunk = state.to_bytes(4, "little")
        out[i : i + 4] = chunk[: min(4, n - i)]
    return bytes(out)


def build_fixtures(
    size: int,
    states: int,
    seed: int,
    *,
    tmp_dir: str | None = None,
) -> dict[str, Any]:
    """Build fixtures for every scenario, keyed by scenario name.

    `size` scales each scenario's workload (file bytes, grid side length,
    token count) so `--size` behaves consistently across scenarios.
    """
    tmp_root = Path(tempfile.mkdtemp(dir=tmp_dir, prefix="kube_bench_"))

    fixtures: dict[str, Any] = {
        "tmp_root": tmp_root,
        "size": size,
        "states": states,
        "seed": seed,
    }
    fixtures["stream_import"] = _build_stream_import_fixtures(tmp_root, size, states, seed)
    fixtures["table_conversion"] = _build_table_conversion_fixtures(tmp_root, size, states, seed)
    fixtures["lm_dataset"] = _build_lm_dataset_fixtures(tmp_root, size, seed)
    fixtures["viewer_ops"] = _build_viewer_ops_fixtures(size, states, seed)
    return fixtures


def _build_stream_import_fixtures(tmp_root: Path, size: int, states: int, seed: int) -> dict[str, Any]:
    """Source file for the streaming-vs-eager import scenario.

    `size` is the cube side length elsewhere; here it scales the source file
    to roughly `size^3` bytes so bigger `--size` runs exercise a bigger file,
    without requiring a literal multi-GB fixture for a quick benchmark run.
    """
    byte_count = max(1 << 16, size * size * size)
    src_path = tmp_root / "stream_import_source.bin"
    src_path.write_bytes(_random_bytes(byte_count, seed))
    return {
        "src_path": src_path,
        "byte_count": byte_count,
        "states": states,
        "stream_dst": tmp_root / "stream_import_streamed.kube",
        "eager_dst": tmp_root / "stream_import_eager.kube",
    }


def _build_table_conversion_fixtures(tmp_root: Path, size: int, states: int, seed: int) -> dict[str, Any]:
    """A populated mapped `.kube` file for the table-conversion scenario."""
    kube_path = tmp_root / "table_conversion.kube"
    kube = Kube.create(kube_path, (size, size, size), states=states)
    payload = _random_bytes(kube.capacity_bytes(), seed)
    kube.encode(payload)
    kube.flush()
    kube.close()
    return {
        "kube_path": kube_path,
        "shape": (size, size, size),
        "states": states,
        "parquet_path": tmp_root / "table_conversion.parquet",
    }


def _build_lm_dataset_fixtures(tmp_root: Path, size: int, seed: int) -> dict[str, Any]:
    """A Parquet token table plus a destination `.kubed` root for the LLM scenario.

    `size` scales total token count (`size^2` rows of `size` tokens each) so
    this scenario grows with `--size` without needing numpy/pyarrow installed
    just to build the fixture — the Parquet write goes through Polars, which
    is a hard Kube dependency.
    """
    import polars as pl

    sequence_length = max(8, min(size, 512))
    rows = max(8, size)
    state = int(seed) or 1
    sequences = []
    for _ in range(rows):
        seq = []
        for _ in range(sequence_length):
            state = _xorshift32(state)
            seq.append(state % 50_257)
        sequences.append(seq)

    tokens_path = tmp_root / "lm_dataset_tokens.parquet"
    pl.DataFrame({"input_ids": sequences}).write_parquet(tokens_path)

    total_tokens = rows * sequence_length
    return {
        "tokens_path": tokens_path,
        "sequence_length": sequence_length,
        "total_tokens": total_tokens,
        "dataset_root": tmp_root / "lm_dataset.kubed",
    }


def _build_viewer_ops_fixtures(size: int, states: int, seed: int) -> dict[str, Any]:
    """An in-memory kube plus pre-built edit ops for the viewer scenario."""
    viewer_size = max(4, min(size, 96))
    kube = Kube(viewer_size, viewer_size, viewer_size, states=states)
    payload = _random_bytes(kube.capacity_bytes(), seed)
    kube.encode(payload)

    state = int(seed) or 1
    edit_ops: list[tuple[int, int, int, int, int]] = []
    for _ in range(256):
        state = _xorshift32(state)
        x = state % viewer_size
        state = _xorshift32(state)
        y = state % viewer_size
        state = _xorshift32(state)
        z = state % viewer_size
        state = _xorshift32(state)
        face = state % 6
        state = _xorshift32(state)
        value = state % states
        edit_ops.append((x, y, z, face, value))

    return {
        "kube": kube,
        "shape": (viewer_size, viewer_size, viewer_size),
        "states": states,
        "edit_ops": edit_ops,
    }
