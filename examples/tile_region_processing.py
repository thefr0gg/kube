"""Process a bounded Kube region tile by tile.

This example exists to show how `iter_region(...)` can drive deterministic
spatial preprocessing without materializing an entire large cube as one
DataFrame.

Workflow:

1. Open a mapped `.kube` file in read-only mode.
2. Iterate a bounded region in fixed-size sub-cubes.
3. Compute a tiny per-tile summary that can be logged or exported.
4. Optionally export the full region as a single Arrow IPC file via
   `kube.to_arrow(bounds=...)` for downstream tools.

Typical usage:

    python examples/tile_region_processing.py data/world.kube

    python examples/tile_region_processing.py data/world.kube \
        --bounds 0:128,0:128,0:64 \
        --tile-shape 32x32x16 \
        --summary-out reports/world-tiles.parquet \
        --arrow-out reports/world-region.arrow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from kube import open as open_kube
from kube.schemas import SCHEMA_WIDE_FACES


def parse_bounds(text: str):
    """Parse `x0:x1,y0:y1,z0:z1` bounds into the tuple form Kube expects."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("bounds must look like x0:x1,y0:y1,z0:z1")

    parsed = []
    for part in parts:
        values = [value.strip() for value in part.split(":", 1)]
        if len(values) != 2:
            raise argparse.ArgumentTypeError("each bounds segment must look like start:stop")
        parsed.append((int(values[0]), int(values[1])))
    return tuple(parsed)


def parse_shape(text: str) -> tuple[int, int, int]:
    """Parse `AxBxC` style shapes used for tile sizes."""
    separators = ("x", "X", ",")
    normalized = text
    for separator in separators[1:]:
        normalized = normalized.replace(separator, separators[0])
    parts = [part.strip() for part in normalized.split("x")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("tile shape must look like 32x32x16")
    return tuple(int(part) for part in parts)


def parse_args() -> argparse.Namespace:
    """Define the CLI for the tiled region example."""
    parser = argparse.ArgumentParser(
        description="Iterate a region of a kube in deterministic tile-sized chunks"
    )
    parser.add_argument("kube_path", type=Path, help="Mapped `.kube` file to inspect")
    parser.add_argument(
        "--bounds",
        type=parse_bounds,
        default=((0, 128), (0, 128), (0, 64)),
        help="Region bounds as x0:x1,y0:y1,z0:z1",
    )
    parser.add_argument(
        "--tile-shape",
        type=parse_shape,
        default=(32, 32, 16),
        help="Tile shape as AxBxC",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional Parquet path for the per-tile summary table",
    )
    parser.add_argument(
        "--arrow-out",
        type=Path,
        help="Optional Arrow IPC path for the full region exported in one call via to_arrow(bounds=...)",
    )
    return parser.parse_args()


def main() -> None:
    """Open the kube, walk tiles, and emit a small summary report."""
    args = parse_args()
    rows: list[dict[str, int]] = []

    with open_kube(args.kube_path, mode="r") as kube:
        print(f"kube  : {kube}")
        print(f"schema: {list(kube.schema(SCHEMA_WIDE_FACES).keys())}")
        print()

        for origin, tile in kube.iter_region(args.bounds, tile_shape=args.tile_shape):
            rows.append(
                {
                    "origin_x": int(origin[0]),
                    "origin_y": int(origin[1]),
                    "origin_z": int(origin[2]),
                    "rows": int(tile.height),
                    "packed_nonzero": int(tile.filter(pl.col("packed") != 0).height),
                    "face_0_sum": int(tile["face_0"].sum()),
                    "face_4_sum": int(tile["face_4"].sum()),
                }
            )

        if args.arrow_out is not None:
            table = kube.to_arrow(bounds=args.bounds)
            try:
                import pyarrow.ipc as ipc
                with ipc.new_file(str(args.arrow_out), table.schema) as writer:
                    writer.write_table(table)
                print(f"arrow region export : {args.arrow_out}  ({table.num_rows:,} rows)")
            except ImportError:
                print("skipping arrow export: pyarrow not installed")

    summary = pl.DataFrame(rows)
    print(summary)

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary.write_parquet(args.summary_out)
        print(f"wrote summary: {args.summary_out}")


if __name__ == "__main__":
    main()
