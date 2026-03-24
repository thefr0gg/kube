"""Round-trip a normal file through a `.kube` container.

This example exists to show Kube as a reversible disk-backed container for
ordinary bytes rather than as a viewer-first tool.

Workflow:

1. Stream a source file into a mapped `.kube` artifact.
2. Inspect the resulting container metadata.
3. Decode the payload back into bytes and verify it matches the source.
4. Optionally export a restored copy, an Arrow IPC file, or a Parquet file.

Typical usage:

    python examples/roundtrip_file_container.py input.bin data/input.kube

    python examples/roundtrip_file_container.py input.bin data/input.kube \
        --restored-out restored.bin \
        --arrow-out data/input.arrow \
        --parquet-out data/input.parquet \
        --schema-mode wide_faces
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kube import import_file
from kube.schemas import SCHEMA_CELL_PACKED, SCHEMA_LONG_FACES, SCHEMA_WIDE_FACES


def parse_args() -> argparse.Namespace:
    """Define the command line for the binary round-trip example."""
    parser = argparse.ArgumentParser(
        description="Import a normal file into Kube and verify byte-for-byte round-trip"
    )
    parser.add_argument("src", type=Path, help="Source file to wrap in a `.kube` container")
    parser.add_argument("dst_kube", type=Path, help="Destination `.kube` path to create")
    parser.add_argument("--states", type=int, default=4, help="Face-state count for the destination kube")
    parser.add_argument(
        "--restored-out",
        type=Path,
        help="Optional path to write the decoded payload back out as a normal file",
    )
    parser.add_argument(
        "--arrow-out",
        type=Path,
        help="Optional path to export the grid as an Arrow IPC file (requires pyarrow)",
    )
    parser.add_argument(
        "--parquet-out",
        type=Path,
        help="Optional path to export the grid as a Parquet file (requires pyarrow)",
    )
    parser.add_argument(
        "--schema-mode",
        choices=[SCHEMA_WIDE_FACES, SCHEMA_LONG_FACES, SCHEMA_CELL_PACKED],
        default=SCHEMA_WIDE_FACES,
        help="Column layout for Arrow/Parquet export",
    )
    return parser.parse_args()


def main() -> None:
    """Run the example import, verification, and optional export flow."""
    args = parse_args()
    source_bytes = args.src.read_bytes()

    with import_file(args.src, args.dst_kube, states=args.states) as kube:
        summary = kube.inspect()
        decoded = kube.decode()
        matches = decoded == source_bytes

        print("kube summary")
        print("============")
        print(json.dumps(summary, indent=2))
        print()
        print(f"kube            : {kube}")
        print(f"source bytes    : {len(source_bytes)}")
        print(f"decoded bytes   : {len(decoded)}")
        print(f"round-trip ok   : {matches}")
        print()
        print("schema (wide_faces)")
        print("-------------------")
        for col, dtype in kube.schema(SCHEMA_WIDE_FACES).items():
            print(f"  {col:<12} {dtype}")

        if not matches:
            raise SystemExit("decoded payload did not match the source file")

        if args.restored_out is not None:
            kube.export_file(args.restored_out)
            print(f"\nrestored copy   : {args.restored_out}")

        if args.arrow_out is not None:
            path = kube.to_arrow(schema_mode=args.schema_mode)
            import pyarrow.ipc as ipc
            with ipc.new_file(str(args.arrow_out), path.schema) as writer:
                writer.write_table(path)
            print(f"arrow export    : {args.arrow_out}  ({args.schema_mode})")

        if args.parquet_out is not None:
            kube.to_parquet(args.parquet_out, schema_mode=args.schema_mode)
            print(f"parquet export  : {args.parquet_out}  ({args.schema_mode})")


if __name__ == "__main__":
    main()
