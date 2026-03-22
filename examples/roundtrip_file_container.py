"""Round-trip a normal file through a `.kube` container.

This example exists to show Kube as a reversible disk-backed container for
ordinary bytes rather than as a viewer-first tool.

Workflow:

1. Stream a source file into a mapped `.kube` artifact.
2. Inspect the resulting container metadata.
3. Decode the payload back into bytes and verify it matches the source.
4. Optionally export a restored copy to disk.

Typical usage:

    python examples/roundtrip_file_container.py input.bin data/input.kube

    python examples/roundtrip_file_container.py input.bin data/input.kube \
        --restored-out restored.bin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kube import import_file


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
        print(f"source bytes : {len(source_bytes)}")
        print(f"decoded bytes: {len(decoded)}")
        print(f"round-trip ok: {matches}")

        if not matches:
            raise SystemExit("decoded payload did not match the source file")

        if args.restored_out is not None:
            kube.export_file(args.restored_out)
            print(f"restored copy: {args.restored_out}")


if __name__ == "__main__":
    main()
