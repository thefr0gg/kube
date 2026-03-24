"""Command-line interface for Kube.

This module exists to expose the project capabilities as terminal commands.
The CLI covers three major workflow families:

- `.kube` storage creation, import/export, inspection, and stats
- `.kubed` dataset creation, ingest, sampling, and reporting
- the native viewer/editor for interactive inspection

The implementation is intentionally split into small handler functions so the
argparse definitions can stay descriptive while the command execution paths stay
easy to test and reason about.
"""

from __future__ import annotations

import argparse
import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from kube import (
    create,
    create_sharded,
    import_file,
    import_parquet_text,
    import_token_ids,
    open as open_kube,
    open_dataset,
)
from kube.logs import DEFAULT_LOG_LEVEL, LOG_LEVELS, configure_logging, get_logger, log_event
from kube.model import Kube


TOP_LEVEL_EPILOG = """Examples:
  kube create data/demo.kube --shape 16x16x16 --states 4
  kube import README.md data/readme.kube --states 4
  kube inspect data/readme.kube
  kube stats data/readme.kube --slice z=0 --normalize
  kube view data/readme.kube --lod 2 --window-size 1600x1000
  kube dataset-create corpora/wiki.kubed --shards 8 --sequence-length 1024
  kube dataset-sample corpora/wiki.kubed --sequence-length 256 --count 2
"""


class KubeHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show defaults and preserve multiline examples in `--help` output."""


def _package_version() -> str:
    """Resolve the installed package version for the CLI `--version` flag."""
    try:
        return version("kube")
    except PackageNotFoundError:
        return "0.0.0+local"


def _shape_label(shape: tuple[int, int, int]) -> str:
    """Render a `(W, H, D)` tuple using the CLI's compact shape syntax."""
    return "x".join(str(part) for part in shape)


def _slice_label(slice_spec: tuple[int, int] | None) -> str | None:
    """Render an internal `(axis, index)` slice tuple back into CLI syntax."""
    if slice_spec is None:
        return None
    axis_name = "xyz"[slice_spec[0]]
    return f"{axis_name}={slice_spec[1]}"


def _window_size_label(size: tuple[int, int]) -> str:
    """Render a window size tuple using the CLI's `WIDTHxHEIGHT` syntax."""
    return f"{size[0]}x{size[1]}"


def _emit_json(payload: Any) -> None:
    """Write one pretty JSON payload to stdout."""
    print(json.dumps(payload, indent=2))


def _emit_optional_json(args, payload: dict[str, Any], *, text_lines: list[str]) -> None:
    """Emit either JSON or concise text for commands with dual output modes."""
    if args.output == "json":
        _emit_json(payload)
        return
    print("\n".join(text_lines))


def _parser(
    subparsers,
    name: str,
    *,
    help_text: str,
    description: str,
    epilog: str,
    aliases: tuple[str, ...] = (),
):
    """Create one subcommand parser with Kube's shared help formatting."""
    return subparsers.add_parser(
        name,
        aliases=list(aliases),
        help=help_text,
        description=description,
        epilog=epilog,
        formatter_class=KubeHelpFormatter,
    )


def _parse_shape(text: str) -> tuple[int, int, int]:
    """Parse either `N` or `WxHxD` CLI shape syntax."""
    if "x" not in text.lower():
        size = int(text)
        return (size, size, size)
    parts = text.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be N or WxHxD")
    try:
        dims = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape values must be integers") from exc
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("shape values must be positive")
    return dims


def _parse_slice(text: str) -> tuple[int, int]:
    """Parse `x=N`, `y=N`, or `z=N` slice syntax from the CLI."""
    axis_name, _, raw_index = text.partition("=")
    axis_name = axis_name.strip().lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis_name not in axis_map or not raw_index:
        raise argparse.ArgumentTypeError("slice must be x=N, y=N, or z=N")
    try:
        return axis_map[axis_name], int(raw_index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("slice index must be an integer") from exc


def _parse_window_size(text: str) -> tuple[int, int]:
    """Parse `WIDTHxHEIGHT` viewer sizes from the CLI."""
    width_text, _, height_text = text.lower().partition("x")
    if not width_text or not height_text:
        raise argparse.ArgumentTypeError("window size must be WIDTHxHEIGHT")
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window size must be WIDTHxHEIGHT") from exc
    if width < 320 or height < 240:
        raise argparse.ArgumentTypeError("window size is too small")
    return (width, height)


def _add_common_cli_options(parser: argparse.ArgumentParser) -> None:
    """Attach global logging and output controls to the root parser."""
    parser.add_argument(
        "--log-level",
        choices=tuple(LOG_LEVELS),
        default=DEFAULT_LOG_LEVEL.lower(),
        help="stderr log verbosity for Kube status messages",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="shortcut for --log-level debug",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only emit errors on stderr",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="optional file that receives the same Kube-formatted log lines as stderr",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="preferred result format for commands that support both text and JSON",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )


def _configure_cli_logging(args) -> None:
    """Set up the shared Kube logger from parsed CLI flags."""
    level = "debug" if args.verbose else args.log_level
    configure_logging(level=level, quiet=args.quiet, log_file=args.log_file)
    scope = "cli" if not getattr(args, "command", None) else f"cli.{args.command.replace('-', '_')}"
    args.logger = get_logger(scope)


def _handle_create(args) -> None:
    """Create an empty mapped kube file from CLI arguments."""
    log_event(
        args.logger,
        logging.INFO,
        "creating mapped kube",
        path=args.path,
        shape=args.shape,
        states=args.states,
        force=args.force,
    )
    with create(args.path, args.shape, states=args.states, overwrite=args.force) as kube:
        payload = {"operation": "create", **kube.inspect()}
    log_event(
        args.logger,
        logging.INFO,
        "created mapped kube",
        path=args.path,
        dtype=payload["dtype"],
        capacity_bytes=payload["capacity_bytes"],
        storage=payload["storage"],
    )
    _emit_optional_json(
        args,
        payload,
        text_lines=[
            f"Created {args.path}",
            (
                f"Shape: {_shape_label(payload['grid_shape'])}  "
                f"States: {payload['states_per_face']}  "
                f"DType: {payload['dtype']}  "
                f"Capacity: {payload['capacity_bytes']} bytes"
            ),
        ],
    )


def _handle_import(args) -> None:
    """Import a normal file into a `.kube` container from the CLI."""
    log_event(
        args.logger,
        logging.INFO,
        "importing file into kube",
        src=args.src,
        dst=args.dst,
        shape=args.shape,
        states=args.states,
    )
    with import_file(args.src, args.dst, states=args.states, shape=args.shape) as kube:
        payload = {"operation": "import", "src": str(args.src), "dst": str(args.dst), "inspect": kube.inspect()}
    log_event(
        args.logger,
        logging.INFO,
        "completed file import",
        dst=args.dst,
        packed_bytes=payload["inspect"]["packed_bytes"],
        source_byte_count=payload["inspect"]["source_byte_count"],
    )
    _emit_optional_json(
        args,
        payload,
        text_lines=[
            f"Imported {args.src} -> {args.dst}",
            (
                f"Shape: {_shape_label(payload['inspect']['grid_shape'])}  "
                f"States: {payload['inspect']['states_per_face']}  "
                f"Packed: {payload['inspect']['packed_bytes']} bytes  "
                f"Source: {payload['inspect']['source_byte_count']} bytes"
            ),
        ],
    )


def _handle_export(args) -> None:
    """Export the logical byte stream stored in a kube back to disk."""
    log_event(args.logger, logging.INFO, "exporting kube payload", src=args.src, dst=args.dst)
    with open_kube(args.src, mode="r") as kube:
        kube.export_file(args.dst)
        inspect_payload = kube.inspect()
    payload = {
        "operation": "export",
        "src": str(args.src),
        "dst": str(args.dst),
        "source_byte_count": inspect_payload["source_byte_count"],
        "grid_shape": inspect_payload["grid_shape"],
    }
    log_event(
        args.logger,
        logging.INFO,
        "exported kube payload",
        dst=args.dst,
        source_byte_count=inspect_payload["source_byte_count"],
    )
    _emit_optional_json(
        args,
        payload,
        text_lines=[
            f"Exported {args.src} -> {args.dst}",
            f"Logical source bytes: {inspect_payload['source_byte_count']}",
        ],
    )


def _handle_inspect(args) -> None:
    """Print kube metadata and storage stats as JSON."""
    log_event(args.logger, logging.INFO, "inspecting kube", path=args.path)
    with open_kube(args.path, mode="r") as kube:
        payload = kube.inspect()
    log_event(
        args.logger,
        logging.INFO,
        "inspection complete",
        path=args.path,
        shape=payload["grid_shape"],
        dtype=payload["dtype"],
        storage=payload["storage"],
    )
    _emit_json(payload)


def _handle_stats(args) -> None:
    """Compute face-value histograms, optionally restricted to one slice."""
    log_event(
        args.logger,
        logging.INFO,
        "computing face histogram",
        path=args.path,
        slice=_slice_label(args.slice_spec),
        normalize=args.normalize,
    )
    with open_kube(args.path, mode="r") as kube:
        bounds = None
        if args.slice_spec is not None:
            slice_axis, slice_index = args.slice_spec
            bounds = kube.slice_bounds(slice_axis, slice_index)
        histogram = kube.face_histogram(bounds=bounds, normalize=args.normalize)
        payload = {
            "shape": kube.shape,
            "states": kube.S,
            "bounds": bounds,
            "normalize": args.normalize,
            "rows": histogram.to_dicts(),
        }
    log_event(
        args.logger,
        logging.INFO,
        "face histogram ready",
        rows=len(payload["rows"]),
        slice=_slice_label(args.slice_spec),
    )
    _emit_json(payload)


def _handle_view(args) -> None:
    """Open the native viewer/editor for a kube file."""
    log_event(
        args.logger,
        logging.INFO,
        "launching viewer",
        path=args.path,
        slice=_slice_label(args.slice_spec),
        lod=args.lod,
        read_only=args.read_only,
        window_size=_window_size_label(args.window_size),
    )
    with open_kube(args.path, mode="r" if args.read_only else "r+") as kube:
        slice_axis = None
        slice_index = None
        if args.slice_spec is not None:
            slice_axis, slice_index = args.slice_spec
        kube.view(
            read_only=args.read_only,
            slice_axis=slice_axis,
            slice_index=slice_index,
            lod=args.lod,
            window_size=args.window_size,
        )
    log_event(args.logger, logging.INFO, "viewer session finished", path=args.path)


def _handle_dataset_create(args) -> None:
    """Create an empty sharded `.kubed` dataset root."""
    log_event(
        args.logger,
        logging.INFO,
        "creating sharded dataset",
        path=args.path,
        shards=args.shards,
        states=args.states,
        token_bytes=args.token_bytes,
        tokens_per_shard=args.tokens_per_shard,
        sequence_length=args.sequence_length,
    )
    with create_sharded(
        args.path,
        shards=args.shards,
        states=args.states,
        token_bytes=args.token_bytes,
        tokens_per_shard=args.tokens_per_shard,
        sequence_length=args.sequence_length,
        dataset_name=args.name,
    ) as dataset:
        payload = dataset.inspect()
    log_event(
        args.logger,
        logging.INFO,
        "created sharded dataset",
        root=payload["root"],
        shard_count=payload["shard_count"],
        total_token_count=payload["total_token_count"],
    )
    _emit_json(payload)


def _handle_dataset_import_text(args) -> None:
    """Tokenize raw text rows and import them into a `.kubed` dataset."""
    log_event(
        args.logger,
        logging.INFO,
        "importing text rows into dataset",
        src=args.src,
        dst=args.dst,
        text_column=args.text_column,
        tokenizer=args.tokenizer,
        sequence_length=args.sequence_length,
        eos_token_id=args.eos_token_id,
    )
    with import_parquet_text(
        args.src,
        args.dst,
        text_column=args.text_column,
        tokenizer_path=args.tokenizer,
        eos_token_id=args.eos_token_id,
    ) as dataset:
        dataset.update_sequence_defaults(sequence_length=args.sequence_length)
        payload = dataset.inspect()
    log_event(
        args.logger,
        logging.INFO,
        "dataset text import complete",
        root=payload["root"],
        total_document_count=payload["total_document_count"],
        total_token_count=payload["total_token_count"],
    )
    _emit_json(payload)


def _handle_dataset_import_tokens(args) -> None:
    """Import pre-tokenized rows into a `.kubed` dataset."""
    log_event(
        args.logger,
        logging.INFO,
        "importing token rows into dataset",
        src=args.src,
        dst=args.dst,
        tokens_column=args.tokens_column,
        sequence_length=args.sequence_length,
        eos_token_id=args.eos_token_id,
    )
    with import_token_ids(
        args.src,
        args.dst,
        tokens_column=args.tokens_column,
        eos_token_id=args.eos_token_id,
    ) as dataset:
        if args.sequence_length is not None:
            dataset.update_sequence_defaults(sequence_length=args.sequence_length)
        payload = dataset.inspect()
    log_event(
        args.logger,
        logging.INFO,
        "dataset token import complete",
        root=payload["root"],
        total_document_count=payload["total_document_count"],
        total_token_count=payload["total_token_count"],
    )
    _emit_json(payload)


def _handle_dataset_inspect(args) -> None:
    """Print high-level dataset metadata plus shard rows as JSON."""
    log_event(args.logger, logging.INFO, "inspecting dataset", path=args.path)
    with open_dataset(args.path, mode="r") as dataset:
        payload = dataset.inspect()
        payload["shards"] = dataset.shard_frame().to_dicts()
    log_event(
        args.logger,
        logging.INFO,
        "dataset inspection complete",
        shard_count=payload["shard_count"],
        total_token_count=payload["total_token_count"],
    )
    _emit_json(payload)


def _handle_dataset_stats(args) -> None:
    """Report shard/document stats and optionally export them to Parquet."""
    log_event(
        args.logger,
        logging.INFO,
        "collecting dataset stats",
        path=args.path,
        export_parquet=args.export_parquet,
        export_documents=args.export_documents,
    )
    with open_dataset(args.path, mode="r") as dataset:
        shards = dataset.shard_frame()
        documents = dataset.document_frame()
        payload = {
            "dataset": dataset.inspect(),
            "shards": shards.to_dicts(),
            "documents": {
                "count": documents.height,
                "path": str((Path(args.path) / "stats" / "documents.parquet")) if documents.height else None,
            },
        }
        if args.export_parquet is not None:
            dataset.export_stats_parquet(args.export_parquet)
            payload["export_parquet"] = str(args.export_parquet)
        if args.export_documents is not None:
            dataset.export_document_parquet(args.export_documents)
            payload["export_documents"] = str(args.export_documents)
    log_event(
        args.logger,
        logging.INFO,
        "dataset stats ready",
        shard_rows=len(payload["shards"]),
        document_count=payload["documents"]["count"],
    )
    _emit_json(payload)


def _handle_dataset_sample(args) -> None:
    """Sample causal LM windows from a dataset for debugging or inspection."""
    log_event(
        args.logger,
        logging.INFO,
        "sampling dataset sequences",
        path=args.path,
        sequence_length=args.sequence_length,
        count=args.count,
        stride=args.stride,
        resume_state_in=args.resume_state_in,
        resume_state_out=args.resume_state_out,
    )
    with open_dataset(args.path, mode="r") as dataset:
        start_state = None
        if args.resume_state_in is not None:
            start_state = dataset.load_resume_state(args.resume_state_in)
        rows = []
        last_resume_state = start_state
        for sequence_index, sample in enumerate(
            dataset.iter_sequences(
                args.sequence_length,
                stride=args.stride,
                start_state=start_state,
            )
        ):
            if sequence_index >= args.count:
                break
            rows.append(
                {
                    "sequence_index": sequence_index,
                    "shard_index": sample["shard_index"],
                    "token_offset": sample["token_offset"],
                    "input_ids": sample["input_ids"],
                    "target_ids": sample["target_ids"],
                }
            )
            last_resume_state = sample["resume_state"]
        if last_resume_state is not None and args.resume_state_out is not None:
            dataset.save_resume_state(args.resume_state_out, last_resume_state)
        payload = {
            "dataset": dataset.inspect(),
            "sequence_length": args.sequence_length,
            "count": len(rows),
            "rows": rows,
        }
        if args.export_parquet is not None:
            dataset.export_sequence_sample_parquet(
                args.export_parquet,
                sequence_length=args.sequence_length,
                count=args.count,
                stride=args.stride,
            )
            payload["export_parquet"] = str(args.export_parquet)
        if args.resume_state_out is not None and last_resume_state is not None:
            payload["resume_state_out"] = str(args.resume_state_out)
        if last_resume_state is not None:
            payload["resume_state"] = last_resume_state
    log_event(
        args.logger,
        logging.INFO,
        "dataset sampling complete",
        returned_rows=payload["count"],
        resume_state_saved=args.resume_state_out is not None and "resume_state_out" in payload,
    )
    _emit_json(payload)


def _handle_demo(args) -> None:
    """Launch an in-memory demo kube filled with repeated sample bytes."""
    log_event(
        args.logger,
        logging.INFO,
        "launching in-memory demo",
        shape=args.shape,
        states=args.states,
        lod=args.lod,
        window_size=_window_size_label(args.window_size),
    )
    kube = Kube(*args.shape, states=args.states)
    try:
        sample = b"Hello Kube! "
        payload_bytes = (sample * ((kube.capacity_bytes() // len(sample)) + 1))[: kube.capacity_bytes()]
        kube.encode(payload_bytes)
        payload = {"operation": "demo", **kube.inspect()}
        _emit_optional_json(
            args,
            payload,
            text_lines=[
                str(kube),
                f"Demo shape: {_shape_label(kube.shape)}  States: {kube.S}  Capacity: {kube.capacity_bytes()} bytes",
            ],
        )
        kube.view(lod=args.lod, window_size=args.window_size)
    finally:
        kube.close()
    log_event(args.logger, logging.INFO, "demo viewer session finished", shape=args.shape)


def build_parser() -> argparse.ArgumentParser:
    """Build the full top-level argparse parser for the Kube CLI."""
    parser = argparse.ArgumentParser(
        prog="kube",
        description=(
            "Kube CLI for packed cube storage, dataset preparation, and the native viewer.\n\n"
            "Kube keeps byte-oriented data in `.kube` containers and token-oriented corpora in\n"
            "sharded `.kubed` dataset roots."
        ),
        epilog=TOP_LEVEL_EPILOG,
        formatter_class=KubeHelpFormatter,
    )
    _add_common_cli_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    create_parser = _parser(
        subparsers,
        "create",
        help_text="create an empty mapped .kube file",
        description="Create a new on-disk `.kube` container with the requested shape and state count.",
        epilog="Example:\n  kube create data/demo.kube --shape 32x16x8 --states 4",
    )
    create_parser.add_argument("path", type=Path, help="destination `.kube` file to create")
    create_parser.add_argument("--shape", required=True, type=_parse_shape, metavar="N|WxHxD", help="cube dimensions")
    create_parser.add_argument("--states", default=4, type=int, help="number of allowed values per face")
    create_parser.add_argument("--force", action="store_true", help="overwrite an existing destination file")
    create_parser.set_defaults(func=_handle_create)

    import_parser = _parser(
        subparsers,
        "import",
        help_text="stream a file into a .kube file",
        description="Read a normal file and encode its bytes into a new `.kube` container.",
        epilog="Example:\n  kube import README.md data/readme.kube --states 4 --shape 24x24x24",
    )
    import_parser.add_argument("src", type=Path, help="source file to encode")
    import_parser.add_argument("dst", type=Path, help="destination `.kube` file")
    import_parser.add_argument("--states", default=4, type=int, help="number of allowed values per face")
    import_parser.add_argument("--shape", type=_parse_shape, metavar="N|WxHxD", help="optional target shape override")
    import_parser.set_defaults(func=_handle_import)

    export_parser = _parser(
        subparsers,
        "export",
        help_text="stream bytes back out of a .kube file",
        description="Decode the logical byte stream stored in a `.kube` file back to a normal file.",
        epilog="Example:\n  kube export data/readme.kube README.copy.md",
    )
    export_parser.add_argument("src", type=Path, help="source `.kube` file")
    export_parser.add_argument("dst", type=Path, help="destination file path")
    export_parser.set_defaults(func=_handle_export)

    inspect_parser = _parser(
        subparsers,
        "inspect",
        help_text="inspect kube metadata and storage stats",
        description="Print packed-storage, shape, capacity, and metadata details for one `.kube` file.",
        epilog="Example:\n  kube inspect data/readme.kube",
    )
    inspect_parser.add_argument("path", type=Path, help="target `.kube` file")
    inspect_parser.set_defaults(func=_handle_inspect)

    dataset_create_parser = _parser(
        subparsers,
        "dataset-create",
        help_text="create a sharded .kubed dataset root",
        description="Create an empty sharded dataset root that can later receive token or text imports.",
        epilog="Example:\n  kube dataset-create corpora/wiki.kubed --shards 16 --sequence-length 1024",
    )
    dataset_create_parser.add_argument("path", type=Path, help="destination `.kubed` dataset root")
    dataset_create_parser.add_argument("--shards", required=True, type=int, help="number of shard files to create")
    dataset_create_parser.add_argument("--states", default=256, type=int, help="token value space per face")
    dataset_create_parser.add_argument("--token-bytes", default=4, type=int, help="storage width for one token")
    dataset_create_parser.add_argument("--tokens-per-shard", default=1 << 20, type=int, help="target shard capacity in tokens")
    dataset_create_parser.add_argument("--sequence-length", default=1024, type=int, help="default sequence length recorded in the manifest")
    dataset_create_parser.add_argument("--name", help="optional human-readable dataset name")
    dataset_create_parser.set_defaults(func=_handle_dataset_create)

    dataset_import_text_parser = _parser(
        subparsers,
        "dataset-import-text",
        help_text="tokenize text from Parquet or Arrow into a .kubed dataset",
        description="Read text rows from Parquet or Arrow/IPC, tokenize them, and import them into a sharded dataset.",
        epilog="Example:\n  kube dataset-import-text rows.parquet corpora/wiki.kubed --text-column text --tokenizer tokenizer.json",
    )
    dataset_import_text_parser.add_argument("src", type=Path, help="Parquet or Arrow/IPC input file")
    dataset_import_text_parser.add_argument("dst", type=Path, help="destination `.kubed` dataset root")
    dataset_import_text_parser.add_argument("--text-column", default="text", help="column containing raw text")
    dataset_import_text_parser.add_argument("--tokenizer", required=True, type=Path, help="path to a tokenizer JSON file")
    dataset_import_text_parser.add_argument("--sequence-length", default=1024, type=int, help="default sequence length recorded after import")
    dataset_import_text_parser.add_argument("--eos-token-id", type=int, help="optional EOS token ID override")
    dataset_import_text_parser.set_defaults(func=_handle_dataset_import_text)

    dataset_import_tokens_parser = _parser(
        subparsers,
        "dataset-import-tokens",
        help_text="import pretokenized rows from Parquet or Arrow into a .kubed dataset",
        description="Read rows that already contain token ID arrays and append them into a sharded dataset.",
        epilog="Example:\n  kube dataset-import-tokens rows.parquet corpora/wiki.kubed --tokens-column input_ids",
    )
    dataset_import_tokens_parser.add_argument("src", type=Path, help="Parquet or Arrow/IPC input file")
    dataset_import_tokens_parser.add_argument("dst", type=Path, help="destination `.kubed` dataset root")
    dataset_import_tokens_parser.add_argument("--tokens-column", default="input_ids", help="column containing token ID arrays")
    dataset_import_tokens_parser.add_argument("--sequence-length", type=int, help="optional manifest sequence length override")
    dataset_import_tokens_parser.add_argument("--eos-token-id", type=int, help="optional EOS token ID appended between rows")
    dataset_import_tokens_parser.set_defaults(func=_handle_dataset_import_tokens)

    dataset_inspect_parser = _parser(
        subparsers,
        "dataset-inspect",
        help_text="inspect a sharded .kubed dataset",
        description="Print dataset manifest details together with per-shard summary rows.",
        epilog="Example:\n  kube dataset-inspect corpora/wiki.kubed",
    )
    dataset_inspect_parser.add_argument("path", type=Path, help="target `.kubed` dataset root")
    dataset_inspect_parser.set_defaults(func=_handle_dataset_inspect)

    dataset_stats_parser = _parser(
        subparsers,
        "dataset-stats",
        help_text="show shard/document stats for a .kubed dataset",
        description="Compute dataset-wide shard and document summaries and optionally export them to Parquet.",
        epilog="Example:\n  kube dataset-stats corpora/wiki.kubed --export-parquet stats/shards.parquet",
    )
    dataset_stats_parser.add_argument("path", type=Path, help="target `.kubed` dataset root")
    dataset_stats_parser.add_argument("--export-parquet", type=Path, help="optional Parquet output for shard stats")
    dataset_stats_parser.add_argument("--export-documents", type=Path, help="optional Parquet output for document stats")
    dataset_stats_parser.set_defaults(func=_handle_dataset_stats)

    dataset_sample_parser = _parser(
        subparsers,
        "dataset-sample",
        help_text="sample causal LM sequences from a .kubed dataset",
        description="Iterate causal language-model windows from a dataset and print a small sample payload.",
        epilog="Example:\n  kube dataset-sample corpora/wiki.kubed --sequence-length 256 --count 4 --stride 128",
    )
    dataset_sample_parser.add_argument("path", type=Path, help="target `.kubed` dataset root")
    dataset_sample_parser.add_argument("--sequence-length", required=True, type=int, help="tokens per sampled window")
    dataset_sample_parser.add_argument("--count", default=4, type=int, help="number of sample rows to return")
    dataset_sample_parser.add_argument("--stride", type=int, help="token offset between windows")
    dataset_sample_parser.add_argument("--export-parquet", type=Path, help="optional Parquet output for the sampled rows")
    dataset_sample_parser.add_argument("--resume-state-in", type=Path, help="resume-state JSON file to continue from")
    dataset_sample_parser.add_argument("--resume-state-out", type=Path, help="path to save the final resume state")
    dataset_sample_parser.set_defaults(func=_handle_dataset_sample)

    stats_parser = _parser(
        subparsers,
        "stats",
        help_text="compute per-face value histograms",
        description="Compute face-value histograms across the whole kube or one requested slice.",
        epilog="Example:\n  kube stats data/demo.kube --slice x=3 --normalize",
    )
    stats_parser.add_argument("path", type=Path, help="target `.kube` file")
    stats_parser.add_argument("--slice", dest="slice_spec", type=_parse_slice, metavar="x=N|y=N|z=N", help="optional slice restriction")
    stats_parser.add_argument("--normalize", action="store_true", help="report proportions instead of raw counts")
    stats_parser.set_defaults(func=_handle_stats)

    view_parser = _parser(
        subparsers,
        "view",
        help_text="view a kube file",
        description="Open the native Kube viewer/editor for one `.kube` file.",
        epilog="Example:\n  kube view data/demo.kube --slice z=4 --lod 2 --window-size 1600x1000",
    )
    view_parser.add_argument("path", type=Path, help="target `.kube` file")
    view_parser.add_argument("--slice", dest="slice_spec", type=_parse_slice, metavar="x=N|y=N|z=N", help="start in slice mode at the given slice")
    view_parser.add_argument("--lod", default=1, type=int, help="viewer sampling stride")
    view_parser.add_argument("--window-size", default=(1440, 900), type=_parse_window_size, metavar="WIDTHxHEIGHT", help="initial viewer window size")
    view_parser.add_argument("--read-only", action="store_true", help="disable edit/autosave operations")
    view_parser.set_defaults(func=_handle_view)

    demo_parser = _parser(
        subparsers,
        "demo",
        help_text="open an in-memory demo kube",
        description="Build a small in-memory demo kube, fill it with repeated sample bytes, and open the viewer.",
        epilog="Example:\n  kube demo --shape 12x12x12 --states 4 --lod 2",
    )
    demo_parser.add_argument("--shape", default=(8, 8, 8), type=_parse_shape, metavar="N|WxHxD", help="demo cube dimensions")
    demo_parser.add_argument("--states", default=4, type=int, help="number of allowed values per face")
    demo_parser.add_argument("--lod", default=1, type=int, help="viewer sampling stride")
    demo_parser.add_argument("--window-size", default=(1440, 900), type=_parse_window_size, metavar="WIDTHxHEIGHT", help="initial viewer window size")
    demo_parser.set_defaults(func=_handle_demo)

    return parser


def main() -> None:
    """Build the CLI parser, configure logging, and dispatch one subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    _configure_cli_logging(args)
    try:
        args.func(args)
    except BrokenPipeError:
        raise SystemExit(0)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if args.verbose and not args.quiet:
            args.logger.exception("command failed")
        else:
            log_event(args.logger, logging.ERROR, "command failed", error=message)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
