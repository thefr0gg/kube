"""CLI argument definitions and handler for `kube benchmark`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kube.benchmark._reporter import format_table, plot_results, to_json_dict
from kube.benchmark._runner import BenchmarkConfig, run_benchmarks

_ALL_WORKLOADS = [
    "random_read",
    "random_write",
    "neighborhood",
    "region_extract",
    "full_scan",
    "histogram",
    "polars_pipeline",
    "batch_iter",
    "lm_sequence_iter",
    "mutation",
]


def add_benchmark_parser(parser: argparse.ArgumentParser) -> None:
    """Attach benchmark-specific arguments to *parser*."""
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        metavar="N",
        help="grid side length (use 32/64/128/256)",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=4,
        help="face state count",
    )
    parser.add_argument(
        "--workloads",
        type=_parse_list,
        default=None,
        metavar="W,...",
        help=f"comma-separated workload names; available: {', '.join(_ALL_WORKLOADS)}",
    )
    parser.add_argument(
        "--backends",
        type=_parse_list,
        default=None,
        metavar="B,...",
        help="comma-separated backend names (kube, numpy, binary, polars)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="timing repetitions per workload",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="save throughput bar chart to benchmark_results.png",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        dest="dataset_path",
        metavar="PATH",
        help=".kubed dataset path for the lm_sequence_iter workload",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        dest="tmp_dir",
        metavar="PATH",
        help="directory for temporary fixture files (default: system tmp)",
    )


def handle_benchmark(args: argparse.Namespace) -> None:
    """Build config from parsed args, run benchmarks, and emit results."""
    config = BenchmarkConfig(
        size=args.size,
        states=args.states,
        workloads=args.workloads,
        backends=args.backends,
        iterations=args.iterations,
        seed=args.seed,
        dataset_path=args.dataset_path,
        tmp_dir=args.tmp_dir,
        plot=args.plot,
    )

    result = run_benchmarks(config)

    if getattr(args, "output", "text") == "json":
        print(json.dumps(to_json_dict(result), indent=2))
    else:
        print(format_table(result))

    if args.plot:
        out_path = "benchmark_results.png"
        plot_results(result, out_path)
        print(f"\nChart saved to {out_path}")


def _parse_list(text: str) -> list[str]:
    """Parse a comma-separated option value into a list of stripped strings."""
    return [item.strip() for item in text.split(",") if item.strip()]
