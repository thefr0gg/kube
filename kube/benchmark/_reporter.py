"""Table formatter, JSON serializer, and optional matplotlib plot."""

from __future__ import annotations

import json
from typing import Any

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:
    _pkg_version = None  # type: ignore[assignment]
    PackageNotFoundError = Exception  # type: ignore[assignment,misc]

from kube.benchmark._runner import BenchmarkResult, WorkloadResult


def _kube_version() -> str:
    try:
        return _pkg_version("kube")
    except PackageNotFoundError:
        return "0.0.0+local"


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:8.1f}"


def _fmt_throughput(workload: str, throughput: float) -> str:
    unit = "tok/s" if "lm_sequence_iter" in workload else "ops/s"
    if throughput >= 1_000_000:
        return f"{throughput / 1_000_000:5.1f}M {unit}"
    if throughput >= 1_000:
        return f"{throughput / 1_000:7.0f}k {unit}"
    return f"{throughput:7.0f}  {unit}"


def format_table(result: BenchmarkResult) -> str:
    """Return the human-readable benchmark table as a string."""
    lines: list[str] = []
    cfg = result.config
    ver = result.kube_version
    lines.append(f"kube benchmark  v{ver}")
    lines.append(
        f"size={cfg['size']}³  s={cfg['states']} states  {cfg['iterations']} iters"
    )
    sep = "─" * 72
    lines.append(sep)
    lines.append(f"{'Workload':<22} {'Backend':<16} {'Throughput':>16} {'Best ms':>9} {'Mean ms':>9}")
    lines.append(sep)

    prev_workload = ""
    for r in result.results:
        if r.workload != prev_workload and prev_workload:
            lines.append("")
        prev_workload = r.workload

        label = r.workload
        if r.size_label:
            label = f"{label}[{r.size_label}]"

        lines.append(
            f"{label:<22} "
            f"{r.backend:<16} "
            f"{_fmt_throughput(r.workload, r.timing.throughput):>16} "
            f"{_fmt_ms(r.timing.best_s):>9} "
            f"{_fmt_ms(r.timing.mean_s):>9}"
        )

    lines.append(sep)
    return "\n".join(lines)


def to_json_dict(result: BenchmarkResult) -> dict[str, Any]:
    """Serialize a :class:`BenchmarkResult` to a JSON-safe dict."""
    rows = []
    for r in result.results:
        rows.append(
            {
                "workload": r.workload,
                "backend": r.backend,
                "size_label": r.size_label,
                "notes": r.notes,
                "throughput": r.timing.throughput,
                "best_ms": r.timing.best_s * 1000,
                "mean_ms": r.timing.mean_s * 1000,
                "ops": r.timing.ops,
                "checksum": r.timing.checksum,
                "samples_ms": [s * 1000 for s in r.timing.samples],
            }
        )
    return {
        "kube_version": result.kube_version,
        "timestamp": result.timestamp,
        "config": result.config,
        "results": rows,
    }


def plot_results(result: BenchmarkResult, output_path: str) -> None:
    """Save a throughput bar chart to *output_path* (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for --plot; install it with: pip install matplotlib"
        ) from exc

    labels = [f"{r.workload}\n{r.backend}" for r in result.results]
    throughputs = [r.timing.throughput / 1_000_000 for r in result.results]

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), 5))
    ax.bar(range(len(labels)), throughputs)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Throughput (M ops/s)")
    cfg = result.config
    ax.set_title(
        f"kube benchmark v{result.kube_version}  "
        f"[size={cfg['size']}³  s={cfg['states']} states]"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
