"""Table formatter, JSON serializer, and optional matplotlib plot."""

from __future__ import annotations

from typing import Any

from kube.benchmark._runner import BenchmarkResult


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:9.1f}"


def _fmt_throughput(throughput: float, unit: str) -> str:
    if throughput >= 1_000_000:
        return f"{throughput / 1_000_000:6.1f}M {unit}"
    if throughput >= 1_000:
        return f"{throughput / 1_000:8.0f}k {unit}"
    return f"{throughput:8.0f}  {unit}"


def format_table(result: BenchmarkResult) -> str:
    """Return the human-readable benchmark table as a string."""
    lines: list[str] = []
    cfg = result.config
    ver = result.kube_version
    lines.append(f"kube benchmark  v{ver}")
    lines.append(f"size={cfg['size']}  states={cfg['states']}  {cfg['iterations']} iters")
    sep = "-" * 78
    lines.append(sep)
    lines.append(f"{'Scenario':<18} {'Backend':<32} {'Throughput':>16} {'Best ms':>10} {'Mean ms':>10}")
    lines.append(sep)

    prev_scenario = ""
    for r in result.results:
        if r.scenario != prev_scenario and prev_scenario:
            lines.append("")
        prev_scenario = r.scenario

        lines.append(
            f"{r.scenario:<18} "
            f"{r.backend:<32} "
            f"{_fmt_throughput(r.timing.throughput, r.unit):>16} "
            f"{_fmt_ms(r.timing.best_s):>10} "
            f"{_fmt_ms(r.timing.mean_s):>10}"
        )

    lines.append(sep)
    return "\n".join(lines)


def to_json_dict(result: BenchmarkResult) -> dict[str, Any]:
    """Serialize a :class:`BenchmarkResult` to a JSON-safe dict."""
    rows = []
    for r in result.results:
        rows.append(
            {
                "scenario": r.scenario,
                "backend": r.backend,
                "unit": r.unit,
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

    labels = [f"{r.scenario}\n{r.backend}" for r in result.results]
    throughputs = [r.timing.throughput for r in result.results]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(range(len(labels)), throughputs)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Throughput (ops/s, log scale)")
    cfg = result.config
    ax.set_title(f"kube benchmark v{result.kube_version}  [size={cfg['size']}  states={cfg['states']}]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
