"""Benchmark Kube against a one-bit binary occupancy grid.

This example exists to make the tradeoff between Kube's cell/face/state model
and a simple binary grid concrete.

The benchmark is intentionally not framed as "which one is universally
better." They solve different problems:

- binary occupancy stores one undirected bit per cell
- Kube stores six oriented face states per cell

What this script measures:

1. Storage math for the chosen grid shape.
2. Random write throughput.
3. Random read throughput for native Kube access.

What this script explains:

- binary is smaller because it stores less information
- Kube costs more per cell because it stores a richer local state model
- Kube's value is not just speed, but directional expressiveness plus direct
  random access over packed storage
- Kube now always uses its native storage engine for packed random IO

Typical usage:

    python examples/benchmark_kube_vs_binary.py

    python examples/benchmark_kube_vs_binary.py --shape 128x128x64 --states 4 --ops 200000
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

from kube import Kube


@dataclass
class TimingResult:
    """Store repeated timing samples for one benchmark case."""

    samples: list[float]
    checksum: int

    @property
    def best(self) -> float:
        """Return the fastest recorded sample in seconds."""
        return min(self.samples)

    @property
    def mean(self) -> float:
        """Return the average timing across repeats in seconds."""
        return statistics.mean(self.samples)


class BinaryOccupancyGrid:
    """Tiny one-bit-per-cell baseline used for comparison.

    This baseline exists to represent the classic binary system: one cell,
    one occupancy bit, no face orientation, no multi-state measurements.
    """

    def __init__(self, width: int, height: int, depth: int) -> None:
        """Allocate a packed bitset for a rectangular occupancy grid."""
        self.W = int(width)
        self.H = int(height)
        self.D = int(depth)
        self.cell_count = self.W * self.H * self.D
        self.data = bytearray((self.cell_count + 7) // 8)

    def _flat_index(self, x: int, y: int, z: int) -> int:
        """Map one 3D cell coordinate into the flat bitset index."""
        return x * (self.H * self.D) + y * self.D + z

    def set_cell(self, x: int, y: int, z: int, value: int) -> None:
        """Write one occupancy bit."""
        index = self._flat_index(x, y, z)
        byte_index = index >> 3
        bit_index = index & 7
        mask = 1 << bit_index
        if value:
            self.data[byte_index] |= mask
        else:
            self.data[byte_index] &= 0xFF ^ mask

    def get_cell(self, x: int, y: int, z: int) -> int:
        """Read one occupancy bit."""
        index = self._flat_index(x, y, z)
        byte_index = index >> 3
        bit_index = index & 7
        return (self.data[byte_index] >> bit_index) & 1

    @property
    def nbytes(self) -> int:
        """Return the packed storage bytes used by the bitset."""
        return len(self.data)


def parse_shape(text: str) -> tuple[int, int, int]:
    """Parse `AxBxC` CLI shape syntax."""
    normalized = text.lower().replace(",", "x")
    parts = [part.strip() for part in normalized.split("x")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must look like 64x64x64")
    return tuple(int(part) for part in parts)


def parse_args() -> argparse.Namespace:
    """Define the CLI for the benchmark example."""
    parser = argparse.ArgumentParser(
        description="Benchmark Kube's cell/face/state model against a simple binary occupancy grid"
    )
    parser.add_argument("--shape", type=parse_shape, default=(64, 64, 64), help="Grid shape as AxBxC")
    parser.add_argument(
        "--states",
        type=int,
        default=4,
        help="Number of states per Kube face; binary baseline always stays 1 bit per cell",
    )
    parser.add_argument("--ops", type=int, default=200_000, help="Random operations per timing run")
    parser.add_argument("--repeats", type=int, default=5, help="Measured repeats per benchmark")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic seed for pseudo-random operations")
    return parser.parse_args()


def xorshift32(state: int) -> int:
    """Return the next deterministic pseudo-random integer."""
    state &= 0xFFFFFFFF
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= (state >> 17) & 0xFFFFFFFF
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def build_ops(
    width: int,
    height: int,
    depth: int,
    states: int,
    count: int,
    seed: int,
) -> list[tuple[int, int, int, int, int]]:
    """Build a deterministic sequence of random access operations."""
    state = int(seed) or 1
    ops: list[tuple[int, int, int, int, int]] = []
    for _ in range(int(count)):
        state = xorshift32(state)
        x = state % width
        state = xorshift32(state)
        y = state % height
        state = xorshift32(state)
        z = state % depth
        state = xorshift32(state)
        face = state % 6
        state = xorshift32(state)
        value = state % states
        ops.append((x, y, z, face, value))
    return ops


def measure(fn, *, repeats: int) -> TimingResult:
    """Time a callable repeatedly and keep its checksum alive."""
    samples: list[float] = []
    checksum = 0
    for _ in range(int(repeats)):
        start = time.perf_counter()
        checksum = int(fn())
        samples.append(time.perf_counter() - start)
    return TimingResult(samples=samples, checksum=checksum)


def benchmark_binary_writes(grid: BinaryOccupancyGrid, ops: list[tuple[int, int, int, int, int]]) -> int:
    """Apply random occupancy writes to the binary baseline."""
    checksum = 0
    for x, y, z, _, value in ops:
        bit = value & 1
        grid.set_cell(x, y, z, bit)
        checksum ^= bit
    return checksum


def benchmark_binary_reads(grid: BinaryOccupancyGrid, ops: list[tuple[int, int, int, int, int]]) -> int:
    """Read random occupancy bits from the binary baseline."""
    checksum = 0
    for x, y, z, _, _ in ops:
        checksum += grid.get_cell(x, y, z)
    return checksum


def benchmark_kube_writes(kube: Kube, compiled_ops, ops: list[tuple[int, int, int, int, int]]) -> int:
    """Apply random face-state writes to Kube.

    The benchmark intentionally uses the batched API so the native path can
    amortize bridge overhead instead of paying one `ctypes` call per face edit.
    """
    kube.set_faces_batch(compiled_ops)
    checksum = 0
    for _, _, _, _, value in ops:
        checksum ^= value
    return checksum


def benchmark_kube_reads(kube: Kube, compiled_ops) -> int:
    """Read random face states from Kube through the batched API."""
    return sum(kube.get_faces_batch(compiled_ops))


def fmt_int(value: int) -> str:
    """Format an integer with separators for CLI output."""
    return f"{int(value):,}"


def fmt_seconds(value: float) -> str:
    """Format a duration compactly for CLI output."""
    return f"{value:.6f}s"


def fmt_rate(ops: int, seconds: float) -> str:
    """Format operations-per-second for CLI output."""
    if seconds <= 0:
        return "inf"
    return f"{ops / seconds:,.0f} ops/s"


def print_storage_summary(
    shape: tuple[int, int, int],
    states: int,
    binary: BinaryOccupancyGrid,
    kube: Kube,
) -> None:
    """Print the representational and storage differences between the models."""
    cell_count = shape[0] * shape[1] * shape[2]
    kube_stats = kube.memory_stats()
    binary_local_states = 2
    kube_local_states = states**6

    print("representation")
    print("==============")
    print(f"shape                     : {shape[0]} x {shape[1]} x {shape[2]}")
    print(f"cells                     : {fmt_int(cell_count)}")
    print()
    print("binary occupancy baseline")
    print("-------------------------")
    print("model                     : 1 undirected occupancy bit per cell")
    print(f"bits per cell             : 1")
    print(f"payload bytes             : {fmt_int(binary.nbytes)}")
    print(f"local states per cell     : {fmt_int(binary_local_states)}")
    print()
    print("kube cell/face/state model")
    print("--------------------------")
    print(f"model                     : 6 oriented faces per cell, {states} states per face")
    print(f"bits per face             : {kube_stats['bits_per_face']}")
    print(f"bits per cell             : {kube.bits_per_cell}")
    print(f"payload bytes             : {fmt_int(kube_stats['packed_bytes'])}")
    print(f"local states per cell     : {fmt_int(kube_local_states)}")
    print(f"storage engine            : {kube.memory_stats()['acceleration']}")
    print()
    print("tradeoff")
    print("--------")
    print(f"storage multiplier        : {kube_stats['packed_bytes'] / binary.nbytes:.2f}x binary")
    print(f"state-space multiplier    : {kube_local_states / binary_local_states:,.0f}x binary")


def print_timing_table(
    ops: int,
    binary_writes: TimingResult,
    kube_writes: TimingResult,
    binary_reads: TimingResult,
    kube_reads: TimingResult,
) -> None:
    """Print a small benchmark table."""
    print()
    print("timings")
    print("=======")
    print(f"{'case':18} {'binary best':>14} {'kube best':>14} {'binary rate':>16} {'kube rate':>16}")
    print(f"{'-'*18} {'-'*14} {'-'*14} {'-'*16} {'-'*16}")
    print(
        f"{'random writes':18} "
        f"{fmt_seconds(binary_writes.best):>14} "
        f"{fmt_seconds(kube_writes.best):>14} "
        f"{fmt_rate(ops, binary_writes.best):>16} "
        f"{fmt_rate(ops, kube_writes.best):>16}"
    )
    print(
        f"{'random reads':18} "
        f"{fmt_seconds(binary_reads.best):>14} "
        f"{fmt_seconds(kube_reads.best):>14} "
        f"{fmt_rate(ops, binary_reads.best):>16} "
        f"{fmt_rate(ops, kube_reads.best):>16}"
    )
    print()
    print("interpretation")
    print("==============")
    print("Binary wins on raw storage cost because it stores less information.")
    print("Kube uses its native storage engine for packed random IO.")
    print("Kube spends more bits and usually more CPU per operation because each cell")
    print("carries six oriented face measurements instead of one occupancy bit.")
    print("The gain is that Kube can represent directional local state directly,")
    print("which a one-bit binary grid cannot express without extra side structures.")


def main() -> None:
    """Run the benchmark and print a human-readable comparison."""
    args = parse_args()
    width, height, depth = args.shape
    ops = build_ops(width, height, depth, args.states, args.ops, args.seed)

    binary = BinaryOccupancyGrid(width, height, depth)
    kube = Kube(width, height, depth, states=args.states)
    compiled = kube.compile_face_batch(ops)

    print_storage_summary(args.shape, args.states, binary, kube)

    binary_writes = measure(lambda: benchmark_binary_writes(binary, ops), repeats=args.repeats)
    kube_writes = measure(lambda: benchmark_kube_writes(kube, compiled, ops), repeats=args.repeats)
    binary_reads = measure(lambda: benchmark_binary_reads(binary, ops), repeats=args.repeats)
    kube_reads = measure(lambda: benchmark_kube_reads(kube, compiled), repeats=args.repeats)

    print_timing_table(args.ops, binary_writes, kube_writes, binary_reads, kube_reads)


if __name__ == "__main__":
    main()
