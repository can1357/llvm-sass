"""Correctness check and A/B timing of one kernel compiled several ways.

Runs on the GPU host. Each `--cubin label=path` is checked against a NumPy
reference and bit-for-bit against the first (baseline) variant, then timed:
variants alternate sample by sample so clock and thermal drift hit all of
them equally, and each sample is the mean of `--inner` launches.

    python tools/bench.py saxpy --cubin nvcc=k.nvcc.cubin --cubin ours=k.ours.cubin
    python tools/bench.py fmix --rounds 4096 --warps-per-sm 4 --cubin ...
"""
from __future__ import annotations

import argparse
import signal
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cudrv import Driver, ptr, scalar  # noqa: E402

SM_COUNT_ATTR = 16  # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT


@dataclass
class Case:
    """One launch configuration: packed parameters, output reset/readback and a reference."""

    params: list[bytes]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    reset: Callable[[], None]
    output: Callable[[], np.ndarray]
    want: np.ndarray
    rtol: float = 1e-6


def mismatch(got: np.ndarray, want: np.ndarray, rtol: float) -> str | None:
    """Elements off by more than `rtol` (exact for integers)."""
    if np.issubdtype(got.dtype, np.integer):
        bad = np.flatnonzero(got != want.astype(got.dtype))
    else:
        scale = max(float(np.abs(want).max()), 1e-30)
        bad = np.flatnonzero(np.abs(got.astype(np.float64) - want) > rtol * np.maximum(np.abs(want), scale * 1e-3))
    return None if bad.size == 0 else f"{bad.size} mismatches, first at {bad[0]}: {got.flat[bad[0]]} != {want.flat[bad[0]]}"


def saxpy(cu: Driver, opts) -> Case:
    n = opts.n
    rng = np.random.default_rng(0)
    a = np.float32(1.75)
    x = rng.standard_normal(n, dtype=np.float32)
    y0 = rng.standard_normal(n, dtype=np.float32)
    px, py = cu.upload(x), cu.upload(y0)
    block = 256
    return Case(
        [scalar(np.int32, n), scalar(np.float32, a), ptr(px), ptr(py)],
        ((n + block - 1) // block, 1, 1),
        (block, 1, 1),
        reset=lambda: cu.write(py, y0),
        output=lambda: cu.read(py, y0),
        want=np.float64(a) * x + y0,
    )


def fmix_ref(x: np.ndarray, rounds: int) -> np.ndarray:
    x = x.copy()
    for r in range(rounds):
        x ^= x >> np.uint32(16)
        x *= np.uint32(0x85EBCA6B)
        x ^= x >> np.uint32(13)
        x *= np.uint32(0xC2B2AE35)
        x ^= x >> np.uint32(16)
        x += np.uint32(r)
    return x


def xorshift_ref(x: np.ndarray, rounds: int) -> np.ndarray:
    x = x | np.uint32(1)
    y = np.full_like(x, 0x9E3779B9)
    for _ in range(rounds):
        x ^= x << np.uint32(13)
        x ^= x >> np.uint32(17)
        x ^= x << np.uint32(5)
        y = ((y << np.uint32(7)) | (y >> np.uint32(25))) + x
    return x ^ y


def horner_ref(x: np.ndarray, rounds: int) -> np.ndarray:
    acc = np.ones_like(x)
    xd = x.astype(np.float64)
    for _ in range(rounds):
        acc = (acc.astype(np.float64) * xd + 0.5).astype(np.float32)
        acc = (acc.astype(np.float64) * xd - 0.25).astype(np.float32)
    return acc


CHAINS = {"fmix": (np.uint32, fmix_ref), "xorshift": (np.uint32, xorshift_ref), "horner": (np.float32, horner_ref)}


def attn_fwd(cu: Driver, opts) -> Case:
    """kernels/attn.cu: O = softmax(Q K^T / sqrt(D)) V per head, [H][S][D], D = 64."""
    heads, s, d, bq = opts.heads, opts.seq, 64, 64
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((heads, s, d), dtype=np.float32) for _ in range(3))
    pq, pk, pv = cu.upload(q), cu.upload(k), cu.upload(v)
    po = cu.alloc(q.nbytes)
    scores = np.einsum("hqd,hkd->hqk", q.astype(np.float64), k.astype(np.float64)) / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    want = np.einsum("hqk,hkd->hqd", p / p.sum(axis=-1, keepdims=True), v.astype(np.float64))
    return Case(
        [ptr(pq), ptr(pk), ptr(pv), ptr(po), scalar(np.int32, s), scalar(np.float32, np.log2(np.e) / np.sqrt(d))],
        (s // bq, heads, 1),
        (bq, 1, 1),
        reset=lambda: cu.fill(po, 0, q.nbytes),
        output=lambda: cu.read(po, q),
        want=want,
        rtol=1e-3,  # ex2.approx/rcp.approx softmax vs float64
    )


def chain(name: str):
    dtype, ref = CHAINS[name]

    def make(cu: Driver, opts) -> Case:
        sms = cu.attribute(SM_COUNT_ATTR)
        block = min(opts.warps_per_sm, 32) * 32
        grid = sms * max(1, opts.warps_per_sm // 32)
        n = grid * block
        rng = np.random.default_rng(0)
        if dtype == np.float32:
            x = rng.uniform(-0.9, 0.9, n).astype(np.float32)
        else:
            x = rng.integers(0, 2**32, n, dtype=np.uint32)
        px, po = cu.upload(x), cu.alloc(x.nbytes)
        return Case(
            [ptr(px), ptr(po), scalar(np.int32, opts.rounds)],
            (grid, 1, 1),
            (block, 1, 1),
            reset=lambda: cu.fill(po, 0, x.nbytes),
            output=lambda: cu.read(po, x),
            want=ref(x, opts.rounds),
            rtol=1e-5,
        )

    return make


KERNELS: dict[str, Callable] = {"saxpy": saxpy, "attn_fwd": attn_fwd, **{k: chain(k) for k in CHAINS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", choices=sorted(KERNELS))
    ap.add_argument("--cubin", action="append", required=True, help="label=path")
    ap.add_argument("--n", type=int, default=1 << 24, help="saxpy elements")
    ap.add_argument("--rounds", type=int, default=4096, help="chain rounds")
    ap.add_argument("--warps-per-sm", type=int, default=4, help="chain occupancy (4 = one warp per SMSP)")
    ap.add_argument("--heads", type=int, default=8, help="attention heads")
    ap.add_argument("--seq", type=int, default=2048, help="attention sequence length (multiple of 64)")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=30, help="hard wall-clock limit, seconds")
    ap.add_argument("--no-ref", action="store_true",
                    help="diagnostic variants: require bit-identity with the baseline only")
    opts = ap.parse_args()
    # SIGALRM's default action kills the process even inside a blocking
    # driver call; the driver then tears down the context and any hung kernel.
    signal.alarm(opts.timeout)

    cu = Driver()
    print(f"device: {cu.name}")
    fns = {}
    for spec in opts.cubin:
        label, _, path = spec.partition("=")
        fns[label] = cu.load(Path(path).read_bytes(), opts.kernel)
    case = KERNELS[opts.kernel](cu, opts)

    failed = False
    baseline = None
    for label, fn in fns.items():
        case.reset()
        cu.launch(fn, case.grid, case.block, case.params)
        cu.sync()
        got = case.output()
        err = None if opts.no_ref else mismatch(got, case.want, case.rtol)
        if baseline is not None and not np.array_equal(got, baseline):
            same = ", differs from baseline"
            if opts.no_ref:
                err = "differs from baseline"
        else:
            same = "" if baseline is None else ", bit-identical to baseline"
        baseline = got if baseline is None else baseline
        print(f"{label:>10}: {'ok' if err is None else 'FAIL ' + err}{same}")
        failed |= err is not None
    if failed:
        return 1

    launches = {label: (lambda fn=fn: cu.launch(fn, case.grid, case.block, case.params)) for label, fn in fns.items()}
    samples: dict[str, list[float]] = {label: [] for label in fns}
    for launch in launches.values():
        cu.time_us(launch, opts.inner)
    for _ in range(opts.reps):
        for label, launch in launches.items():
            samples[label].append(cu.time_us(launch, opts.inner))
    base = statistics.median(next(iter(samples.values())))
    for label, xs in samples.items():
        med = statistics.median(xs)
        print(f"{label:>10}: {med:9.2f} us  p10 {np.percentile(xs, 10):9.2f}  p90 {np.percentile(xs, 90):9.2f}  speedup {base / med:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
