"""Bit-for-bit differential tests against nvcc for warp-level kernels.

Every kernel in the module has the signature `(const uint4 *in, uint4 *out)`
and runs as one warp; nvcc's build defines the expected output. Inputs are
random f16 values in [-2, 2) packed two per 32-bit word, valid both as
tensor-core operands and as raw bits (--inputs f16), or f32 values over the
whole range: small and large magnitudes, denormals, zeros, infinities, NaN
(--inputs f32), or uniformly random 32-bit words (--inputs u32).

    python sass/tools/difftest.py --ours out/tc_tests --nvcc tc_tests.nvcc.cubin
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cudrv import Driver, ptr  # noqa: E402

WORDS = 128 * 4  # in/out: 128 uint4


def f32_inputs(rng: np.random.Generator) -> np.ndarray:
    """f32 bit patterns: a quarter in [-8, 8), the rest log-uniform over
    2^-149..2^128 (denormals to overflow) with random signs, then specials."""
    vals = rng.uniform(-8, 8, WORDS).astype(np.float32)
    mag = np.exp2(rng.uniform(-149, 128, WORDS)).astype(np.float32)
    sign = np.where(rng.random(WORDS) < 0.5, -1, 1).astype(np.float32)
    vals[WORDS // 4:] = (mag * sign)[WORDS // 4:]
    specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 0.5, 2.0 ** -126,
                         2.0 ** -127, 2.0 ** 126, 2.0 ** 127, 3.0, -126.0, -127.5, 88.0],
                        np.float32)
    vals[:specials.size] = specials
    return vals.view(np.uint32).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True, help="dir with <stem>.<kernel>.cubin")
    ap.add_argument("--nvcc", type=Path, required=True)
    ap.add_argument("-k", "--kernel", action="append")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--inputs", choices=("f16", "f32", "u32"), default="f16")
    opts = ap.parse_args()
    signal.alarm(opts.timeout)

    cu = Driver()
    nvcc = cu.module(opts.nvcc.read_bytes())
    rng = np.random.default_rng(7)
    if opts.inputs == "f16":
        data = rng.uniform(-2, 2, WORDS * 2).astype(np.float16).view(np.uint32)
    elif opts.inputs == "f32":
        data = f32_inputs(rng)
    else:
        data = rng.integers(0, 2**32, WORDS, dtype=np.uint32)
    pin = cu.upload(data)
    failures = 0
    for path in sorted(opts.ours.glob("*.cubin")):
        name = path.stem.split(".", 1)[1]
        if opts.kernel and name not in opts.kernel:
            continue
        outs = {}
        for label, fn in (("nvcc", cu.function(nvcc, name)), ("ours", cu.load(path.read_bytes(), name))):
            po = cu.alloc(data.nbytes)
            cu.fill(po, 0xAB, data.nbytes)
            cu.launch(fn, (1, 1, 1), (32, 1, 1), [ptr(pin), ptr(po)])
            cu.sync()
            outs[label] = cu.read(po, data)
            cu.free(po)
        bad = np.flatnonzero(outs["ours"] != outs["nvcc"])
        status = "ok" if bad.size == 0 else f"FAIL {bad.size} words differ, first at {bad[0]} " \
            f"(input {data[bad[0]]:#010x}): {outs['ours'][bad[0]]:#010x} != {outs['nvcc'][bad[0]]:#010x}"
        failures += bad.size != 0
        print(f"{name:14} {status}")
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
