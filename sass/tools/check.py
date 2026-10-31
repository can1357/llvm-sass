"""Correctness suite for kernels/tests.cu on the GPU.

Each kernel computes out[i] = f(a[i], b[i]) for i < n. Our per-kernel cubins
(`<dir>/tests.<kernel>.cubin` from tools/sassc.py) are checked against a
NumPy reference (integers and comparisons exactly, FP within 1e-6 relative)
and bit-for-bit against the nvcc build of the same source.

    python tools/check.py --ours out/tests --nvcc kernels/tests.nvcc.cubin [-k t_add]
"""
from __future__ import annotations

import argparse
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cudrv import Driver, ptr, scalar  # noqa: E402

i8, u8, i16, u16 = np.int8, np.uint8, np.int16, np.uint16
i32, u32, i64, u64, f32 = np.int32, np.uint32, np.int64, np.uint64, np.float32
N = 1000
BLOCK = 128


@dataclass
class Spec:
    a: type
    b: type
    out: type
    ref: Callable[[np.ndarray, np.ndarray], np.ndarray]
    fp_range: float = 1e3  # magnitude of random FP inputs


def rotl(a, s):
    s = s & 31
    return np.where(s == 0, a, (a << s) | (a >> ((32 - s) & 31)))


def fma32(a, b, c):
    return (a.astype(np.float64) * b.astype(np.float64) + np.float64(c)).astype(f32)


def loop_ref(x, b):
    acc = np.zeros_like(x)
    k = b & 15
    for j in range(16):
        acc = np.where(j < k, fma32(acc, x, np.float64(j)), acc)
    return acc


def diamond_ref(a, b):
    r = np.where(a > b, a * i32(3) - b, (b << 2) ^ a)
    return np.where(r & 1, r + i32(11), r)


def collatz_ref(a, b):
    x = a | u32(1)
    steps = np.zeros_like(x)
    for _ in range(64):
        live = x != 1
        x = np.where(live, np.where(x & 1, u32(3) * x + u32(1), x >> 1), x)
        steps += live.astype(u32)
    return steps * u32(1000) + (x & u32(0xFF)) + (b & u32(3))


def geom_ref(a, b):
    i = np.arange(len(a), dtype=i32)
    grid = (len(a) + BLOCK - 1) // BLOCK
    return (i % BLOCK) + (i // BLOCK) * 1000 + BLOCK * 7 + grid * 13


def tile_ref(a, b):
    i = np.arange(N)
    j = i // BLOCK * BLOCK + BLOCK - 1 - i % BLOCK
    return np.where(j < N, a[np.minimum(j, N - 1)], 0) + b


SPECS: dict[str, Spec] = {
    "t_add": Spec(i32, i32, i32, lambda a, b: a + b),
    "t_sub": Spec(i32, i32, i32, lambda a, b: a - b),
    "t_neg": Spec(i32, i32, i32, lambda a, b: -a),
    "t_add3": Spec(i32, i32, i32, lambda a, b: a + b + (a >> 3)),
    "t_mul": Spec(i32, i32, i32, lambda a, b: a * b),
    "t_mad": Spec(i32, i32, i32, lambda a, b: a * b + (b ^ i32(0x55))),
    "t_mulimm": Spec(i32, i32, i32, lambda a, b: a * i32(37) - b * i32(3)),
    "t_mulhi_u": Spec(u32, u32, u32, lambda a, b: ((a.astype(u64) * b.astype(u64)) >> u64(32)).astype(u32)),
    "t_mulhi_s": Spec(i32, i32, i32, lambda a, b: ((a.astype(i64) * b.astype(i64)) >> 32).astype(i32)),
    "t_logic": Spec(u32, u32, u32, lambda a, b: (a & b) ^ (a | u32(0xF0F0)) ^ ~b),
    "t_andn": Spec(u32, u32, u32, lambda a, b: a & ~b),
    "t_xor3": Spec(u32, u32, u32, lambda a, b: a ^ b ^ (a >> u32(7))),
    "t_shl": Spec(u32, u32, u32, lambda a, b: a << (b & u32(31))),
    "t_shr": Spec(u32, u32, u32, lambda a, b: a >> (b & u32(31))),
    "t_sar": Spec(i32, i32, i32, lambda a, b: a >> (b & 31)),
    "t_shimm": Spec(u32, u32, u32, lambda a, b: (a << u32(5)) + (a >> u32(27)) + (b.view(i32) >> 9).view(u32)),
    "t_rotl": Spec(u32, u32, u32, lambda a, b: rotl(a, b)),
    "t_rotimm": Spec(u32, u32, u32, lambda a, b: (a << u32(13)) | (a >> u32(19))),
    "t_minmax": Spec(i32, i32, i32, lambda a, b: np.minimum(a, b) - np.maximum(a, i32(7))),
    "t_uminmax": Spec(u32, u32, u32, lambda a, b: np.minimum(a, b) + np.maximum(a, b >> u32(1))),
    "t_cmpsel": Spec(i32, i32, i32, lambda a, b: np.where(a < b, a, b + i32(1))),
    "t_ucmpsel": Spec(u32, u32, u32, lambda a, b: np.where(a <= b, a ^ b, u32(3))),
    "t_eqsel": Spec(i32, i32, i32, lambda a, b: (a == b) + (a != 0) * 2 + (b > 5) * 4),
    "t_bool": Spec(i32, i32, i32, lambda a, b: ((a < b) & (b < 0)) | (a == 7)),
    "t_sext8": Spec(i32, i32, i32, lambda a, b: a.astype(i8).astype(i32) + b.astype(i16).astype(i32)),
    "t_add64": Spec(i64, i64, i64, lambda a, b: a + b),
    "t_sub64": Spec(i64, i64, i64, lambda a, b: a - b),
    "t_mul64": Spec(i64, i64, i64, lambda a, b: a * b),
    "t_widemul": Spec(i32, i32, i64, lambda a, b: a.astype(i64) * b.astype(i64) + 5),
    "t_wideumul": Spec(u32, u32, u64, lambda a, b: a.astype(u64) * b.astype(u64)),
    "t_shl64": Spec(u64, u64, u64, lambda a, b: a << (b & u64(63))),
    "t_shr64": Spec(u64, u64, u64, lambda a, b: a >> (b & u64(63))),
    "t_sar64": Spec(i64, i64, i64, lambda a, b: a >> (b & 63)),
    "t_shimm64": Spec(u64, u64, u64, lambda a, b: (a << u64(3)) ^ (a >> u64(40)) ^ (b.view(i64) >> 33).view(u64) ^ (b << u64(32))),
    "t_logic64": Spec(u64, u64, u64, lambda a, b: (a & b) | (a ^ ~b)),
    "t_cmp64": Spec(i64, i64, i32, lambda a, b: (a < b) + (a <= b) * 2 + (a == b) * 4 + (a != b) * 8 + (a > b) * 16 + (a >= b) * 32),
    "t_ucmp64": Spec(u64, u64, i32, lambda a, b: (a < b) + (a <= b) * 2 + (a > b) * 4 + (a >= b) * 8),
    "t_sel64": Spec(i64, i64, i64, lambda a, b: np.where(a > b, a, b * i64(3))),
    "t_ext64": Spec(i32, u32, i64, lambda a, b: a.astype(i64) + b.astype(i64)),
    "t_trunc64": Spec(i64, i64, i32, lambda a, b: (a >> 17).astype(i32) + b.astype(i32)),
    "t_fma": Spec(f32, f32, f32, lambda a, b: fma32(a, b, 1.5)),
    "t_fadd": Spec(f32, f32, f32, lambda a, b: a + b),
    "t_fsub": Spec(f32, f32, f32, lambda a, b: fma32(-b, np.full_like(b, 0.25), a.astype(np.float64))),
    "t_fneg": Spec(f32, f32, f32, lambda a, b: -a * b),
    "t_fabs": Spec(f32, f32, f32, lambda a, b: np.abs(a) + b),
    "t_fminmax": Spec(f32, f32, f32, lambda a, b: np.fmin(a, b) - np.fmax(a, f32(2))),
    "t_fcmp": Spec(f32, f32, i32, lambda a, b: (a < b) + (a <= b) * 2 + (a > b) * 4 + (a >= b) * 8 + (a == b) * 16 + (a != b) * 32),
    "t_fcmpu": Spec(f32, f32, i32, lambda a, b: (~(a < b)) * 1 + (~(a >= b)) * 2 + (a != a) * 4),
    "t_fsel": Spec(f32, f32, f32, lambda a, b: np.where(a > b, a * f32(2), b)),
    "t_i2f": Spec(i32, u32, f32, lambda a, b: a.astype(f32) + b.astype(f32)),
    "t_f2i": Spec(f32, f32, i32, lambda a, b: np.trunc(a).astype(i32) + np.trunc(np.abs(b)).astype(u32).astype(i32), fp_range=1e6),
    "t_bits": Spec(f32, i32, i32, lambda a, b: a.view(i32) ^ b.astype(f32).view(i32)),
    "t_loop": Spec(f32, i32, f32, loop_ref, fp_range=1.0),
    "t_diamond": Spec(i32, i32, i32, diamond_ref),
    "t_while": Spec(u32, u32, u32, collatz_ref),
    "t_bytes": Spec(u8, i8, i16, lambda a, b: (a.astype(i32) * 3 + b.astype(i32)).astype(i16)),
    "t_halves": Spec(u16, i16, u8, lambda a, b: (a.astype(i32) + b.astype(i32) * 5).astype(u8)),
    "t_geom": Spec(i32, i32, i32, geom_ref),
    "t_tile": Spec(i32, i32, i32, tile_ref),
}


def inputs(dtype, rng: np.random.Generator, fp_range: float) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        x = (rng.standard_normal(N) * fp_range).astype(dtype)
        if fp_range <= 1e3:
            x[:8] = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -2.0, 3.5]
        return x
    info = np.iinfo(dtype)
    x = rng.integers(info.min, info.max, N, dtype=dtype, endpoint=True)
    small = rng.integers(-40 if info.min < 0 else 0, 40, N // 4).astype(dtype)
    x[: len(small)] = small
    edges = [0, 1, info.max, info.min, 7, 5] + ([-1] if info.min < 0 else [])
    x[len(small): len(small) + len(edges)] = np.array(edges).astype(dtype)
    return x


def compare(name: str, got: np.ndarray, ref: np.ndarray) -> str | None:
    if np.issubdtype(got.dtype, np.floating):
        ref = ref.astype(got.dtype)
        both_nan = np.isnan(got) & np.isnan(ref)
        close = np.isclose(got, ref, rtol=1e-6, atol=0) | both_nan | (got == ref)
        bad = np.flatnonzero(~close)
    else:
        bad = np.flatnonzero(got != ref.astype(got.dtype))
    if bad.size == 0:
        return None
    i = bad[0]
    return f"{bad.size} mismatches; first i={i}: got {got[i]!r}, want {ref.astype(got.dtype)[i]!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--nvcc", type=Path, required=True)
    ap.add_argument("-k", "--kernel", action="append")
    ap.add_argument("--timeout", type=int, default=30, help="hard wall-clock limit, seconds")
    opts = ap.parse_args()
    signal.alarm(opts.timeout)  # A hung kernel kills the run instead of the GPU.

    cu = Driver()
    nvcc_mod = cu.module(opts.nvcc.read_bytes())
    rng = np.random.default_rng(1)
    failures = 0
    for name, spec in SPECS.items():
        if opts.kernel and name not in opts.kernel:
            continue
        a = inputs(spec.a, rng, spec.fp_range)
        b = inputs(spec.b, rng, spec.fp_range)
        with np.errstate(all="ignore"):
            want = np.asarray(spec.ref(a, b))
        pa, pb = cu.upload(a), cu.upload(b)
        like = np.zeros(N, spec.out)
        outs = {}
        try:
            fns = {"nvcc": cu.function(nvcc_mod, name),
                   "ours": cu.load((opts.ours / f"tests.{name}.cubin").read_bytes(), name)}
            for label, fn in fns.items():
                po = cu.alloc(like.nbytes)
                cu.fill(po, 0xAB, like.nbytes)
                cu.launch(fn, ((N + BLOCK - 1) // BLOCK, 1, 1), (BLOCK, 1, 1),
                          [ptr(pa), ptr(pb), ptr(po), scalar(i32, N)])
                cu.sync()
                outs[label] = cu.read(po, like)
                cu.free(po)
        except Exception as e:  # noqa: BLE001 - report and continue the suite
            print(f"{name:12} ERROR {e}")
            failures += 1
            continue
        finally:
            cu.free(pa)
            cu.free(pb)
        err_ref = compare(name, outs["ours"], want)
        err_nvcc = compare(name, outs["nvcc"], want)
        same = outs["ours"].tobytes() == outs["nvcc"].tobytes()
        status = "ok" if err_ref is None and same else "FAIL"
        failures += status != "ok"
        detail = []
        if err_ref:
            detail.append(f"vs ref: {err_ref}")
        if not same:
            detail.append("differs from nvcc" + (f" (nvcc vs ref: {err_nvcc})" if err_nvcc else ""))
        print(f"{name:12} {status} {'; '.join(detail)}")
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
