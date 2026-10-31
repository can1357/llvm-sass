"""Bit-for-bit differential tests against nvcc for kernels/globals.cu:
dynamic shared memory (launch sharedMemBytes), __constant__ data (one table
written through cuModuleGetGlobal, the rest initializer-only) and by-value
struct / __grid_constant__ parameters packed by hand.

    python tools/globals_test.py --ours out/globals --nvcc globals.nvcc.cubin [-k dsmem]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cudrv import Driver, ptr, scalar  # noqa: E402

KERNELS = ["dsmem", "dyn_only", "dyn_alias", "dyn_big", "const_ops", "const_init",
           "const_ptr", "struct_ops", "params_mix", "psel", "pmod"]
drv = Driver()


def pad8(b):
    return b + b"\0" * (-len(b) % 8)


def P(o, a, n, s):
    return struct.pack("<QQif", o, a, n, s)


BIG = np.arange(20, dtype=np.float32) * 1.25 - 3.0
TMB = (np.arange(128) * 37 % 251 - 125).astype(np.int8)


def run(mod, name):
    fn = drv.function(mod, name)
    if name == "dsmem":
        a = drv.upload(np.linspace(-3, 5, 128, dtype=np.float32))
        o = drv.alloc(128 * 4)
        drv.launch(fn, (1, 1, 1), (128, 1, 1), [ptr(o), ptr(a)], shared=128 * 4)
        drv.sync()
        return [drv.read(o, np.zeros(128, np.float32))]
    if name == "dyn_only":
        a = drv.upload(np.linspace(-7, 9, 1024, dtype=np.float32))
        o = drv.alloc(1024 * 4)
        drv.launch(fn, (1, 1, 1), (256, 1, 1), [ptr(o), ptr(a)], shared=256 * 16)
        drv.sync()
        return [drv.read(o, np.zeros(1024, np.float32))]
    if name == "dyn_alias":
        o = drv.alloc(256 * 4)
        drv.launch(fn, (1, 1, 1), (256, 1, 1), [ptr(o)], shared=256 * 4)
        drv.sync()
        return [drv.read(o, np.zeros(256, np.int32))]
    if name == "dyn_big":
        o = drv.alloc(256 * 4)
        drv.max_dynamic_shared(fn, 65536)
        drv.launch(fn, (1, 1, 1), (256, 1, 1), [ptr(o)], shared=65536)
        drv.sync()
        return [drv.read(o, np.zeros(256, np.int32))]
    if name == "const_ops":
        g, size = drv.global_(mod, "cscal")
        assert size == 4, size
        drv.write(g, np.array([7], np.int32))
        o = drv.alloc(64 * 4)
        drv.launch(fn, (1, 1, 1), (64, 1, 1), [ptr(o)])
        drv.sync()
        out = [drv.read(o, np.zeros(64, np.float32))]
        # also read back the initialized table through cuModuleGetGlobal
        g2, size2 = drv.global_(mod, "ctab")
        out.append(drv.read(g2, np.zeros(size2 // 4, np.float32)))
        return out
    if name == "const_init":
        o = drv.alloc(512 * 4)
        q = drv.alloc(256 * 8)
        drv.launch(fn, (1, 1, 1), (256, 1, 1), [ptr(o), ptr(q), scalar(np.int32, 13)])
        drv.sync()
        return [drv.read(o, np.zeros(512, np.int32)), drv.read(q, np.zeros(256, np.int64))]
    if name == "const_ptr":
        o = drv.alloc(256 * 4)
        drv.launch(fn, (1, 1, 1), (256, 1, 1), [ptr(o), scalar(np.int32, 11)])
        drv.sync()
        return [drv.read(o, np.zeros(256, np.float32))]
    if name == "psel":
        o = drv.alloc(256 * 4)
        drv.launch(fn, (1, 1, 1), (256, 1, 1), [ptr(o), BIG.tobytes()])
        drv.sync()
        return [drv.read(o, np.zeros(256, np.float32))]
    if name == "struct_ops":
        a = drv.upload(np.linspace(-2, 3, 256, dtype=np.float32))
        o = drv.upload(np.full(256, -1, np.float32))
        drv.launch(fn, (1, 1, 1), (256, 1, 1), [P(o, a, 200, 1.5), BIG.tobytes()])
        drv.sync()
        return [drv.read(o, np.zeros(256, np.float32))]
    if name == "params_mix":
        a = drv.upload(np.linspace(-2, 3, 256, dtype=np.float32))
        o = drv.alloc(256 * 4)
        oi = drv.alloc(256 * 4)
        params = [pad8(struct.pack("<b", -3)), P(o, a, 11, -0.75), BIG.tobytes(),
                  struct.pack("<q", 0x123456789ab), struct.pack("<4f", 1.5, -2.5, 3.25, 100.0),
                  pad8(struct.pack("<3b", 5, -6, 7)), pad8(struct.pack("<h", -1234)),
                  TMB.tobytes(), pad8(struct.pack("<i", 99999)), ptr(oi)]
        drv.launch(fn, (1, 1, 1), (256, 1, 1), params)
        drv.sync()
        return [drv.read(o, np.zeros(256, np.float32)), drv.read(oi, np.zeros(256, np.int32))]
    if name == "pmod":
        o = drv.alloc(256 * 4)
        drv.launch(fn, (1, 1, 1), (256, 1, 1), [ptr(o), BIG.tobytes(), P(0, 0, 40, 0.0)])
        drv.sync()
        return [drv.read(o, np.zeros(256, np.int32))]
    raise KeyError(name)


ap = argparse.ArgumentParser()
ap.add_argument("--ours", required=True, help="prefix of our <prefix>.<kernel>.cubin files")
ap.add_argument("--nvcc", required=True, type=Path, help="nvcc -cubin build of globals.cu")
ap.add_argument("-k", action="append", help="kernel(s) to run (default: all)")
opts = ap.parse_args()
nv_mod = drv.module(opts.nvcc.read_bytes())
ok = True
for name in opts.k or KERNELS:
    try:
        ref = run(nv_mod, name)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: nvcc run failed: {e}")
        ok = False
        continue
    try:
        got = run(drv.module(Path(f"{opts.ours}.{name}.cubin").read_bytes()), name)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: FAIL (ours: {e})")
        ok = False
        continue
    same = all(r.tobytes() == g.tobytes() for r, g in zip(ref, got))
    ok &= same
    print(f"{name}: {'bit-identical' if same else 'MISMATCH'}"
          f"  ({sum(r.size for r in ref)} values; sample nvcc={ref[0][:4]} ours={got[0][:4]})")
    if not same:
        for r, g in zip(ref, got):
            idx = [i for i in range(r.size) if r.flat[i].tobytes() != g.flat[i].tobytes()][:8]
            print("   first diffs:", [(i, r.flat[i], g.flat[i]) for i in idx])
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
