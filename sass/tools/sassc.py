"""CUDA -> SASS -> cubin through the LLVM SASS backend.

    python sass/tools/sassc.py sass/kernels/saxpy.cu -o out/        # saxpy.ll, saxpy.sass, saxpy.<kernel>.cubin
    python sass/tools/sassc.py sass/kernels/saxpy.ll -o out/        # start from IR
    python sass/tools/sassc.py --arch sm90 sass/kernels/saxpy.cu    # H100/H200 instead of GB10
    python sass/tools/sassc.py sass/kernels/saxpy.cu --llc-args=-sass-lsu-interval=0

Kernels in one module are assembled into separate cubins (the assembler takes
one `#fn` block at a time). Dependency-checker warnings from the assembler
are errors here: they mean the backend's control codes are wrong.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
from pathlib import Path

# The root of an llvm-sass overlay checkout (build in ./build) or sass/ of the
# llvm-project tree (build in ../build).
ROOT = Path(__file__).resolve().parents[1]
BUILD = next((d for d in (ROOT / "build", ROOT.parent / "build") if (d / "bin" / "llc").exists()),
             ROOT.parent / "build")
# A clang with the NVPTX target (CUDA device-side frontend); Apple's lacks it.
CLANG = os.environ.get("CLANG", "/opt/homebrew/opt/llvm/bin/clang")
LLC = Path(os.environ.get("LLC", BUILD / "bin" / "llc"))
sys.path.insert(0, str(ROOT / "sass-dec"))


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        sys.exit(f"{' '.join(map(str, cmd))}\n{proc.stderr}")


def kernels(sass: str) -> dict[str, str]:
    """Split llc output into `#fn` blocks keyed by kernel name."""
    out, name, lines = {}, None, []
    for line in sass.splitlines():
        if m := re.match(r"#fn (\S+)\(", line):
            name, lines = m.group(1), []
        if name:
            lines.append(line)
            if line == "}":
                out[name] = "\n".join(lines) + "\n"
                name = None
    return out


# Assembler arch -> (clang --cuda-gpu-arch, llc -mcpu). GB10's sm_121 runs
# the sm_120 ISA; the cubin is tagged sm_121.
# nvcc --use_fast_math is -ftz=true -prec-div=false -prec-sqrt=false plus the
# __expf-style intrinsics for libm calls: afn on every call and division,
# preserve-sign denormals, and the macro CUDA's math headers key on. Unlike
# clang's -ffast-math it does not reassociate.
FAST_MATH = ["-fapprox-func", "-fcuda-flush-denormals-to-zero", "-D__USE_FAST_MATH__"]

ARCHES = {"sm90": ("sm_90", "sm_90"), "sm120": ("sm_120", "sm_120"),
          "sm121": ("sm_120", "sm_121")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("."))
    ap.add_argument("--arch", default="sm121", choices=sorted(ARCHES),
                    help="sm90: H100/H200; sm120: RTX 50; sm121: GB10")
    ap.add_argument("--llc-args", default="", help="extra llc flags, space separated")
    ap.add_argument("--fast-math", action="store_true",
                    help="as nvcc --use_fast_math: approximate math functions, "
                         "division and square root (MUFU), denormals flushed (.FTZ)")
    opts = ap.parse_args()
    opts.out.mkdir(parents=True, exist_ok=True)
    stem = opts.src.stem
    gpu_arch, cpu = ARCHES[opts.arch]

    ll = opts.src
    if opts.src.suffix == ".cu":
        ll = opts.out / f"{stem}.ll"
        # Without a CUDA install clang cannot infer the PTX ISA version, which
        # gates builtins such as __nvvm_shfl_sync_*. bf16 arithmetic stays in
        # bf16 (no excess float precision), as nvcc rounds each operation.
        run([CLANG, "-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib",
             f"--cuda-gpu-arch={gpu_arch}", "-Xclang", "-target-feature", "-Xclang", "+ptx87",
             "-Xclang", "-fbfloat16-excess-precision=none",
             "-fcuda-short-ptr", "-O3", "-S", "-emit-llvm", *FAST_MATH * opts.fast_math,
             "-include", str(ROOT / "kernels" / "cuda_min.h"), str(opts.src), "-o", str(ll)])
    sass_path = opts.out / f"{stem}.sass"
    run([str(LLC), "-march=sass", f"-mcpu={cpu}", "-O3", *opts.llc_args.split(), str(ll),
         "-o", str(sass_path)])

    from assembler import assemble

    for name, src in kernels(sass_path.read_text()).items():
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cubin = assemble(src, kernel_name=name, arch=opts.arch, strict_deps=True)
        if "[depcheck]" in err.getvalue():
            sys.exit(f"{name}: assembler dependency check failed\n{err.getvalue()}")
        path = opts.out / f"{stem}.{name}.cubin"
        path.write_bytes(cubin)
        n = sum(1 for line in src.splitlines() if ";[" in line)
        print(f"{path}  ({n} instructions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
