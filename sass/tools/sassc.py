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


# Occupancy retry (sm_90). Under a launch-bounds count the allocator uses
# registers freely up to it and rematerializes nothing (values hoisted to
# the kernel entry each keep a register), so a kernel can land just above an
# occupancy step that it fits under: attention decode1 took 196 registers
# (2 CTAs/SM) where 168 (3 CTAs/SM) compiles without spills. Each such kernel
# is compiled again capped at the next step ("nvvm.maxnreg", which
# SASS::maxRegCount honors) and the capped code is kept only if it spills
# no more than the original; failing that, again with the pre-RA scheduler
# keeping more registers free under the cap (its schedule, not the
# allocator, is then what overflows). Kernels with setmaxnreg are left
# alone: their regions' counts are split from the launch count. A kernel
# whose launch bounds ask for a minimum of CTAs per SM ("nvvm.minctasm") is
# retried only while it fits no more than that minimum: the step above it
# (decode1: 2 -> 3 CTAs/SM) pays, a further one under the cap costs the
# schedule more than the occupancy brings (sliding decode16, 3 -> 4 CTAs/SM
# at 138 -> 128 registers: the main loop +640 cycles, graph us +2.6%).
RETRY_SCHED_SLACK = (None, 16, 24)

def regs_for_blocks(threads: int, blocks: int) -> int:
    """Largest count with which `blocks` CTAs of `threads` fit on an SM
    (SASS::maxRegCount: 16K registers per sub-partition, 8-register units,
    a CTA's warps spread over the 4 sub-partitions)."""
    warps = blocks * -(-threads // 32)
    return min(16384 // (-(-warps // 4) * 32) // 8 * 8, 255)


def next_step(threads: int, regs: int) -> int | None:
    """Register count of the next occupancy step below `regs`, if any."""
    warps = -(-threads // 32)
    blocks = 1
    while (blocks + 1) * warps <= 64 and regs_for_blocks(threads, blocks + 1) >= regs:
        blocks += 1
    if (blocks + 1) * warps > 64 or regs_for_blocks(threads, blocks) < regs:
        return None
    return regs_for_blocks(threads, blocks + 1)


def blocks_fitting(threads: int, regs: int) -> int:
    """CTAs of `threads` that fit on an SM with `regs` registers each."""
    warps = -(-threads // 32)
    blocks = 0
    while (blocks + 1) * warps <= 64 and regs_for_blocks(threads, blocks + 1) >= regs:
        blocks += 1
    return blocks


def kernel_attrs(ll: str, name: str) -> tuple[str, str] | None:
    """(define line, its attributes text) of kernel `name` in IR text."""
    m = re.search(rf"^define [^\n]*@{re.escape(name)}\([^\n]*$", ll, re.M)
    if not m:
        return None
    line = m.group(0)
    attrs = [line]
    for g in re.findall(r" #(\d+)\b", line):
        if a := re.search(rf"^attributes #{g} = \{{([^\n]*)\}}$", ll, re.M):
            attrs.append(a.group(1))
    return line, " ".join(attrs)


def pragma(block: str, what: str) -> int:
    m = re.search(rf"#pragma {what}\((\d+)\)", block)
    return int(m.group(1)) if m else 0


def spills(block: str) -> int:
    return len(re.findall(r"^\s+(?:@!?U?P\w+ )?(?:STL|LDL)\b", block, re.M))


def occupancy_retry(ll_path: Path, sass_path: Path, llc_cmd: list[str]) -> None:
    ll = ll_path.read_text()
    sass = sass_path.read_text()
    blocks = kernels(sass)
    caps = {}
    for name, block in blocks.items():
        found = kernel_attrs(ll, name)
        if not found or '"nvvm.maxnreg"' in found[1]:
            continue
        if not (m := re.search(r'"nvvm.maxntid"="([\d,]+)"', found[1])):
            continue
        threads = 1
        for d in m.group(1).split(","):
            threads *= int(d)
        regs = pragma(block, "MAXREG_COUNT")
        if (mc := re.search(r'"nvvm.minctasm"="(\d+)"', found[1])) and \
                blocks_fitting(threads, regs) > int(mc.group(1)):
            continue
        if ("USETMAXREG" not in block and (cap := next_step(threads, regs))
                and re.search(r" #\d+\b", found[0])):
            caps[name] = (cap, found[0])
    if not caps:
        return
    for cap, line in caps.values():
        group = re.findall(r" #\d+\b", line)[-1]
        ll = ll.replace(line, line.replace(group, f'{group} "nvvm.maxnreg"="{cap}"', 1), 1)
    retry_ll = ll_path.with_suffix(".maxnreg.ll")
    retry_sass = sass_path.with_suffix(".maxnreg.sass")
    retry_ll.write_text(ll)
    for slack in RETRY_SCHED_SLACK:
        extra = [f"-sass-sched-reg-slack={slack}"] if slack else []
        run([*llc_cmd, *extra, str(retry_ll), "-o", str(retry_sass)])
        capped = kernels(retry_sass.read_text())
        for name, (cap, _) in list(caps.items()):
            old, new = blocks[name], capped.get(name, "")
            if (new and pragma(new, "MAXREG_COUNT") <= cap
                    and pragma(new, "STACK_SIZE") <= pragma(old, "STACK_SIZE")
                    and spills(new) <= spills(old)):
                print(f"{name}: {pragma(old, 'MAXREG_COUNT')} -> {pragma(new, 'MAXREG_COUNT')} "
                      f"registers (capped at {cap}{f', scheduler slack {slack}' if slack else ''})")
                sass = sass.replace(old, new, 1)
                del caps[name]
        if not caps:
            break
    for name, (cap, _) in caps.items():
        print(f"{name}: {pragma(blocks[name], 'MAXREG_COUNT')} registers (spills when capped at {cap})")
    sass_path.write_text(sass)
    retry_ll.unlink()
    retry_sass.unlink()


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
    llc_cmd = [str(LLC), "-march=sass", f"-mcpu={cpu}", "-O3", *opts.llc_args.split()]
    run([*llc_cmd, str(ll), "-o", str(sass_path)])
    if opts.arch == "sm90" and ll.suffix == ".ll":
        occupancy_retry(ll, sass_path, llc_cmd)

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
