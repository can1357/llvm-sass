"""Static per-loop statistics of a lifted/generated SASS listing (no GPU).

For each backward branch: instructions, summed stall cycles (single-warp
issue time lower bound), scoreboard waits, and the instruction mix.

    python tools/loopstats.py out/attn/attn.sass kernels/attn.nvcc.sass
"""
import re
import sys
from collections import Counter

INST = re.compile(r"^\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)[^;]*;\[(\d):(\d):\{([\d,]*)\}:(\d+):(\d)")


def loops(path):
    lines = open(path).read().splitlines()
    labels, insts = {}, []
    for line in lines:
        if m := re.search(r"#def_label\(([^)]+)\)", line):
            labels[m.group(1)] = len(insts)
            continue
        if m := INST.match(line):
            insts.append((m.group(1), int(m.group(5)), m.group(4), line))
    for i, (mn, *_rest, line) in enumerate(insts):
        if mn.startswith("BRA") and (t := re.search(r"#label\(([^)]+)\)", line)):
            start = labels.get(t.group(1))
            if start is not None and start <= i:
                body = insts[start:i + 1]
                yield t.group(1), body


for path in sys.argv[1:]:
    for label, body in loops(path):
        if len(body) < 20:
            continue
        stalls = sum(s for _, s, _, _ in body)
        waits = sum(1 for _, _, req, _ in body if req)
        mix = Counter(mn.split(".")[0] for mn, *_ in body)
        print(f"{path}: loop {label}: {len(body)} instrs, {stalls} stall cycles, {waits} SB waits")
        print("   ", ", ".join(f"{k} {v}" for k, v in mix.most_common(10)))
