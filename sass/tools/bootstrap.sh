#!/bin/sh
# One-time setup of a fresh clone: fetch the pinned sass-dec assembler, apply
# our fixes, generate its sm_120 instruction database and configure an llc
# build with the SASS target.
# Then: ninja -C build llc
set -e
cd "$(dirname "$0")/../.."

git submodule update --init sass/sass-dec
if git -C sass/sass-dec apply --check ../patches/sass-dec.patch 2>/dev/null; then
  git -C sass/sass-dec apply ../patches/sass-dec.patch
fi
(cd sass/sass-dec && python3 tools/parse_sm120.py)
cmake -G Ninja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD=NVPTX -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SASS
