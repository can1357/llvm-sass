//===-- SASSTargetInfo.h - SASS target registry -----------------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_TARGETINFO_SASSTARGETINFO_H
#define LLVM_LIB_TARGET_SASS_TARGETINFO_SASSTARGETINFO_H

namespace llvm {
class Target;
/// The `sass` target, selected by name (`llc -march=sass`); it consumes
/// nvptx64 IR as produced by `clang -x cuda --cuda-device-only`.
Target &getTheSASSTarget();
} // namespace llvm

#endif
