//===-- SASSGMMAFlags.h - HGMMA/QGMMA flags operand -------------*- C++ -*-===//
//
// The flags immediate of the GMMA_* instructions (SASSInstrWGMMA.td): shape,
// types and modifiers, printed as the mnemonic by SASSInstPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SASS_MCTARGETDESC_SASSGMMAFLAGS_H
#define LLVM_LIB_TARGET_SASS_MCTARGETDESC_SASSGMMAFLAGS_H

#include <cstdint>

namespace llvm::SASS::GMMA {

/// Input element formats.
enum Fmt : unsigned { F16 = 0, BF16 = 1, TF32 = 2, E4M3 = 3, E5M2 = 4 };

// [5:0] N/8, [6] f32 accumulators, [9:7] A format, [12:10] B format,
// [13] negate A, [14] negate B, [15] transpose A, [16] transpose B.
constexpr unsigned F32Bit = 1u << 6, NegA = 1u << 13, NegB = 1u << 14,
                   TnspA = 1u << 15, TnspB = 1u << 16;

inline unsigned make(unsigned N, bool F32, Fmt A, Fmt B) {
  return (N / 8) | (F32 ? F32Bit : 0) | (A << 7) | (B << 10);
}
inline unsigned n(unsigned F) { return (F & 0x3f) * 8; }
inline bool f32(unsigned F) { return F & F32Bit; }
inline Fmt fmtA(unsigned F) { return Fmt((F >> 7) & 7); }
inline Fmt fmtB(unsigned F) { return Fmt((F >> 10) & 7); }
inline bool isQ(unsigned F) { return fmtA(F) >= E4M3; }
/// K of the m64nNkK shape.
inline unsigned k(unsigned F) {
  return isQ(F) ? 32 : fmtA(F) == TF32 ? 8 : 16;
}
/// Accumulator registers per thread.
inline unsigned accRegs(unsigned F) { return n(F) / (f32(F) ? 2 : 4); }

} // namespace llvm::SASS::GMMA

#endif
