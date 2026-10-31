//===-- SASSBaseInfo.h - Shared SASS definitions ---------------*- C++ -*-===//
//
// TSFlags layout (see SASSInstrInfo.td) and condition-code spellings shared
// by the instruction printer and the codegen passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SASS_MCTARGETDESC_SASSBASEINFO_H
#define LLVM_LIB_TARGET_SASS_MCTARGETDESC_SASSBASEINFO_H

#include "llvm/MC/MCInstrDesc.h"
#include <cstdint>

namespace llvm::SASS {

/// Scoreboard that LDGDEPBAR commits async-copy groups to and DEPBAR_LE
/// waits on; never handed out for ordinary results.
constexpr unsigned AsyncCopySB = 5;
/// Latency mnemonic index into the generated SASSLatencyTables.inc arrays.
inline unsigned latMnem(const MCInstrDesc &D) { return D.TSFlags & 0x3f; }
/// The result is written back asynchronously and tracked by a write
/// scoreboard (LDG, LDC, S2R, MUFU, ...).
inline bool writesSB(const MCInstrDesc &D) { return (D.TSFlags >> 6) & 1; }
/// Register sources are read after issue and must be protected by a read
/// scoreboard until released (all decoupled MIO/UDP instructions).
inline bool readsLate(const MCInstrDesc &D) { return (D.TSFlags >> 7) & 1; }
/// Minimum stall cycles after issue (ISA MIN_WAIT_NEEDED; 5 for BRA/EXIT).
inline unsigned minStall(const MCInstrDesc &D) { return (D.TSFlags >> 8) & 0xf; }

/// ISETP condition: [2:0] LT LE GT GE EQ NE, bit 3 unsigned.
inline const char *intCondStr(unsigned CC) {
  static const char *const Strs[] = {"LT", "LE", "GT", "GE", "EQ", "NE"};
  return Strs[CC & 7];
}
inline bool intCondUnsigned(unsigned CC) { return CC & 8; }

/// FSETP condition: ordered, unordered (U suffix), NUM and NAN.
inline const char *fpCondStr(unsigned CC) {
  static const char *const Strs[] = {"LT",  "LE",  "GT",  "GE",  "EQ",
                                     "NE",  "LTU", "LEU", "GTU", "GEU",
                                     "EQU", "NEU", "NUM", "NAN"};
  return Strs[CC];
}

/// Constant bank 0 layout on sm_120: launch geometry, the default global
/// memory descriptor and the kernel parameter block.
enum : unsigned {
  CBankGlobalDesc = 0x358,
  CBankParamBase = 0x380,
};

} // namespace llvm::SASS

#endif
