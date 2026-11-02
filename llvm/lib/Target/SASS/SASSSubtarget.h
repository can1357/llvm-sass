//===-- SASSSubtarget.h - SASS subtarget ------------------------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSSUBTARGET_H
#define LLVM_LIB_TARGET_SASS_SASSSUBTARGET_H

#include "MCTargetDesc/SASSMCTargetDesc.h"
#include "SASSFrameLowering.h"
#include "SASSISelLowering.h"
#include "SASSInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include <memory>

#define GET_SUBTARGETINFO_HEADER
#include "SASSGenSubtargetInfo.inc"

namespace llvm {

class SASSSubtarget : public SASSGenSubtargetInfo {
  // Set by ParseSubtargetFeatures before the members below are built.
  bool HasBlackwellISA = false;
  SASSInstrInfo InstrInfo;
  SASSFrameLowering FrameLowering;
  SASSTargetLowering TLInfo;
  std::unique_ptr<const SelectionDAGTargetInfo> TSInfo;

public:
  SASSSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                const TargetMachine &TM);
  ~SASSSubtarget() override;

  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

  /// sm_100+ instructions (IADD.64, LDCU) and constant-bank ABI; false on
  /// Hopper (sm_90).
  bool hasBlackwellISA() const { return HasBlackwellISA; }
  /// Constant bank 0 offset of the kernel parameter block.
  unsigned paramBase() const { return HasBlackwellISA ? 0x380 : 0x210; }
  /// Constant bank 0 offset of the default global memory descriptor.
  unsigned globalDescOffset() const { return HasBlackwellISA ? 0x358 : 0x208; }
  /// Constant bank 0 offset of the initial per-thread stack pointer (a
  /// local-window address; the frame lies below it).
  unsigned stackPtrOffset() const { return HasBlackwellISA ? 0x37c : 0x28; }
  /// Constant bank 0 offset of the 64-bit generic address of the local
  /// window: generic = base + local address.
  unsigned localBaseOffset() const { return HasBlackwellISA ? 0x2f8 : 0x20; }

  const SASSInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const SASSFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const SASSTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const SASSRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override {
    return TSInfo.get();
  }

  // Instruction order comes from SASSPreRASched alone. LLVM's
  // GenericScheduler hoisted every LDS/LDG.128 of an unrolled tile to the
  // block top while under-reporting pressure (the allocator then ran out of
  // GPRs); the post-RA list scheduler, blind to the LSU queue, re-hoisted
  // shared loads into one burst (attention: 1.050x vs 1.073x of nvcc).
  bool enableMachineScheduler() const override { return false; }
  bool enablePostRAScheduler() const override { return false; }

  // Per-lane liveness of the wgmma accumulator tuples (GACC<n>): without it a
  // tuple built lane by lane around a GMMA interferes as a whole and the
  // allocator spills it (attention q_len 8: 1232 -> 704 stack bytes, 10x
  // faster). Hopper only for now: Blackwell output stays as validated.
  bool enableSubRegLiveness() const override { return !HasBlackwellISA; }
};

} // namespace llvm

#endif
