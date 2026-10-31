//===-- SASSInstrInfo.h - SASS instruction information ----------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSINSTRINFO_H
#define LLVM_LIB_TARGET_SASS_SASSINSTRINFO_H

#include "SASSRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "SASSGenInstrInfo.inc"

namespace llvm {

class SASSSubtarget;

/// Branch conditions and predicates are the guard operand pair
/// [predicate register, negate flag], so IfConverter can apply a branch
/// condition directly as an instruction guard.
class SASSInstrInfo : public SASSGenInstrInfo {
  const SASSRegisterInfo RI;

public:
  explicit SASSInstrInfo(const SASSSubtarget &ST);

  const SASSRegisterInfo &getRegisterInfo() const { return RI; }

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   const DebugLoc &DL, Register DestReg, Register SrcReg,
                   bool KillSrc, bool RenamableDest = false,
                   bool RenamableSrc = false) const override;

  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify) const override;
  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;
  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override;
  bool
  reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;

  bool isPredicated(const MachineInstr &MI) const override;
  bool isPredicable(const MachineInstr &MI) const override;
  bool PredicateInstruction(MachineInstr &MI,
                            ArrayRef<MachineOperand> Pred) const override;
  bool SubsumesPredicate(ArrayRef<MachineOperand> Pred1,
                         ArrayRef<MachineOperand> Pred2) const override;
  bool ClobbersPredicate(MachineInstr &MI, std::vector<MachineOperand> &Pred,
                         bool SkipDead) const override;
  bool isProfitableToIfCvt(MachineBasicBlock &MBB, unsigned NumCycles,
                           unsigned ExtraPredCycles,
                           BranchProbability Probability) const override;
  bool isProfitableToIfCvt(MachineBasicBlock &TMBB, unsigned NumT,
                           unsigned ExtraT, MachineBasicBlock &FMBB,
                           unsigned NumF, unsigned ExtraF,
                           BranchProbability Probability) const override;
  bool isProfitableToDupForIfCvt(MachineBasicBlock &MBB, unsigned NumCycles,
                                 BranchProbability Probability) const override;

  unsigned getInstSizeInBytes(const MachineInstr &MI) const override {
    return 16;
  }
  bool isHighLatencyDef(int Opc) const override;
};

} // namespace llvm

#endif
