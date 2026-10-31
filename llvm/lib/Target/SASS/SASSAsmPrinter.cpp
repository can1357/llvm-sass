//===-- SASSAsmPrinter.cpp - SASS text in the sass-dec dialect -------------===//
//
// Each kernel becomes one block the sass-dec assembler turns into a cubin:
//
//   #fn saxpy(p0<4>, p1<4>, p2<8>, p3<8>) {
//       #pragma MAXREG_COUNT(10)
//       S2R R0, SR_TID.X;[1:7:{}:1:0]
//   #def_label(LBB0_2)
//       ...
//   }
//
// Every instruction ends in its control bracket ;[wr:rd:{req}:stall:yield]
// from SASSControlCodes.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SASSBaseInfo.h"
#include "MCTargetDesc/SASSInstPrinter.h"
#include "SASS.h"
#include "SASSMachineFunctionInfo.h"
#include "SASSSubtarget.h"
#include "TargetInfo/SASSTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {
class SASSAsmPrinter : public AsmPrinter {
  std::unique_ptr<SASSInstPrinter> Printer;

public:
  static char ID;
  SASSAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID) {}

  StringRef getPassName() const override { return "SASS Assembly Printer"; }

  void emitStartOfAsmFile(Module &) override {}
  void emitEndOfAsmFile(Module &) override {}
  void emitFunctionEntryLabel() override;
  void emitFunctionBodyEnd() override;
  void emitBasicBlockStart(const MachineBasicBlock &MBB) override;
  void emitBasicBlockEnd(const MachineBasicBlock &) override {}
  void emitInstruction(const MachineInstr *MI) override;
  void emitGlobalVariable(const GlobalVariable *GV) override;

private:
  MCInst lower(const MachineInstr &MI);
  unsigned regCount() const;
};
} // namespace

char SASSAsmPrinter::ID = 0;

INITIALIZE_PASS(SASSAsmPrinter, "sass-asm-printer", "SASS Assembly Printer",
                false, false)

/// REGCOUNT as ptxas reports it: highest GPR + 1, plus the two registers the
/// hardware reserves at the top of the allocation.
unsigned SASSAsmPrinter::regCount() const {
  // setmaxnreg: the CTA pool starts from the __launch_bounds__ allocation.
  if (unsigned Entry = SASS::setMaxRegEntryCount(*MF))
    return Entry;
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  int Max = 1; // R1 is the reserved stack pointer.
  for (const MachineBasicBlock &B : *MF)
    for (const MachineInstr &MI : B)
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI->regunits(MO.getReg()))
          for (MCRegUnitRootIterator R(U, TRI); R.isValid(); ++R)
            if (SASS::GPR32RegClass.contains(*R) && *R != SASS::RZ)
              Max = std::max<int>(Max, TRI->getEncodingValue(*R));
      }
  return Max + 3;
}

void SASSAsmPrinter::emitFunctionEntryLabel() {
  auto *FI = MF->getInfo<SASSFunctionInfo>();
  std::string S;
  raw_string_ostream OS(S);
  OS << "#fn " << CurrentFnSym->getName() << '(';
  for (auto [I, Size] : enumerate(FI->ParamSizes)) {
    OS << (I ? ", " : "") << 'p' << I << '<' << Size;
    if (unsigned A = FI->ParamAligns[I])
      OS << ':' << A;
    OS << '>';
  }
  OS << ") {\n    #pragma MAXREG_COUNT(" << regCount() << ')';
  // Per-thread local memory below R1 (SASSFrameLowering).
  if (uint64_t Stack = MF->getFrameInfo().getStackSize())
    OS << "\n    #pragma STACK_SIZE(" << Stack << ')';
  const Module &M = *MF->getFunction().getParent();
  uint64_t Shared = SASS::sharedBytes(M);
  if (Shared || SASS::hasDynamicShared(M))
    OS << "\n    #pragma SHARED(" << Shared << ')';
  SASS::printConstantObjects(MF->getFunction(), OS);
  OutStreamer->emitRawText(OS.str());
  Printer.reset(static_cast<SASSInstPrinter *>(TM.getTarget().createMCInstPrinter(
      TM.getTargetTriple(), 0, TM.getMCAsmInfo(), *TM.getMCInstrInfo(),
      TM.getMCRegisterInfo())));
}

// Shared variables are allocated through the `#pragma SHARED` window and
// addressed by SASS::sharedOffset; __constant__ data is declared in each
// kernel block that uses it (SASS::printConstantObjects).
void SASSAsmPrinter::emitGlobalVariable(const GlobalVariable *GV) {
  unsigned AS = GV->getAddressSpace();
  if (AS == 3 || AS == 4 || GV->getName().starts_with("llvm."))
    return;
  report_fatal_error(Twine("SASS: ") +
                     (AS == 1 ? "__device__ variable '" : "global variable '") +
                     GV->getName() +
                     "' is not supported (only __shared__ and __constant__)");
}

void SASSAsmPrinter::emitFunctionBodyEnd() {
  // ptxas's guard against running off the end of the code.
  std::string End = SASSInstPrinter::labelName(CurrentFnSym->getName()) + "_end";
  OutStreamer->emitRawText("    #def_label(" + End + ")\n    BRA #label(" + End +
                           ");[7:7:{}:0:1]\n}");
}

void SASSAsmPrinter::emitBasicBlockStart(const MachineBasicBlock &MBB) {
  if (&MBB != &MF->front())
    OutStreamer->emitRawText("    #def_label(" +
                             SASSInstPrinter::labelName(MBB.getSymbol()->getName()) +
                             ")");
}

MCInst SASSAsmPrinter::lower(const MachineInstr &MI) {
  MCInst Inst;
  Inst.setOpcode(MI.getOpcode());
  for (const MachineOperand &MO : MI.operands()) {
    switch (MO.getType()) {
    case MachineOperand::MO_Register:
      if (MO.isImplicit())
        continue;
      Inst.addOperand(MCOperand::createReg(MO.getReg()));
      break;
    case MachineOperand::MO_Immediate:
      Inst.addOperand(MCOperand::createImm(MO.getImm()));
      break;
    case MachineOperand::MO_FPImmediate:
      Inst.addOperand(MCOperand::createImm(
          MO.getFPImm()->getValueAPF().bitcastToAPInt().getZExtValue()));
      break;
    case MachineOperand::MO_MachineBasicBlock:
      Inst.addOperand(MCOperand::createExpr(
          MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), OutContext)));
      break;
    case MachineOperand::MO_RegisterMask:
      break;
    default:
      report_fatal_error("SASS: unsupported machine operand");
    }
  }
  return Inst;
}

void SASSAsmPrinter::emitInstruction(const MachineInstr *MI) {
  SASS_MC::verifyInstructionPredicates(MI->getOpcode(),
                                       getSubtargetInfo().getFeatureBits());
  auto *FI = MF->getInfo<SASSFunctionInfo>();
  auto It = FI->Ctrl.find(MI);
  if (It == FI->Ctrl.end())
    report_fatal_error("SASS: instruction without control code");
  const SASSCtrl &C = It->second;

  std::string S;
  raw_string_ostream OS(S);
  OS << "    ";
  MCInst Inst = lower(*MI);
  if (SASS::takesFTZ(MI->getDesc()) &&
      MF->getDenormalMode(APFloat::IEEEsingle()).Output ==
          DenormalMode::PreserveSign)
    Inst.setFlags(SASS::InstFlagFTZ);
  Printer->printInst(&Inst, 0, "", getSubtargetInfo(), OS);
  OS << ";[" << C.Wr << ':' << C.Rd << ":{";
  bool First = true;
  for (unsigned SB = 0; SB < 6; ++SB)
    if (C.Req & (1u << SB)) {
      OS << (First ? "" : ",") << SB;
      First = false;
    }
  OS << "}:" << C.Stall << ':' << (C.Yield ? 1 : 0);
  if (C.Reuse)
    OS << ':' << C.Reuse;
  OS << ']';
  OutStreamer->emitRawText(OS.str());
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSASSAsmPrinter() {
  RegisterAsmPrinter<SASSAsmPrinter> X(getTheSASSTarget());
}
