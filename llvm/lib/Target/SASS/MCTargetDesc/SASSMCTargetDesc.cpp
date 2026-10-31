//===-- SASSMCTargetDesc.cpp - SASS MC layer registration -----------------===//

#include "SASSMCTargetDesc.h"
#include "SASSInstPrinter.h"
#include "SASSMCAsmInfo.h"
#include "TargetInfo/SASSTargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "SASSGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "SASSGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "SASSGenRegisterInfo.inc"

namespace {
/// The dialect has no sections or directives: section switches print nothing.
class SASSTargetStreamer : public MCTargetStreamer {
public:
  using MCTargetStreamer::MCTargetStreamer;
  void changeSection(const MCSection *, MCSection *, uint32_t,
                     raw_ostream &) override {}
};
} // namespace

static MCInstrInfo *createSASSMCInstrInfo() {
  auto *X = new MCInstrInfo();
  InitSASSMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createSASSMCRegisterInfo(const Triple &TT) {
  auto *X = new MCRegisterInfo();
  InitSASSMCRegisterInfo(X, SASS::R1);
  return X;
}

static MCSubtargetInfo *createSASSMCSubtargetInfo(const Triple &TT,
                                                  StringRef CPU, StringRef FS) {
  return createSASSMCSubtargetInfoImpl(TT, CPU.empty() ? "sm_121" : CPU,
                                       /*TuneCPU=*/CPU, FS);
}

static MCAsmInfo *createSASSMCAsmInfo(const MCRegisterInfo &MRI,
                                      const Triple &TT,
                                      const MCTargetOptions &Options) {
  return new SASSMCAsmInfo(TT, Options);
}

static MCInstPrinter *createSASSMCInstPrinter(const Triple &T,
                                              unsigned SyntaxVariant,
                                              const MCAsmInfo &MAI,
                                              const MCInstrInfo &MII,
                                              const MCRegisterInfo &MRI) {
  return new SASSInstPrinter(MAI, MII, MRI);
}

static MCTargetStreamer *createSASSAsmTargetStreamer(MCStreamer &S,
                                                     formatted_raw_ostream &,
                                                     MCInstPrinter *) {
  return new SASSTargetStreamer(S);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSASSTargetMC() {
  Target &T = getTheSASSTarget();
  RegisterMCAsmInfoFn X(T, createSASSMCAsmInfo);
  TargetRegistry::RegisterMCInstrInfo(T, createSASSMCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(T, createSASSMCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(T, createSASSMCSubtargetInfo);
  TargetRegistry::RegisterMCInstPrinter(T, createSASSMCInstPrinter);
  TargetRegistry::RegisterAsmTargetStreamer(T, createSASSAsmTargetStreamer);
}
