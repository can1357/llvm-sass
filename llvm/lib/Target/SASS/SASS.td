//===-- SASS.td - NVIDIA SASS target ------------------------*- tablegen -*-===//
//
// Emits NVIDIA SASS (machine code below PTX) in the text dialect of the
// sass-dec assembler, which encodes it into a loadable cubin. Targets Hopper
// (sm_90: H100, H200) and consumer/DGX Blackwell (sm_120; cubins for GB10
// are tagged sm_121), selected with -mcpu.
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

include "SASSRegisterInfo.td"
include "SASSSchedule.td"
include "SASSInstrInfo.td"

def SASSInstrInfo : InstrInfo;

defm : RemapAllTargetPseudoPointerOperands<GPR64>;

def FeatureBlackwell
    : SubtargetFeature<"blackwell", "HasBlackwellISA", "true",
                       "Blackwell ISA and ABI: IADD.64, LDCU, launch geometry "
                       "and parameters at c[0x0][0x360]/c[0x0][0x380]">;

def : ProcessorModel<"sm_90", SASSModel, []>;
def : ProcessorModel<"sm_90a", SASSModel, []>;
def : ProcessorModel<"sm_120", SASSModel, [FeatureBlackwell]>;
def : ProcessorModel<"sm_121", SASSModel, [FeatureBlackwell]>;

def SASS : Target {
  let InstructionSet = SASSInstrInfo;
}
