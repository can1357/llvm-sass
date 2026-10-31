//===-- SASS.td - NVIDIA SASS target ------------------------*- tablegen -*-===//
//
// Emits NVIDIA SASS (machine code below PTX) in the text dialect of the
// sass-dec assembler, which encodes it into a loadable cubin. The ISA is
// sm_120 (consumer/DGX Blackwell); cubins are tagged sm_121 for GB10.
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

include "SASSRegisterInfo.td"
include "SASSSchedule.td"
include "SASSInstrInfo.td"

def SASSInstrInfo : InstrInfo;

defm : RemapAllTargetPseudoPointerOperands<GPR64>;

def : ProcessorModel<"sm_120", SASSModel, []>;
def : ProcessorModel<"sm_121", SASSModel, []>;

def SASS : Target {
  let InstructionSet = SASSInstrInfo;
}
