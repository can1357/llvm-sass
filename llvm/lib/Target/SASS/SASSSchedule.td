//===-- SASSSchedule.td - Scheduling model -----------------*- tablegen -*-===//
//
// In-order, single-issue per warp scheduler. Fixed-latency classes use the
// ptxas sm_120 table values (int 6, IMAD 5, FMA 4); variable-latency classes
// (scoreboarded) use typical unloaded latencies so the machine scheduler
// hoists them early. The control-code pass, not this model, guarantees
// correctness.
//
//===----------------------------------------------------------------------===//

def WriteALU : SchedWrite;  // int pipe (IADD3, LOP3, SHF, ISETP, SEL, ...)
def WriteIMAD : SchedWrite; // IMAD on the FMA pipe
def WriteFMA : SchedWrite;  // FADD/FMUL/FFMA
def WriteWide : SchedWrite; // IMAD.WIDE, IADD.64
def WriteLDC : SchedWrite;  // constant bank loads
def WriteS2R : SchedWrite;  // special registers
def WriteSFU : SchedWrite;  // MUFU, F2I
def WriteLD : SchedWrite;   // global loads
def WriteLDS : SchedWrite;  // shared loads, shuffles
def WriteST : SchedWrite;   // stores
def WriteBR : SchedWrite;   // BRA/EXIT
def WritePred : SchedWrite; // ISETP/FSETP/PLOP3: up to 14 cycles to a guard

def SASSModel : SchedMachineModel {
  let IssueWidth = 1;
  let MicroOpBufferSize = 0;
  let LoadLatency = 400;
  let PostRAScheduler = 1;
  let CompleteModel = 0;
}

let SchedModel = SASSModel in {
def : WriteRes<WriteALU, []> { let Latency = 6; }
def : WriteRes<WriteIMAD, []> { let Latency = 5; }
def : WriteRes<WriteFMA, []> { let Latency = 4; }
def : WriteRes<WriteWide, []> { let Latency = 7; }
def : WriteRes<WriteLDC, []> { let Latency = 14; }
def : WriteRes<WriteS2R, []> { let Latency = 24; }
def : WriteRes<WriteSFU, []> { let Latency = 20; }
def : WriteRes<WriteLD, []> { let Latency = 400; }
def : WriteRes<WriteLDS, []> { let Latency = 30; }
def : WriteRes<WriteST, []> { let Latency = 1; }
def : WriteRes<WriteBR, []> { let Latency = 1; }
def : WriteRes<WritePred, []> { let Latency = 13; }
}
