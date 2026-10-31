//===-- SASSInstrLocal.td - Local memory (stack) ---------*- tablegen -*-===//
//
// Per-thread local memory: allocas (SASSLowerAlloca puts them in addrspace 5)
// and register spills live in a frame below the stack pointer R1, set up by
// SASSFrameLowering. LDL/STL take a 32-bit local-window address plus a
// signed 24-bit offset, like LDS/STS. Frame indices select to [FI+off]
// operands that SASSRegisterInfo::eliminateFrameIndex rewrites to
// [R1+off]. Generic pointers to locals are base + local address, with the
// base read from constant bank 0 (SASSSubtarget::localBaseOffset).
//
//===----------------------------------------------------------------------===//

def ADDRl : ComplexPattern<i32, 2, "SelectADDRl", [frameindex], []>;

let mayLoad = 1, LatMnem = LAT_LD, SchedRW = [WriteLD] in {
def LDL32 : SASSVarInst<(outs Op32:$d), (ins mems:$addr), "LDL $d, $addr">;
def LDL64 : SASSVarInst<(outs G64:$d), (ins mems:$addr), "LDL.64 $d, $addr">;
def LDL128 : SASSVarInst<(outs G128:$d), (ins mems:$addr), "LDL.128 $d, $addr">;
def LDLU8 : SASSVarInst<(outs Op32:$d), (ins mems:$addr), "LDL.U8 $d, $addr">;
def LDLS8 : SASSVarInst<(outs Op32:$d), (ins mems:$addr), "LDL.S8 $d, $addr">;
def LDLU16 : SASSVarInst<(outs Op32:$d), (ins mems:$addr), "LDL.U16 $d, $addr">;
def LDLS16 : SASSVarInst<(outs Op32:$d), (ins mems:$addr), "LDL.S16 $d, $addr">;
}
let mayStore = 1, LatMnem = LAT_ST, SchedRW = [WriteST] in {
def STL32 : SASSVarInst<(outs), (ins mems:$addr, Op32:$v), "STL $addr, $v">;
def STL64 : SASSVarInst<(outs), (ins mems:$addr, G64:$v), "STL.64 $addr, $v">;
def STL128 : SASSVarInst<(outs), (ins mems:$addr, G128:$v), "STL.128 $addr, $v">;
def STLU8 : SASSVarInst<(outs), (ins mems:$addr, Op32:$v), "STL.U8 $addr, $v">;
def STLU16 : SASSVarInst<(outs), (ins mems:$addr, Op32:$v), "STL.U16 $addr, $v">;
}

defm : MemPats<5, "LDL", ADDRl>;
defm : StorePats<5, "STL", ADDRl>;

// Address of a frame object as a value: IADD Rd, R1, off after frame index
// elimination (a MOV when off is 0).
def to_tframeindex : SDNodeXForm<frameindex, [{
  return CurDAG->getTargetFrameIndex(N->getIndex(), N->getValueType(0));
}]>;
def : Pat<(i32 frameindex:$fi), (IADDri (i32 (to_tframeindex $fi)), 0)>;
def : Pat<(add frameindex:$fi, imm:$c), (IADDri (i32 (to_tframeindex $fi)), imm:$c)>;
def : Pat<(add frameindex:$fi, i32:$x), (IADD3rri $x, (i32 (to_tframeindex $fi)), 0)>;

// Frame indices typed as generic (i64) pointers come from stack temporaries
// made during legalization: their generic address is base + local address.
class GenericFrameAddr<int base>
    : Pat<(i64 frameindex:$fi),
          (IADD64rr (LDC64 base),
                    (REG_SEQUENCE GPR64, (IADDri (i32 (to_tframeindex $fi)), 0), sub0,
                                         (i32 RZ), sub1))>;
let Predicates = [NoBlackwellISA] in
def : GenericFrameAddr<0x20>;
let Predicates = [HasBlackwellISA] in
def : GenericFrameAddr<0x2f8>;

// Spills of predicates and uniform registers go through a GPR scavenged
// when eliminateFrameIndex expands these (P: SEL/ISETP around an 8-bit
// slot; URn: MOV/R2UR around 32-bit slots).
let isPseudo = 1, isCodeGenOnly = 1, isPredicable = 0 in {
let mayStore = 1 in {
def SPILL_P : SASSInst<(outs), (ins mems:$addr, PRED:$v), "">;
def SPILL_UR32 : SASSInst<(outs), (ins mems:$addr, UGPR32:$v), "">;
def SPILL_UR64 : SASSInst<(outs), (ins mems:$addr, UG64:$v), "">;
}
let mayLoad = 1 in {
def RELOAD_P : SASSInst<(outs PRED:$d), (ins mems:$addr), "">;
def RELOAD_UR32 : SASSInst<(outs UGPR32:$d), (ins mems:$addr), "">;
def RELOAD_UR64 : SASSInst<(outs UG64:$d), (ins mems:$addr), "">;
}
// wgmma accumulator tuples (4-aligned): one STL.128/LDL.128 per quad.
foreach S = !range(8, 65, 4) in {
let mayStore = 1 in
def SPILL_GACC#S : SASSInst<(outs), (ins mems:$addr, !cast<RegisterClass>("GACC"#S):$v), "">;
let mayLoad = 1 in
def RELOAD_GACC#S : SASSInst<(outs !cast<RegisterClass>("GACC"#S):$d), (ins mems:$addr), "">;
}
}
