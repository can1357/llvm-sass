//===-- SASSInstrUniform.td - Uniform datapath ALU ---------*- tablegen -*-===//
//
// Hopper's uniform datapath: one result per warp in UR0-UR62. Never selected
// from the DAG; SASSUniformRegs rewrites warp-uniform GPR computations into
// these, operand for operand (the GPR twin is named in each comment), and
// copies the results back to GPRs with MOVur where a per-thread instruction
// reads them.
//
// Uniform instructions take a uniform predicate guard, which the PRED-typed
// guard operand cannot express: never predicated.
//
//===----------------------------------------------------------------------===//

let isPredicable = 0 in {

let LatMnem = LAT_ULDC, isReMaterializable = 1 in {
def ULDC32 : SASSInst<(outs UGPR32:$d), (ins simm32:$o), "ULDC $d, c[0x0][$o]">;  // LDC32
def ULDCB32 : SASSInst<(outs UGPR32:$d), (ins simm32:$bank, simm32:$o),          // LDCB32_I
                       "ULDC $d, c[$bank][$o]">;
def ULDCB64 : SASSInst<(outs UG64:$d), (ins simm32:$bank, simm32:$o),            // LDCB64_I
                       "ULDC.64 $d, c[$bank][$o]">;
}

let LatMnem = LAT_UMOV, isReMaterializable = 1, isAsCheapAsAMove = 1,
    isMoveImm = 1 in
def UMOVi : SASSInst<(outs UGPR32:$d), (ins simm32:$v), "UMOV $d, $v">;         // MOVri

// Variable latency (scoreboarded), as S2R.
let LatMnem = LAT_S2UR, SchedRW = [WriteS2R] in {
def S2UR_CTAIDX : SASSVarInst<(outs UGPR32:$d), (ins), "S2UR $d, SR_CTAID.X">;
def S2UR_CTAIDY : SASSVarInst<(outs UGPR32:$d), (ins), "S2UR $d, SR_CTAID.Y">;
def S2UR_CTAIDZ : SASSVarInst<(outs UGPR32:$d), (ins), "S2UR $d, SR_CTAID.Z">;
def S2UR_CGA : SASSVarInst<(outs UGPR32:$d), (ins), "S2UR $d, SR_CgaCtaId">;
}

// UIADD3, UIMAD, ULOP3, USHF and ULEA share the uniform ALU's latencies.
let LatMnem = LAT_UIADD3 in {
def UIADDrr : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, UGPR32:$b),            // IADDrr
                       "UIADD3 $d, UPT, UPT, $a, $b, URZ">;
def UIADD3rrr : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, UGPR32:$b, UGPR32:$c),
                         "UIADD3 $d, UPT, UPT, $a, $b, $c">;
def UIADD3rri : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, UGPR32:$b, simm32:$c),
                         "UIADD3 $d, UPT, UPT, $a, $c, $b">;
def UISUBrr : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, UGPR32:$b),
                       "UIADD3 $d, UPT, UPT, $a, -$b, URZ">;

def UIMADrrr : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, UGPR32:$b, UGPR32:$c),
                        "UIMAD $d, $a, $b, $c">;
def UIMADrir : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, simm32:$b, UGPR32:$c),
                        "UIMAD $d, $a, $b, $c">;
def UIMADWrrr : SASSInst<(outs UG64:$d), (ins UGPR32:$a, UGPR32:$b, UG64:$c),
                         "UIMAD.WIDE $d, $a, $b, $c">;
def UIMADWrir : SASSInst<(outs UG64:$d), (ins UGPR32:$a, simm32:$b, UG64:$c),
                         "UIMAD.WIDE $d, $a, $b, $c">;
def UIMADWrr0 : SASSInst<(outs UG64:$d), (ins UGPR32:$a, UGPR32:$b),
                         "UIMAD.WIDE $d, $a, $b, URZ">;
def UIMADWUrrr : SASSInst<(outs UG64:$d), (ins UGPR32:$a, UGPR32:$b, UG64:$c),
                          "UIMAD.WIDE.U32 $d, $a, $b, $c">;
def UIMADWUrir : SASSInst<(outs UG64:$d), (ins UGPR32:$a, simm32:$b, UG64:$c),
                          "UIMAD.WIDE.U32 $d, $a, $b, $c">;
def UIMADWUrr0 : SASSInst<(outs UG64:$d), (ins UGPR32:$a, UGPR32:$b),
                          "UIMAD.WIDE.U32 $d, $a, $b, URZ">;

def ULOP3rrr : SASSInst<(outs UGPR32:$d),
                        (ins UGPR32:$a, UGPR32:$b, UGPR32:$c, simm32:$lut),
                        "ULOP3.LUT $d, $a, $b, $c, $lut, !UPT">;
def ULOP3rir : SASSInst<(outs UGPR32:$d),
                        (ins UGPR32:$a, simm32:$b, UGPR32:$c, simm32:$lut),
                        "ULOP3.LUT $d, $a, $b, $c, $lut, !UPT">;

def USHLrr : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, UGPR32:$b), "USHF.L.U32 $d, $a, $b, URZ">;
def USHLri : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, simm32:$b), "USHF.L.U32 $d, $a, $b, URZ">;
def USRLrr : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, UGPR32:$b), "USHF.R.U32.HI $d, URZ, $b, $a">;
def USRLri : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, simm32:$b), "USHF.R.U32.HI $d, URZ, $b, $a">;
def USRArr : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, UGPR32:$b), "USHF.R.S32.HI $d, URZ, $b, $a">;
def USRAri : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, simm32:$b), "USHF.R.S32.HI $d, URZ, $b, $a">;
def USHLU64HI : SASSInst<(outs UGPR32:$d), (ins UGPR32:$lo, simm32:$s, UGPR32:$hi),
                         "USHF.L.U64.HI $d, $lo, $s, $hi">;
def USHRU64 : SASSInst<(outs UGPR32:$d), (ins UGPR32:$lo, simm32:$s, UGPR32:$hi),
                       "USHF.R.U64 $d, $lo, $s, $hi">;
def USHRS64 : SASSInst<(outs UGPR32:$d), (ins UGPR32:$lo, simm32:$s, UGPR32:$hi),
                       "USHF.R.S64 $d, $lo, $s, $hi">;

def ULEArrr : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, UGPR32:$b, simm32:$s),
                       "ULEA $d, $a, $b, $s">;
}

} // isPredicable = 0

// A uniform value read by per-thread instructions. Unlike MOVru, the register
// allocator recomputes it next to its users instead of spilling it.
let LatMnem = LAT_MOV, isReMaterializable = 1, isAsCheapAsAMove = 1 in
def MOVur : SASSInst<(outs GPR32:$d), (ins UGPR32:$a), "MOV $d, $a">;

// Per-thread instructions reading a uniform register as their second source
// (the "b" slot accepts a UR on the vector datapath). Named after the GPR
// twin with `u` for the UR operand; same operand order, so SASSUniformRegs
// swaps the opcode and the register.
let LatMnem = LAT_IADD3 in {
def IADDru : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b),           // IADDrr
                      "IADD3 $d, PT, PT, $a, $b, RZ">;
def IADD3rur : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b, GPR32:$c),
                        "IADD3 $d, PT, PT, $a, $b, $c">;
def ISUBru : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b),           // a - b
                      "IADD3 $d, PT, PT, $a, -$b, RZ">;
def ISUBur : SASSInst<(outs GPR32:$d), (ins UGPR32:$a, GPR32:$b),           // a - b
                      "IADD3 $d, PT, PT, -$b, $a, RZ">;
}
let LatMnem = LAT_IADD3, isPredicable = 0 in {
def IADD3COru : SASSInst<(outs GPR32:$d, PRED:$co), (ins GPR32:$a, UGPR32:$b),
                         "IADD3 $d, $co, PT, $a, $b, RZ">;
def IADD3Xru : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b, PRED:$ci),
                        "IADD3.X $d, PT, PT, $a, $b, RZ, $ci, !PT">;
}
let LatMnem = LAT_IMAD, SchedRW = [WriteIMAD] in
def IMADrur : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b, GPR32:$c),
                       "IMAD $d, $a, $b, $c">;
let LatMnem = LAT_IMAD, SchedRW = [WriteWide] in {
def IMADWrur : SASSInst<(outs G64:$d), (ins GPR32:$a, UGPR32:$b, G64:$c),
                        "IMAD.WIDE $d, $a, $b, $c">;
def IMADWru0 : SASSInst<(outs G64:$d), (ins GPR32:$a, UGPR32:$b),
                        "IMAD.WIDE $d, $a, $b, RZ">;
def IMADWUrur : SASSInst<(outs G64:$d), (ins GPR32:$a, UGPR32:$b, G64:$c),
                         "IMAD.WIDE.U32 $d, $a, $b, $c">;
def IMADWUru0 : SASSInst<(outs G64:$d), (ins GPR32:$a, UGPR32:$b),
                         "IMAD.WIDE.U32 $d, $a, $b, RZ">;
}
let LatMnem = LAT_LOP3 in
def LOP3rur : SASSInst<(outs GPR32:$d),
                       (ins GPR32:$a, UGPR32:$b, GPR32:$c, simm32:$lut),
                       "LOP3.LUT $d, $a, $b, $c, $lut, !PT">;
let LatMnem = LAT_SHF in {
def SHLru : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b), "SHF.L.U32 $d, $a, $b, RZ">;
def SRLru : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b), "SHF.R.U32.HI $d, RZ, $b, $a">;
def SRAru : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b), "SHF.R.S32.HI $d, RZ, $b, $a">;
def SHLU64HIru : SASSInst<(outs GPR32:$d), (ins GPR32:$lo, UGPR32:$s, GPR32:$hi),  // SHLU64HIrr
                          "SHF.L.U64.HI $d, $lo, $s, $hi">;
}
let LatMnem = LAT_LEA in
def LEAru : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b, simm32:$s),
                     "LEA $d, $a, $b, $s">;
// 64-bit (a << s) + UR pair, as ptxas forms pointer + index with a uniform
// pointer: low word with carry-out, high word from the bits shifted out of a
// (zero- or sign-extended) plus the carry. SASSUniformRegs rewrites
// IMAD.WIDE[.U32] a, 2^s, {UR pair} with these.
let LatMnem = LAT_LEA, isPredicable = 0 in {
def LEACOru : SASSInst<(outs GPR32:$d, PRED:$co), (ins GPR32:$a, UGPR32:$b, simm32:$s),
                       "LEA $d, $co, $a, $b, $s">;
def LEAHIXru : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b, simm32:$s, PRED:$ci),
                        "LEA.HI.X $d, $a, $b, RZ, $s, $ci">;
def LEAHIXSX32ru : SASSInst<(outs GPR32:$d),
                            (ins GPR32:$a, UGPR32:$b, simm32:$s, PRED:$ci),
                            "LEA.HI.X.SX32 $d, $a, $b, $s, $ci">;
}
let LatMnem = LAT_ISETP, SchedRW = [WritePred] in {
def ISETPru : SASSInst<(outs PRED:$p),
                       (ins GPR32:$a, UGPR32:$b, icc:$cc, PRED:$c, negflag:$cn),
                       "ISETP.${cc}.AND $p, PT, $a, $b, ${cn}$c">;
def ISETPORru : SASSInst<(outs PRED:$p),                                       // ISETPORrr
                         (ins GPR32:$a, UGPR32:$b, icc:$cc, PRED:$c, negflag:$cn),
                         "ISETP.${cc}.OR $p, PT, $a, $b, ${cn}$c">;
}
let LatMnem = LAT_SEL in
def SELru : SASSInst<(outs GPR32:$d), (ins GPR32:$a, UGPR32:$b, PRED:$p, negflag:$pn),
                     "SEL $d, $a, $b, ${pn}$p">;
