//===-- SASSInstrInfo.td - SASS instructions ---------------*- tablegen -*-===//
//
// AsmStrings are in the sass-dec assembler dialect: 64/128-bit register
// operands print as explicit groups ({R2,R3}), global addresses as
// desc[{UR4,UR5}][{Ra,Ra+1}+off]. The guard predicate (@P0) and the control
// bracket ;[wr:rd:{req}:stall:yield] are printed by SASSAsmPrinter.
//
//===----------------------------------------------------------------------===//

include "SASSLatency.td"

//===----------------------------------------------------------------------===//
// Operands
//===----------------------------------------------------------------------===//

def simm32 : Operand<i32> { let PrintMethod = "printImm"; }
def fimm32 : Operand<f32> { let PrintMethod = "printF32Imm"; }
def negflag : Operand<i32> { let PrintMethod = "printNegFlag"; }
def icc : Operand<i32> { let PrintMethod = "printICC"; }
def fcc : Operand<i32> { let PrintMethod = "printFCC"; }
def brtarget : Operand<OtherVT> { let PrintMethod = "printBrTarget"; }

// Global address: 64-bit base register pair + signed 24-bit byte offset.
def memg : Operand<i64> {
  let MIOperandInfo = (ops GPR64, i32imm);
  let PrintMethod = "printGlobalAddr";
}
def ADDRri : ComplexPattern<i64, 2, "SelectADDRri", [], []>;

def G64 : RegisterOperand<GPR64, "printGroup">;
def G128 : RegisterOperand<GPR128, "printGroup">;
def UG64 : RegisterOperand<UGPR64, "printGroup">;

// Guard predicate @[!]Pg: PT (always) unless the instruction is predicated.
def pred : PredicateOperand<OtherVT, (ops PRED, i32imm), (ops (i1 PT), (i32 0))>;

//===----------------------------------------------------------------------===//
// Target nodes
//===----------------------------------------------------------------------===//

def SASSret : SDNode<"SASSISD::RET", SDTNone, [SDNPHasChain]>;

// Kernel parameter read from constant bank 0 at a byte offset.
def SASSldc : SDNode<"SASSISD::LDC", SDTypeProfile<1, 1, [SDTCisVT<1, i32>]>>;

// High-word compare of a 64-bit comparison, chained on the low-word result:
// (ahi, bhi, cc, plo) -> i1.
def SASSsetcc_ex : SDNode<"SASSISD::SETCC_EX",
  SDTypeProfile<1, 4, [SDTCisVT<0, i1>, SDTCisVT<1, i32>, SDTCisVT<2, i32>,
                       SDTCisVT<3, i32>, SDTCisVT<4, i1>]>>;

// 32-bit halves of a 64-bit register pair, and the pair of two halves.
def SASSlo : SDNode<"SASSISD::LO", SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i64>]>>;
def SASShi : SDNode<"SASSISD::HI", SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i64>]>>;
def SASSpair : SDNode<"SASSISD::PAIR",
  SDTypeProfile<1, 2, [SDTCisVT<0, i64>, SDTCisVT<1, i32>, SDTCisVT<2, i32>]>>;

// 32x32->64 multiply-add (IMAD.WIDE / IMAD.WIDE.U32): a * b + c.
def SDT_MadWide : SDTypeProfile<1, 3, [SDTCisVT<0, i64>, SDTCisVT<1, i32>,
                                       SDTCisVT<2, i32>, SDTCisVT<3, i64>]>;
def SASSmadw : SDNode<"SASSISD::MAD_WIDE", SDT_MadWide>;
def SASSmadwu : SDNode<"SASSISD::MAD_WIDE_U", SDT_MadWide>;

//===----------------------------------------------------------------------===//
// Instruction format
//===----------------------------------------------------------------------===//

// TSFlags: [5:0] latency mnemonic (SASSLatency.td), [6] result tracked by a
// write scoreboard, [7] sources read late (read scoreboard), [11:8] minimum
// stall after issue.
class SASSInst<dag od, dag id, string asm, list<dag> pattern = []>
    : Instruction {
  let Namespace = "SASS";
  let OutOperandList = od;
  let InOperandList = !con(id, (ins pred:$pg));
  let AsmString = asm;
  let Pattern = pattern;
  let Size = 16;
  let isPredicable = 1;
  let hasSideEffects = 0;
  let SchedRW = [WriteALU];

  bits<6> LatMnem = LAT_NOP;
  bit WriteSB = 0;
  bit ReadSB = 0;
  bits<4> MinStall = 1;
  let TSFlags{5-0} = LatMnem;
  let TSFlags{6} = WriteSB;
  let TSFlags{7} = ReadSB;
  let TSFlags{11-8} = MinStall;
}

// Decoupled (scoreboarded) instructions: MIO/UDP pipes.
class SASSVarInst<dag od, dag id, string asm, list<dag> pattern = []>
    : SASSInst<od, id, asm, pattern> {
  let WriteSB = !ne(!size(od), 0);
  let ReadSB = 1;
}

//===----------------------------------------------------------------------===//
// Moves and constants
//===----------------------------------------------------------------------===//

def lo32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 0xffffffffu, SDLoc(N), MVT::i32);
}]>;
def hi32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() >> 32, SDLoc(N), MVT::i32);
}]>;
def fpbits : SDNodeXForm<fpimm, [{
  return CurDAG->getTargetConstant(
      N->getValueAPF().bitcastToAPInt().getZExtValue(), SDLoc(N), MVT::i32);
}]>;

let LatMnem = LAT_MOV in {
let isAsCheapAsAMove = 1, isReMaterializable = 1, isMoveImm = 1 in
def MOVri : SASSInst<(outs GPR32:$d), (ins simm32:$v), "MOV $d, $v",
                     [(set i32:$d, imm:$v)]>;
let isMoveReg = 1 in {
def MOVrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a), "MOV $d, $a">;
def MOV64rr : SASSInst<(outs G64:$d), (ins G64:$a), "MOV.64 $d, $a">;
}
}
def : Pat<(f32 fpimm:$v), (MOVri (fpbits $v))>;
def : Pat<(i64 imm:$v), (REG_SEQUENCE GPR64, (MOVri (lo32 $v)), sub0,
                                             (MOVri (hi32 $v)), sub1)>;

//===----------------------------------------------------------------------===//
// Integer arithmetic
//===----------------------------------------------------------------------===//

let LatMnem = LAT_IADD in {
def IADDrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "IADD $d, $a, $b",
                      [(set i32:$d, (add i32:$a, i32:$b))]>;
def IADDri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, simm32:$b), "IADD $d, $a, $b",
                      [(set i32:$d, (add i32:$a, imm:$b))]>;
}

let LatMnem = LAT_IADD3 in {
def IADD3rrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                        "IADD3 $d, PT, PT, $a, $b, $c",
                        [(set i32:$d, (add (add i32:$a, i32:$b), i32:$c))]>;
def IADD3rri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, simm32:$c),
                        "IADD3 $d, PT, PT, $a, $c, $b",
                        [(set i32:$d, (add (add i32:$a, i32:$b), imm:$c))]>;
def ISUBrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                      "IADD3 $d, PT, PT, $a, -$b, RZ",
                      [(set i32:$d, (sub i32:$a, i32:$b))]>;
}
def : Pat<(sub 0, i32:$b), (ISUBrr (i32 RZ), $b)>;

let LatMnem = LAT_IMAD, SchedRW = [WriteIMAD] in {
def IMADrrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                       "IMAD $d, $a, $b, $c",
                       [(set i32:$d, (add (mul i32:$a, i32:$b), i32:$c))]>;
def IMADrir : SASSInst<(outs GPR32:$d), (ins GPR32:$a, simm32:$b, GPR32:$c),
                       "IMAD $d, $a, $b, $c",
                       [(set i32:$d, (add (mul i32:$a, imm:$b), i32:$c))]>;
def IMADHIU32 : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                         "IMAD.HI.U32 $d, $a, $b, RZ",
                         [(set i32:$d, (mulhu i32:$a, i32:$b))]>;
def IMADHI : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                      "IMAD.HI $d, $a, $b, RZ",
                      [(set i32:$d, (mulhs i32:$a, i32:$b))]>;
}
def : Pat<(mul i32:$a, i32:$b), (IMADrrr $a, $b, (i32 RZ))>;
def : Pat<(mul i32:$a, imm:$b), (IMADrir $a, imm:$b, (i32 RZ))>;

// LOP3.LUT truth tables over a=0xf0, b=0xcc, c=0xaa.
let LatMnem = LAT_LOP3 in {
def LOP3rrr : SASSInst<(outs GPR32:$d),
                       (ins GPR32:$a, GPR32:$b, GPR32:$c, simm32:$lut),
                       "LOP3.LUT $d, $a, $b, $c, $lut, !PT">;
def LOP3rir : SASSInst<(outs GPR32:$d),
                       (ins GPR32:$a, simm32:$b, GPR32:$c, simm32:$lut),
                       "LOP3.LUT $d, $a, $b, $c, $lut, !PT">;
}
let AddedComplexity = 1 in {
def : Pat<(xor i32:$a, -1), (LOP3rrr $a, (i32 RZ), (i32 RZ), 0x0f)>;
def : Pat<(and i32:$a, (not i32:$b)), (LOP3rrr $a, $b, (i32 RZ), 0x30)>;
def : Pat<(xor (xor i32:$a, i32:$b), i32:$c), (LOP3rrr $a, $b, $c, 0x96)>;
def : Pat<(or (and i32:$a, i32:$b), i32:$c), (LOP3rrr $a, $b, $c, 0xea)>;
def : Pat<(and (xor i32:$a, i32:$b), i32:$c), (LOP3rrr $a, $b, $c, 0x28)>;
}
def : Pat<(and i32:$a, i32:$b), (LOP3rrr $a, $b, (i32 RZ), 0xc0)>;
def : Pat<(or i32:$a, i32:$b), (LOP3rrr $a, $b, (i32 RZ), 0xfc)>;
def : Pat<(xor i32:$a, i32:$b), (LOP3rrr $a, $b, (i32 RZ), 0x3c)>;
def : Pat<(and i32:$a, imm:$b), (LOP3rir $a, imm:$b, (i32 RZ), 0xc0)>;
def : Pat<(or i32:$a, imm:$b), (LOP3rir $a, imm:$b, (i32 RZ), 0xfc)>;
def : Pat<(xor i32:$a, imm:$b), (LOP3rir $a, imm:$b, (i32 RZ), 0x3c)>;

let LatMnem = LAT_SHF in {
def SHLrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "SHF.L.U32 $d, $a, $b, RZ",
                     [(set i32:$d, (shl i32:$a, i32:$b))]>;
def SHLri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, simm32:$b), "SHF.L.U32 $d, $a, $b, RZ",
                     [(set i32:$d, (shl i32:$a, (i32 imm:$b)))]>;
def SRLrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "SHF.R.U32.HI $d, RZ, $b, $a",
                     [(set i32:$d, (srl i32:$a, i32:$b))]>;
def SRLri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, simm32:$b), "SHF.R.U32.HI $d, RZ, $b, $a",
                     [(set i32:$d, (srl i32:$a, (i32 imm:$b)))]>;
def SRArr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "SHF.R.S32.HI $d, RZ, $b, $a",
                     [(set i32:$d, (sra i32:$a, i32:$b))]>;
def SRAri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, simm32:$b), "SHF.R.S32.HI $d, RZ, $b, $a",
                     [(set i32:$d, (sra i32:$a, (i32 imm:$b)))]>;
def ROTLrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "SHF.L.W.U32.HI $d, $a, $b, $a",
                      [(set i32:$d, (rotl i32:$a, i32:$b))]>;
def ROTLri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, simm32:$b), "SHF.L.W.U32.HI $d, $a, $b, $a",
                      [(set i32:$d, (rotl i32:$a, (i32 imm:$b)))]>;
// 64-bit funnel halves: {hi:lo} shifted, one 32-bit word of the result.
def SHLU64HI : SASSInst<(outs GPR32:$d), (ins GPR32:$lo, simm32:$s, GPR32:$hi),
                        "SHF.L.U64.HI $d, $lo, $s, $hi">;
def SHRU64 : SASSInst<(outs GPR32:$d), (ins GPR32:$lo, simm32:$s, GPR32:$hi),
                      "SHF.R.U64 $d, $lo, $s, $hi">;
def SHRS64 : SASSInst<(outs GPR32:$d), (ins GPR32:$lo, simm32:$s, GPR32:$hi),
                      "SHF.R.S64 $d, $lo, $s, $hi">;
}
def NegShift32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((32 - N->getZExtValue()) & 31, SDLoc(N), MVT::i32);
}]>;
def : Pat<(rotr i32:$a, (i32 imm:$b)), (ROTLri $a, (NegShift32 imm:$b))>;

let LatMnem = LAT_VIMNMX in {
def SMINrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "VIMNMX.S32 $d, $a, $b, PT",
                      [(set i32:$d, (smin i32:$a, i32:$b))]>;
def SMAXrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "VIMNMX.S32 $d, $a, $b, !PT",
                      [(set i32:$d, (smax i32:$a, i32:$b))]>;
def UMINrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "VIMNMX.U32 $d, $a, $b, PT",
                      [(set i32:$d, (umin i32:$a, i32:$b))]>;
def UMAXrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "VIMNMX.U32 $d, $a, $b, !PT",
                      [(set i32:$d, (umax i32:$a, i32:$b))]>;
}

//===----------------------------------------------------------------------===//
// Comparisons, predicates and selects
//===----------------------------------------------------------------------===//

// icc: bits [2:0] LT LE GT GE EQ NE, bit 3 unsigned.
class CC<CondCode cond, int val> {
  CondCode Code = cond;
  int Val = val;
}
defvar IntCCs = [CC<SETLT, 0>, CC<SETLE, 1>, CC<SETGT, 2>, CC<SETGE, 3>,
                 CC<SETEQ, 4>, CC<SETNE, 5>, CC<SETULT, 8>, CC<SETULE, 9>,
                 CC<SETUGT, 10>, CC<SETUGE, 11>];
// fcc: LT LE GT GE EQ NE (ordered), LTU..NEU (unordered), NUM, NAN.
defvar FpCCs = [CC<SETOLT, 0>, CC<SETOLE, 1>, CC<SETOGT, 2>, CC<SETOGE, 3>,
                CC<SETOEQ, 4>, CC<SETONE, 5>, CC<SETULT, 6>, CC<SETULE, 7>,
                CC<SETUGT, 8>, CC<SETUGE, 9>, CC<SETUEQ, 10>, CC<SETUNE, 11>,
                CC<SETO, 12>, CC<SETUO, 13>, CC<SETLT, 0>, CC<SETLE, 1>,
                CC<SETGT, 2>, CC<SETGE, 3>, CC<SETEQ, 4>, CC<SETNE, 11>];

let LatMnem = LAT_ISETP, SchedRW = [WritePred] in {
def ISETPrr : SASSInst<(outs PRED:$p),
                       (ins GPR32:$a, GPR32:$b, icc:$cc, PRED:$c, negflag:$cn),
                       "ISETP.${cc}.AND $p, PT, $a, $b, ${cn}$c">;
def ISETPri : SASSInst<(outs PRED:$p),
                       (ins GPR32:$a, simm32:$b, icc:$cc, PRED:$c, negflag:$cn),
                       "ISETP.${cc}.AND $p, PT, $a, $b, ${cn}$c">;
def ISETPEXrr : SASSInst<(outs PRED:$p),
                         (ins GPR32:$a, GPR32:$b, icc:$cc, PRED:$x),
                         "ISETP.${cc}.AND.EX $p, PT, $a, $b, PT, $x",
                         [(set i1:$p, (SASSsetcc_ex i32:$a, i32:$b, timm:$cc, i1:$x))]>;
// Predicate copy/negation: (RZ == RZ) AND [!]Ps.
def PMOV : SASSInst<(outs PRED:$p), (ins PRED:$c, negflag:$cn),
                    "ISETP.EQ.AND $p, PT, RZ, RZ, ${cn}$c">;
}
foreach C = IntCCs in {
  def : Pat<(setcc i32:$a, i32:$b, C.Code), (ISETPrr $a, $b, C.Val, (i1 PT), 0)>;
  def : Pat<(setcc i32:$a, imm:$b, C.Code), (ISETPri $a, imm:$b, C.Val, (i1 PT), 0)>;
}
def : Pat<(i1 -1), (PMOV (i1 PT), 0)>;
def : Pat<(i1 0), (PMOV (i1 PT), 1)>;
def : Pat<(xor i1:$p, -1), (PMOV $p, 1)>;

let LatMnem = LAT_PLOP3, SchedRW = [WritePred] in
def PLOP3 : SASSInst<(outs PRED:$d), (ins PRED:$a, PRED:$b, simm32:$lut),
                     "PLOP3.LUT $d, PT, $a, $b, PT, $lut, 0x0">;
def : Pat<(and i1:$a, i1:$b), (PLOP3 $a, $b, 0xc0)>;
def : Pat<(or i1:$a, i1:$b), (PLOP3 $a, $b, 0xfc)>;
def : Pat<(xor i1:$a, i1:$b), (PLOP3 $a, $b, 0x3c)>;

let LatMnem = LAT_SEL in {
def SELrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, PRED:$p, negflag:$pn),
                     "SEL $d, $a, $b, ${pn}$p">;
def SELri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, simm32:$b, PRED:$p, negflag:$pn),
                     "SEL $d, $a, $b, ${pn}$p">;
}
def : Pat<(select i1:$p, i32:$a, i32:$b), (SELrr $a, $b, $p, 0)>;
def : Pat<(select i1:$p, f32:$a, f32:$b), (SELrr $a, $b, $p, 0)>;
def : Pat<(select i1:$p, i32:$a, imm:$b), (SELri $a, imm:$b, $p, 0)>;
def : Pat<(select i1:$p, imm:$a, i32:$b), (SELri $b, imm:$a, $p, 1)>;
def : Pat<(i32 (zext i1:$p)), (SELri (i32 RZ), 1, $p, 1)>;
def : Pat<(i32 (anyext i1:$p)), (SELri (i32 RZ), 1, $p, 1)>;
def : Pat<(i32 (sext i1:$p)), (SELri (i32 RZ), -1, $p, 1)>;
def : Pat<(i1 (trunc i32:$a)), (ISETPri (LOP3rir $a, 1, (i32 RZ), 0xc0), 0, 5, (i1 PT), 0)>;

//===----------------------------------------------------------------------===//
// 64-bit integer
//===----------------------------------------------------------------------===//

def simm32_64 : ImmLeaf<i64, [{ return isInt<32>(Imm); }]>;
def trunc_imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((int32_t)N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;

let LatMnem = LAT_IADD, SchedRW = [WriteWide] in {
def IADD64rr : SASSInst<(outs G64:$d), (ins G64:$a, G64:$b), "IADD.64 $d, $a, $b",
                        [(set i64:$d, (add i64:$a, i64:$b))]>;
def IADD64ri : SASSInst<(outs G64:$d), (ins G64:$a, simm32:$b), "IADD.64 $d, $a, $b">;
def ISUB64rr : SASSInst<(outs G64:$d), (ins G64:$a, G64:$b), "IADD.64 $d, $a, -$b",
                        [(set i64:$d, (sub i64:$a, i64:$b))]>;
}
def : Pat<(add i64:$a, simm32_64:$b), (IADD64ri $a, (trunc_imm $b))>;

let LatMnem = LAT_IMAD, SchedRW = [WriteWide] in {
def IMADWrrr : SASSInst<(outs G64:$d), (ins GPR32:$a, GPR32:$b, G64:$c),
                        "IMAD.WIDE $d, $a, $b, $c">;
def IMADWrir : SASSInst<(outs G64:$d), (ins GPR32:$a, simm32:$b, G64:$c),
                        "IMAD.WIDE $d, $a, $b, $c">;
def IMADWrr0 : SASSInst<(outs G64:$d), (ins GPR32:$a, GPR32:$b),
                        "IMAD.WIDE $d, $a, $b, RZ">;
def IMADWUrrr : SASSInst<(outs G64:$d), (ins GPR32:$a, GPR32:$b, G64:$c),
                         "IMAD.WIDE.U32 $d, $a, $b, $c">;
def IMADWUrir : SASSInst<(outs G64:$d), (ins GPR32:$a, simm32:$b, G64:$c),
                         "IMAD.WIDE.U32 $d, $a, $b, $c">;
def IMADWUrr0 : SASSInst<(outs G64:$d), (ins GPR32:$a, GPR32:$b),
                         "IMAD.WIDE.U32 $d, $a, $b, RZ">;
}
def : Pat<(SASSmadw i32:$a, i32:$b, i64:$c), (IMADWrrr $a, $b, $c)>;
def : Pat<(SASSmadw i32:$a, imm:$b, i64:$c), (IMADWrir $a, imm:$b, $c)>;
def : Pat<(SASSmadwu i32:$a, i32:$b, i64:$c), (IMADWUrrr $a, $b, $c)>;
def : Pat<(SASSmadwu i32:$a, imm:$b, i64:$c), (IMADWUrir $a, imm:$b, $c)>;
def : Pat<(mul (sext i32:$a), (sext i32:$b)), (IMADWrr0 $a, $b)>;
def : Pat<(mul (zext i32:$a), (zext i32:$b)), (IMADWUrr0 $a, $b)>;

def : Pat<(i64 (sext i32:$a)), (REG_SEQUENCE GPR64, $a, sub0, (SRAri $a, 31), sub1)>;
def : Pat<(i64 (zext i32:$a)), (REG_SEQUENCE GPR64, $a, sub0, (MOVri 0), sub1)>;
def : Pat<(i64 (anyext i32:$a)), (REG_SEQUENCE GPR64, $a, sub0, (MOVri 0), sub1)>;
def : Pat<(i32 (trunc i64:$a)), (EXTRACT_SUBREG $a, sub0)>;
def : Pat<(SASSlo i64:$a), (EXTRACT_SUBREG $a, sub0)>;
def : Pat<(SASShi i64:$a), (EXTRACT_SUBREG $a, sub1)>;
def : Pat<(SASSpair i32:$lo, i32:$hi), (REG_SEQUENCE GPR64, $lo, sub0, $hi, sub1)>;
def : Pat<(fshl i32:$hi, i32:$lo, (i32 imm:$c)), (SHLU64HI $lo, imm:$c, $hi)>;
def : Pat<(fshr i32:$hi, i32:$lo, (i32 imm:$c)), (SHRU64 $lo, imm:$c, $hi)>;
def : Pat<(i64 (zext i1:$p)),
          (REG_SEQUENCE GPR64, (SELri (i32 RZ), 1, $p, 1), sub0, (MOVri 0), sub1)>;
def : Pat<(i64 (anyext i1:$p)),
          (REG_SEQUENCE GPR64, (SELri (i32 RZ), 1, $p, 1), sub0, (MOVri 0), sub1)>;
def : Pat<(i64 (sext i1:$p)),
          (REG_SEQUENCE GPR64, (SELri (i32 RZ), -1, $p, 1), sub0,
                               (SELri (i32 RZ), -1, $p, 1), sub1)>;
def : Pat<(i1 (trunc i64:$a)),
          (ISETPri (LOP3rir (EXTRACT_SUBREG $a, sub0), 1, (i32 RZ), 0xc0), 0, 5, (i1 PT), 0)>;

class Logic64<SDNode op, int lut>
    : Pat<(op i64:$a, i64:$b),
          (REG_SEQUENCE GPR64,
             (LOP3rrr (EXTRACT_SUBREG $a, sub0), (EXTRACT_SUBREG $b, sub0), (i32 RZ), lut), sub0,
             (LOP3rrr (EXTRACT_SUBREG $a, sub1), (EXTRACT_SUBREG $b, sub1), (i32 RZ), lut), sub1)>;
def : Logic64<and, 0xc0>;
def : Logic64<or, 0xfc>;
def : Logic64<xor, 0x3c>;
def : Pat<(select i1:$p, i64:$a, i64:$b),
          (REG_SEQUENCE GPR64,
             (SELrr (EXTRACT_SUBREG $a, sub0), (EXTRACT_SUBREG $b, sub0), $p, 0), sub0,
             (SELrr (EXTRACT_SUBREG $a, sub1), (EXTRACT_SUBREG $b, sub1), $p, 0), sub1)>;

//===----------------------------------------------------------------------===//
// FP32
//===----------------------------------------------------------------------===//

let LatMnem = LAT_FADD, SchedRW = [WriteFMA] in {
def FADDrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "FADD $d, $a, $b",
                      [(set f32:$d, (fadd f32:$a, f32:$b))]>;
def FADDri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, fimm32:$b), "FADD $d, $a, $b",
                      [(set f32:$d, (fadd f32:$a, fpimm:$b))]>;
def FSUBrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "FADD $d, $a, -$b",
                      [(set f32:$d, (fsub f32:$a, f32:$b))]>;
def FNEG : SASSInst<(outs GPR32:$d), (ins GPR32:$a), "FADD $d, -$a, -RZ",
                    [(set f32:$d, (fneg f32:$a))]>;
def FABS : SASSInst<(outs GPR32:$d), (ins GPR32:$a), "FADD $d, |$a|, -RZ",
                    [(set f32:$d, (fabs f32:$a))]>;
}
let LatMnem = LAT_FMUL, SchedRW = [WriteFMA] in {
def FMULrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "FMUL $d, $a, $b",
                      [(set f32:$d, (fmul f32:$a, f32:$b))]>;
def FMULri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, fimm32:$b), "FMUL $d, $a, $b",
                      [(set f32:$d, (fmul f32:$a, fpimm:$b))]>;
}
let LatMnem = LAT_FFMA, SchedRW = [WriteFMA] in {
def FFMArrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                       "FFMA $d, $a, $b, $c",
                       [(set f32:$d, (fma f32:$a, f32:$b, f32:$c))]>;
def FFMArir : SASSInst<(outs GPR32:$d), (ins GPR32:$a, fimm32:$b, GPR32:$c),
                       "FFMA $d, $a, $b, $c",
                       [(set f32:$d, (fma f32:$a, fpimm:$b, f32:$c))]>;
def FFMArri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, fimm32:$c),
                       "FFMA $d, $a, $b, $c",
                       [(set f32:$d, (fma f32:$a, f32:$b, fpimm:$c))]>;
def FFMArrn : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                       "FFMA $d, $a, $b, -$c",
                       [(set f32:$d, (fma f32:$a, f32:$b, (fneg f32:$c)))]>;
def FFMAnrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                       "FFMA $d, -$a, $b, $c",
                       [(set f32:$d, (fma (fneg f32:$a), f32:$b, f32:$c))]>;
}
let LatMnem = LAT_FMNMX in {
def FMINrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "FMNMX $d, $a, $b, PT",
                      [(set f32:$d, (fminnum f32:$a, f32:$b))]>;
def FMAXrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b), "FMNMX $d, $a, $b, !PT",
                      [(set f32:$d, (fmaxnum f32:$a, f32:$b))]>;
}
let LatMnem = LAT_FSETP, SchedRW = [WritePred] in {
def FSETPrr : SASSInst<(outs PRED:$p),
                       (ins GPR32:$a, GPR32:$b, fcc:$cc, PRED:$c, negflag:$cn),
                       "FSETP.${cc}.AND $p, PT, $a, $b, ${cn}$c">;
def FSETPri : SASSInst<(outs PRED:$p),
                       (ins GPR32:$a, fimm32:$b, fcc:$cc, PRED:$c, negflag:$cn),
                       "FSETP.${cc}.AND $p, PT, $a, $b, ${cn}$c">;
}
foreach C = FpCCs in {
  def : Pat<(setcc f32:$a, f32:$b, C.Code), (FSETPrr $a, $b, C.Val, (i1 PT), 0)>;
  def : Pat<(setcc f32:$a, fpimm:$b, C.Code), (FSETPri $a, fpimm:$b, C.Val, (i1 PT), 0)>;
}

let LatMnem = LAT_I2FP in {
def I2FS32 : SASSInst<(outs GPR32:$d), (ins GPR32:$a), "I2FP.F32.S32 $d, $a",
                      [(set f32:$d, (sint_to_fp i32:$a))]>;
def I2FU32 : SASSInst<(outs GPR32:$d), (ins GPR32:$a), "I2FP.F32.U32 $d, $a",
                      [(set f32:$d, (uint_to_fp i32:$a))]>;
}
// XU instructions (MUFU, F2I) collect their sources when they win an XU
// admission credit, which can be after issue when the XU queue is busy:
// overwriting a source the next cycle corrupted MUFU.EX2 inputs on GB10, so
// sources stay protected by the read scoreboard.
let LatMnem = LAT_F2I, SchedRW = [WriteSFU] in {
def F2IS32 : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "F2I.TRUNC.NTZ $d, $a",
                         [(set i32:$d, (fp_to_sint f32:$a))]>;
def F2IU32 : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "F2I.U32.TRUNC.NTZ $d, $a",
                         [(set i32:$d, (fp_to_uint f32:$a))]>;
}

// Approximate math, only where the IR allows it.
def arcp_fdiv : PatFrag<(ops node:$a, node:$b), (fdiv node:$a, node:$b), [{
  return N->getFlags().hasAllowReciprocal();
}]>;
def afn_fsqrt : PatFrag<(ops node:$a), (fsqrt node:$a), [{
  return N->getFlags().hasApproximateFuncs();
}]>;
def afn_fexp2 : PatFrag<(ops node:$a), (fexp2 node:$a), [{
  return N->getFlags().hasApproximateFuncs();
}]>;
def afn_flog2 : PatFrag<(ops node:$a), (flog2 node:$a), [{
  return N->getFlags().hasApproximateFuncs();
}]>;
let LatMnem = LAT_MUFU, SchedRW = [WriteSFU] in {
def MUFU_RCP : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.RCP $d, $a">;
def MUFU_SQRT : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.SQRT $d, $a",
                            [(set f32:$d, (afn_fsqrt f32:$a))]>;
def MUFU_EX2 : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.EX2 $d, $a",
                           [(set f32:$d, (afn_fexp2 f32:$a))]>;
def MUFU_LG2 : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.LG2 $d, $a",
                           [(set f32:$d, (afn_flog2 f32:$a))]>;
}
def : Pat<(arcp_fdiv f32:$a, f32:$b), (FMULrr $a, (MUFU_RCP $b))>;

def : Pat<(f32 (bitconvert i32:$a)), (COPY_TO_REGCLASS $a, GPR32)>;
def : Pat<(i32 (bitconvert f32:$a)), (COPY_TO_REGCLASS $a, GPR32)>;

//===----------------------------------------------------------------------===//
// Memory
//===----------------------------------------------------------------------===//

// Global (addrspace 1) accesses go through LDG/STG; NC = non-coherent
// read-only path (LDG.E.CONSTANT) for invariant loads and loads through
// noalias readonly kernel arguments. Generic (addrspace 0) accesses use
// LD/ST, which resolve global, shared and local windows in hardware. Shared
// (addrspace 3) accesses use LDS/STS on 32-bit window addresses.
class GlobalLoad<PatFrag base, bit nc>
    : PatFrag<(ops node:$ptr), (base node:$ptr),
              !strconcat("return isAddrSpace(N, 1) && ",
                         "isNonCoherent(cast<LoadSDNode>(N)) == ",
                         !if(nc, "true", "false"), ";")>;
class SpaceLoad<PatFrag base, int as>
    : PatFrag<(ops node:$ptr), (base node:$ptr),
              "return isAddrSpace(N, " # as # ");">;
class SpaceStore<PatFrag base, int as>
    : PatFrag<(ops node:$v, node:$ptr), (base node:$v, node:$ptr),
              "return isAddrSpace(N, " # as # ");">;

def Op32 : RegisterOperand<GPR32>;

// Shared window address: 32-bit register + signed 24-bit offset.
def mems : Operand<i32> {
  let MIOperandInfo = (ops GPR32, i32imm);
  let PrintMethod = "printSharedAddr";
}
def ADDRs : ComplexPattern<i32, 2, "SelectADDRs", [], []>;

// One width of a memory family: `mn` is the mnemonic prefix ("LDG.E",
// "LD.E", "LDS"), `sz` the width suffix.
multiclass Loads<string mn, string sz, RegisterOperand RC, Operand AddrOp,
                 bit HasNC, list<SchedReadWrite> rw> {
  let mayLoad = 1, LatMnem = !cond(!eq(mn, "LDS"): LAT_LDS, !eq(mn, "LD.E"): LAT_LD,
                                   true: LAT_LDG),
      SchedRW = rw, Uses = !if(!eq(mn, "LDS"), []<Register>, [URD2]) in {
    def "" : SASSVarInst<(outs RC:$d), (ins AddrOp:$addr), mn#sz#" $d, $addr">;
    if HasNC then
      def _NC : SASSVarInst<(outs RC:$d), (ins AddrOp:$addr),
                            mn#sz#".CONSTANT $d, $addr">;
  }
}
multiclass Stores<string mn, string sz, RegisterOperand RC, Operand AddrOp> {
  let mayStore = 1, LatMnem = !cond(!eq(mn, "STS"): LAT_STS, !eq(mn, "ST.E"): LAT_ST,
                                    true: LAT_STG),
      SchedRW = [WriteST], Uses = !if(!eq(mn, "STS"), []<Register>, [URD2]) in
  def "" : SASSVarInst<(outs), (ins AddrOp:$addr, RC:$v), mn#sz#" $addr, $v">;
}

defvar LdRW = [WriteLD];
defm LDG32 : Loads<"LDG.E", "", Op32, memg, 1, LdRW>;
defm LDG64 : Loads<"LDG.E", ".64", G64, memg, 1, LdRW>;
defm LDG128 : Loads<"LDG.E", ".128", G128, memg, 1, LdRW>;
defm LDGU8 : Loads<"LDG.E", ".U8", Op32, memg, 1, LdRW>;
defm LDGS8 : Loads<"LDG.E", ".S8", Op32, memg, 1, LdRW>;
defm LDGU16 : Loads<"LDG.E", ".U16", Op32, memg, 1, LdRW>;
defm LDGS16 : Loads<"LDG.E", ".S16", Op32, memg, 1, LdRW>;
defm LD32 : Loads<"LD.E", "", Op32, memg, 0, LdRW>;
defm LD64 : Loads<"LD.E", ".64", G64, memg, 0, LdRW>;
defm LD128 : Loads<"LD.E", ".128", G128, memg, 0, LdRW>;
defm LDU8 : Loads<"LD.E", ".U8", Op32, memg, 0, LdRW>;
defm LDS8 : Loads<"LD.E", ".S8", Op32, memg, 0, LdRW>;
defm LDU16 : Loads<"LD.E", ".U16", Op32, memg, 0, LdRW>;
defm LDS16 : Loads<"LD.E", ".S16", Op32, memg, 0, LdRW>;
defm LDS32 : Loads<"LDS", "", Op32, mems, 0, [WriteLDS]>;
defm LDS64 : Loads<"LDS", ".64", G64, mems, 0, [WriteLDS]>;
defm LDS128 : Loads<"LDS", ".128", G128, mems, 0, [WriteLDS]>;
defm LDSU8 : Loads<"LDS", ".U8", Op32, mems, 0, [WriteLDS]>;
defm LDSS8 : Loads<"LDS", ".S8", Op32, mems, 0, [WriteLDS]>;
defm LDSU16 : Loads<"LDS", ".U16", Op32, mems, 0, [WriteLDS]>;
defm LDSS16 : Loads<"LDS", ".S16", Op32, mems, 0, [WriteLDS]>;

// Shared loads addressed by a uniform register + offset (formed by
// SASSUniformAddr). The LSU captures a UR address at enqueue, so unlike a
// GPR address it needs no read scoreboard: the UR may be rewritten as soon
// as the load has issued.
def memsu : Operand<i32> {
  let MIOperandInfo = (ops UGPR32, i32imm);
  let PrintMethod = "printSharedAddr";
}
let ReadSB = 0 in {
defm LDS32_UR : Loads<"LDS", "", Op32, memsu, 0, [WriteLDS]>;
defm LDS64_UR : Loads<"LDS", ".64", G64, memsu, 0, [WriteLDS]>;
defm LDS128_UR : Loads<"LDS", ".128", G128, memsu, 0, [WriteLDS]>;
}

defm STG32 : Stores<"STG.E", "", Op32, memg>;
defm STG64 : Stores<"STG.E", ".64", G64, memg>;
defm STG128 : Stores<"STG.E", ".128", G128, memg>;
defm STGU8 : Stores<"STG.E", ".U8", Op32, memg>;
defm STGU16 : Stores<"STG.E", ".U16", Op32, memg>;
defm ST32 : Stores<"ST.E", "", Op32, memg>;
defm ST64 : Stores<"ST.E", ".64", G64, memg>;
defm ST128 : Stores<"ST.E", ".128", G128, memg>;
defm STU8 : Stores<"ST.E", ".U8", Op32, memg>;
defm STU16 : Stores<"ST.E", ".U16", Op32, memg>;
defm STS32 : Stores<"STS", "", Op32, mems>;
defm STS64 : Stores<"STS", ".64", G64, mems>;
defm STS128 : Stores<"STS", ".128", G128, mems>;
defm STSU8 : Stores<"STS", ".U8", Op32, mems>;
defm STSU16 : Stores<"STS", ".U16", Op32, mems>;

// Asynchronous global->shared copies (PTX cp.async.cg), formed by
// SASSAsyncCopy. LDGSTS takes a shared offset of signed 20 bits and a
// global offset of signed 12 bits; it writes no register and reads its
// address registers late. LDGDEPBAR binds all earlier LDGSTS to a counter
// on its write scoreboard, always SB5 (SASS::AsyncCopySB, kept out of the
// general pool by SASSControlCodes); DEPBAR.LE SB5, N waits until at most
// N committed groups are pending.
let mayLoad = 1, mayStore = 1, LatMnem = LAT_STS, ReadSB = 1,
    SchedRW = [WriteST], Uses = [URD2], isPredicable = 0 in
def LDGSTS128 : SASSInst<(outs), (ins mems:$s, memg:$g),
                         "LDGSTS.E.BYPASS.128 $s, $g">;
let hasSideEffects = 1, isPredicable = 0, LatMnem = LAT_NOP in {
// ptxas issues three predicated-off LDS ahead of each group of LDGSTS.
def LDGSTS_PRELUDE : SASSInst<(outs), (ins), "@!PT LDS RZ, [RZ]">;
def LDGDEPBAR : SASSInst<(outs), (ins), "LDGDEPBAR">;
let MinStall = 4 in
def DEPBAR_LE : SASSInst<(outs), (ins i32imm:$n), "DEPBAR.LE SB5, $n">;
}

// Scalar load/store patterns per address space.
multiclass MemPats<int as, string p, ComplexPattern A> {
  defvar nc = !eq(as, 1);
  foreach C = [0, 1] in
  if !or(!eq(C, 0), nc) then {
    defvar sfx = !if(C, "_NC", "");
    defvar LdF = !if(nc, GlobalLoad<load, C>, SpaceLoad<load, as>);
    defvar Z8 = !if(nc, GlobalLoad<zextloadi8, C>, SpaceLoad<zextloadi8, as>);
    defvar S8 = !if(nc, GlobalLoad<sextloadi8, C>, SpaceLoad<sextloadi8, as>);
    defvar E8 = !if(nc, GlobalLoad<extloadi8, C>, SpaceLoad<extloadi8, as>);
    defvar Z16 = !if(nc, GlobalLoad<zextloadi16, C>, SpaceLoad<zextloadi16, as>);
    defvar S16 = !if(nc, GlobalLoad<sextloadi16, C>, SpaceLoad<sextloadi16, as>);
    defvar E16 = !if(nc, GlobalLoad<extloadi16, C>, SpaceLoad<extloadi16, as>);
    def : Pat<(i32 (LdF A:$a)), (!cast<Instruction>(p#"32"#sfx) A:$a)>;
    def : Pat<(f32 (LdF A:$a)), (!cast<Instruction>(p#"32"#sfx) A:$a)>;
    def : Pat<(i64 (LdF A:$a)), (!cast<Instruction>(p#"64"#sfx) A:$a)>;
    def : Pat<(i32 (Z8 A:$a)), (!cast<Instruction>(p#"U8"#sfx) A:$a)>;
    def : Pat<(i32 (E8 A:$a)), (!cast<Instruction>(p#"U8"#sfx) A:$a)>;
    def : Pat<(i32 (S8 A:$a)), (!cast<Instruction>(p#"S8"#sfx) A:$a)>;
    def : Pat<(i32 (Z16 A:$a)), (!cast<Instruction>(p#"U16"#sfx) A:$a)>;
    def : Pat<(i32 (E16 A:$a)), (!cast<Instruction>(p#"U16"#sfx) A:$a)>;
    def : Pat<(i32 (S16 A:$a)), (!cast<Instruction>(p#"S16"#sfx) A:$a)>;
  }
}
multiclass StorePats<int as, string p, ComplexPattern A> {
  def : Pat<(SpaceStore<store, as> i32:$v, A:$a), (!cast<Instruction>(p#"32") A:$a, $v)>;
  def : Pat<(SpaceStore<store, as> f32:$v, A:$a), (!cast<Instruction>(p#"32") A:$a, $v)>;
  def : Pat<(SpaceStore<store, as> i64:$v, A:$a), (!cast<Instruction>(p#"64") A:$a, $v)>;
  def : Pat<(SpaceStore<truncstorei8, as> i32:$v, A:$a), (!cast<Instruction>(p#"U8") A:$a, $v)>;
  def : Pat<(SpaceStore<truncstorei16, as> i32:$v, A:$a), (!cast<Instruction>(p#"U16") A:$a, $v)>;
}
defm : MemPats<1, "LDG", ADDRri>;
defm : MemPats<0, "LD", ADDRri>;
defm : MemPats<3, "LDS", ADDRs>;
defm : StorePats<1, "STG", ADDRri>;
defm : StorePats<0, "ST", ADDRri>;
defm : StorePats<3, "STS", ADDRs>;

// 64/128-bit vector accesses (SASSISD::LDV/STV) are selected in C++: their
// results/operands are register tuples built from scalars.
def SASSldv2 : SDNode<"SASSISD::LDV2", SDTypeProfile<2, 1, [SDTCisSameAs<0, 1>]>,
                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def SASSldv4 : SDNode<"SASSISD::LDV4",
                      SDTypeProfile<4, 1, [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                                           SDTCisSameAs<0, 3>]>,
                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def SASSstv2 : SDNode<"SASSISD::STV2", SDTypeProfile<0, 3, [SDTCisSameAs<0, 1>]>,
                      [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def SASSstv4 : SDNode<"SASSISD::STV4",
                      SDTypeProfile<0, 5, [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                                           SDTCisSameAs<0, 3>]>,
                      [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// Base of this CTA's shared window: (CTA-in-cluster id << 24) + 0x400, past
// the 1 KB the driver reserves at the start of shared memory.
def SASSsmembase : SDNode<"SASSISD::SMEM_BASE", SDTypeProfile<1, 0, [SDTCisVT<0, i32>]>>;
let LatMnem = LAT_S2R, SchedRW = [WriteS2R] in
def S2R_CGA : SASSVarInst<(outs GPR32:$d), (ins), "S2R $d, SR_CgaCtaId">;
let LatMnem = LAT_LEA in
def LEArrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, simm32:$s),
                      "LEA $d, $a, $b, $s",
                      [(set i32:$d, (add (shl i32:$a, (i32 imm:$s)), i32:$b))]>;
def : Pat<(SASSsmembase), (LEArrr (S2R_CGA), (MOVri 0x400), 24)>;

//===----------------------------------------------------------------------===//
// Synchronization, warp shuffles
//===----------------------------------------------------------------------===//

let LatMnem = LAT_BAR, MinStall = 6, hasSideEffects = 1, isConvergent = 1,
    isPredicable = 0 in
def BAR_SYNC : SASSInst<(outs), (ins), "BAR.SYNC.DEFER_BLOCKING 0x0",
                        [(int_nvvm_barrier_cta_sync_aligned_all (i32 0))]>;

// Only full-warp butterflies (mask 0xffffffff, width 32) are supported.
let LatMnem = LAT_SHFL, SchedRW = [WriteLDS], hasSideEffects = 1,
    isConvergent = 1, isPredicable = 0 in {
def SHFLBFLYri : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a, simm32:$l),
                             "SHFL.BFLY PT, $d, $a, $l, 0x1f">;
def SHFLBFLYrr : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$l),
                             "SHFL.BFLY PT, $d, $a, $l, 0x1f">;
}
def : Pat<(i32 (int_nvvm_shfl_sync_bfly_i32 (i32 -1), i32:$a, (i32 imm:$l), (i32 31))),
          (SHFLBFLYri $a, imm:$l)>;
def : Pat<(f32 (int_nvvm_shfl_sync_bfly_f32 (i32 -1), f32:$a, (i32 imm:$l), (i32 31))),
          (SHFLBFLYri $a, imm:$l)>;
def : Pat<(i32 (int_nvvm_shfl_sync_bfly_i32 (i32 -1), i32:$a, i32:$l, (i32 31))),
          (SHFLBFLYrr $a, $l)>;
def : Pat<(f32 (int_nvvm_shfl_sync_bfly_f32 (i32 -1), f32:$a, i32:$l, (i32 31))),
          (SHFLBFLYrr $a, $l)>;

// PTX approximate math maps 1:1 onto MUFU in its flush-to-zero form.
def : Pat<(f32 (int_nvvm_ex2_approx_ftz f32:$a)), (MUFU_EX2 $a)>;
def : Pat<(f32 (int_nvvm_rcp_approx_ftz_f f32:$a)), (MUFU_RCP $a)>;

//===----------------------------------------------------------------------===//
// Constant bank and special registers
//===----------------------------------------------------------------------===//

let LatMnem = LAT_LDC, SchedRW = [WriteLDC], isReMaterializable = 1 in {
def LDC32 : SASSVarInst<(outs GPR32:$d), (ins simm32:$o), "LDC $d, c[0x0][$o]",
                        [(set i32:$d, (SASSldc timm:$o))]>;
def LDC64 : SASSVarInst<(outs G64:$d), (ins simm32:$o), "LDC.64 $d, c[0x0][$o]",
                        [(set i64:$d, (SASSldc timm:$o))]>;
}
def : Pat<(f32 (SASSldc timm:$o)), (LDC32 timm:$o)>;
let LatMnem = LAT_LDCU, SchedRW = [WriteLDC] in
def LDCU64 : SASSVarInst<(outs UG64:$d), (ins simm32:$o), "LDCU.64 $d, c[0x0][$o]">;

class S2R<string sr, Intrinsic intr>
    : SASSVarInst<(outs GPR32:$d), (ins), "S2R $d, "#sr, [(set i32:$d, (intr))]> {
  let LatMnem = LAT_S2R;
  let SchedRW = [WriteS2R];
}
def S2R_TIDX : S2R<"SR_TID.X", int_nvvm_read_ptx_sreg_tid_x>;
def S2R_TIDY : S2R<"SR_TID.Y", int_nvvm_read_ptx_sreg_tid_y>;
def S2R_TIDZ : S2R<"SR_TID.Z", int_nvvm_read_ptx_sreg_tid_z>;
def S2R_CTAIDX : S2R<"SR_CTAID.X", int_nvvm_read_ptx_sreg_ctaid_x>;
def S2R_CTAIDY : S2R<"SR_CTAID.Y", int_nvvm_read_ptx_sreg_ctaid_y>;
def S2R_CTAIDZ : S2R<"SR_CTAID.Z", int_nvvm_read_ptx_sreg_ctaid_z>;
def S2R_LANEID : S2R<"SR_LANEID", int_nvvm_read_ptx_sreg_laneid>;

// Launch geometry lives in constant bank 0 below the parameters.
def : Pat<(int_nvvm_read_ptx_sreg_ntid_x), (LDC32 0x360)>;
def : Pat<(int_nvvm_read_ptx_sreg_ntid_y), (LDC32 0x364)>;
def : Pat<(int_nvvm_read_ptx_sreg_ntid_z), (LDC32 0x368)>;
def : Pat<(int_nvvm_read_ptx_sreg_nctaid_x), (LDC32 0x370)>;
def : Pat<(int_nvvm_read_ptx_sreg_nctaid_y), (LDC32 0x374)>;
def : Pat<(int_nvvm_read_ptx_sreg_nctaid_z), (LDC32 0x378)>;

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

let LatMnem = LAT_BRA, SchedRW = [WriteBR], MinStall = 5 in {
let isBranch = 1, isTerminator = 1, isBarrier = 1 in
def BRA : SASSInst<(outs), (ins brtarget:$t), "BRA $t", [(br bb:$t)]>;
// BRA is predicable by changing it into BRAc (see SASSInstrInfo).
let isBranch = 1, isTerminator = 1 in
def BRAc : SASSInst<(outs), (ins brtarget:$t), "BRA $t">;
}
def : Pat<(brcond i1:$p, bb:$t), (BRAc bb:$t, $p, 0)>;
def : Pat<(brcond (xor i1:$p, -1), bb:$t), (BRAc bb:$t, $p, 1)>;

let LatMnem = LAT_EXIT, SchedRW = [WriteBR], MinStall = 5 in {
let isReturn = 1, isTerminator = 1, isBarrier = 1 in
def EXIT : SASSInst<(outs), (ins), "EXIT", [(SASSret)]>;
let isReturn = 1, isTerminator = 1 in
def EXITc : SASSInst<(outs), (ins), "EXIT">;
}

//===----------------------------------------------------------------------===//
// Uniform datapath (warp-uniform values in UR registers)
//===----------------------------------------------------------------------===//

// Uniform instructions take a uniform predicate guard, which the PRED-typed
// guard operand cannot express: never predicated.
let isPredicable = 0 in {
let LatMnem = LAT_R2UR in
def R2UR : SASSInst<(outs UGPR32:$d), (ins GPR32:$a), "R2UR $d, $a">;
let LatMnem = LAT_UIADD3 in
def UIADD3ri : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a, simm32:$b),
                        "UIADD3 $d, UPT, UPT, $a, $b, URZ">;
let LatMnem = LAT_UMOV in
def UMOV : SASSInst<(outs UGPR32:$d), (ins UGPR32:$a), "UMOV $d, $a">;
}
let LatMnem = LAT_MOV in
def MOVru : SASSInst<(outs GPR32:$d), (ins UGPR32:$a), "MOV $d, $a">;

let LatMnem = LAT_NOP in
def NOP : SASSInst<(outs), (ins), "NOP">;
