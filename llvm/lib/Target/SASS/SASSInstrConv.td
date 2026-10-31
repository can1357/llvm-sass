//===-- SASSInstrConv.td - Rounding, conversions, division helpers -*- tablegen -*-===//
//
// FRND (round to integral), F2I/I2F with explicit rounding modes and 64-bit
// integers, FADD/FMUL/FFMA with explicit rounding (nvvm.{add,mul,fma}.r?.f),
// FCHK, the range check of the IEEE division fast path, and f64 (DADD,
// DMUL, DFMA, DSETP on the fixed-latency FP64 pipe, conversions). Precise
// f32 division, reciprocal and square root and integer division are
// expanded into these by SASSLowerIEEE / SASSLowerIntDiv. All conversions
// except I2FP run on the MIO pipe (scoreboarded, LAT_F2I).
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Round to integral: FRND
//===----------------------------------------------------------------------===//

// ptxas's plain `FRND` (round to nearest even) is spelled FRND.F32: the
// assembler's defaults for a bare FRND select other formats.
let LatMnem = LAT_F2I, SchedRW = [WriteSFU], FTZ = 1 in {
def FRND_RN : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "FRND.F32 $d, $a",
                          [(set f32:$d, (frint f32:$a))]>;
def FRND_RM : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "FRND.FLOOR $d, $a",
                          [(set f32:$d, (ffloor f32:$a))]>;
def FRND_RP : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "FRND.CEIL $d, $a",
                          [(set f32:$d, (fceil f32:$a))]>;
def FRND_RZ : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "FRND.TRUNC $d, $a",
                          [(set f32:$d, (ftrunc f32:$a))]>;
}
def : Pat<(fnearbyint f32:$a), (FRND_RN $a)>;
def : Pat<(froundeven f32:$a), (FRND_RN $a)>;

//===----------------------------------------------------------------------===//
// f32 -> integer: F2I (saturating; NaN -> 0)
//===----------------------------------------------------------------------===//

// One destination type: `ty` is ".U32", ".S64"/".U64" or "" (s32), `ntz`
// the 32-bit-only ".NTZ" suffix. RN has no rounding modifier.
multiclass F2IRound<string ty, string ntz, RegisterOperand RC, ValueType VT,
                    Instruction RZ, Intrinsic rn, Intrinsic rm, Intrinsic rp,
                    Intrinsic rz> {
  let LatMnem = LAT_F2I, SchedRW = [WriteSFU] in {
  def _RN : SASSVarInst<(outs RC:$d), (ins GPR32:$a), "F2I"#ty#ntz#" $d, $a",
                        [(set VT:$d, (rn f32:$a))]>;
  def _RM : SASSVarInst<(outs RC:$d), (ins GPR32:$a), "F2I"#ty#".FLOOR"#ntz#" $d, $a",
                        [(set VT:$d, (rm f32:$a))]>;
  def _RP : SASSVarInst<(outs RC:$d), (ins GPR32:$a), "F2I"#ty#".CEIL"#ntz#" $d, $a",
                        [(set VT:$d, (rp f32:$a))]>;
  }
  def : Pat<(VT (rz f32:$a)), (RZ $a)>;
}
let LatMnem = LAT_F2I, SchedRW = [WriteSFU] in {
def F2I_S64_RZ : SASSVarInst<(outs G64:$d), (ins GPR32:$a), "F2I.S64.TRUNC $d, $a",
                             [(set i64:$d, (fp_to_sint f32:$a))]>;
def F2I_U64_RZ : SASSVarInst<(outs G64:$d), (ins GPR32:$a), "F2I.U64.TRUNC $d, $a",
                             [(set i64:$d, (fp_to_uint f32:$a))]>;
// F2I.FTZ.U32.TRUNC.NTZ: the integer division's reciprocal estimate.
def F2I_U32_RZ_FTZ : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a),
                                 "F2I.FTZ.U32.TRUNC.NTZ $d, $a",
                                 [(set i32:$d, (int_nvvm_f2ui_rz_ftz f32:$a))]>;
}
// Truncation is F2IS32/F2IU32 (fp_to_sint/fp_to_uint, SASSInstrInfo.td).
defm F2I_S32 : F2IRound<"", ".NTZ", Op32, i32, F2IS32, int_nvvm_f2i_rn,
                        int_nvvm_f2i_rm, int_nvvm_f2i_rp, int_nvvm_f2i_rz>;
defm F2I_U32 : F2IRound<".U32", ".NTZ", Op32, i32, F2IU32, int_nvvm_f2ui_rn,
                        int_nvvm_f2ui_rm, int_nvvm_f2ui_rp, int_nvvm_f2ui_rz>;
defm F2I_S64 : F2IRound<".S64", "", G64, i64, F2I_S64_RZ, int_nvvm_f2ll_rn,
                        int_nvvm_f2ll_rm, int_nvvm_f2ll_rp, int_nvvm_f2ll_rz>;
defm F2I_U64 : F2IRound<".U64", "", G64, i64, F2I_U64_RZ, int_nvvm_f2ull_rn,
                        int_nvvm_f2ull_rm, int_nvvm_f2ull_rp, int_nvvm_f2ull_rz>;

//===----------------------------------------------------------------------===//
// Integer -> f32: I2FP (integer pipe; RN and RZ of 32-bit sources) and I2F
//===----------------------------------------------------------------------===//

let LatMnem = LAT_I2FP in {
def I2FS32_RZ : SASSInst<(outs GPR32:$d), (ins GPR32:$a), "I2FP.F32.S32.RZ $d, $a",
                         [(set f32:$d, (int_nvvm_i2f_rz i32:$a))]>;
def I2FU32_RZ : SASSInst<(outs GPR32:$d), (ins GPR32:$a), "I2FP.F32.U32.RZ $d, $a",
                         [(set f32:$d, (int_nvvm_ui2f_rz i32:$a))]>;
}
def : Pat<(int_nvvm_i2f_rn i32:$a), (I2FS32 $a)>;
def : Pat<(int_nvvm_ui2f_rn i32:$a), (I2FU32 $a)>;

let LatMnem = LAT_F2I, SchedRW = [WriteSFU] in {
def I2FS32_RM : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "I2F.RM $d, $a",
                            [(set f32:$d, (int_nvvm_i2f_rm i32:$a))]>;
def I2FS32_RP : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "I2F.RP $d, $a",
                            [(set f32:$d, (int_nvvm_i2f_rp i32:$a))]>;
def I2FU32_RM : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "I2F.U32.RM $d, $a",
                            [(set f32:$d, (int_nvvm_ui2f_rm i32:$a))]>;
def I2FU32_RP : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "I2F.U32.RP $d, $a",
                            [(set f32:$d, (int_nvvm_ui2f_rp i32:$a))]>;
}

multiclass I2F64Round<string ty, SDNode cvt, Intrinsic rn, Intrinsic rz,
                      Intrinsic rm, Intrinsic rp> {
  let LatMnem = LAT_F2I, SchedRW = [WriteSFU] in {
  def _RN : SASSVarInst<(outs GPR32:$d), (ins G64:$a), "I2F"#ty#" $d, $a",
                        [(set f32:$d, (cvt i64:$a))]>;
  def _RZ : SASSVarInst<(outs GPR32:$d), (ins G64:$a), "I2F"#ty#".RZ $d, $a",
                        [(set f32:$d, (rz i64:$a))]>;
  def _RM : SASSVarInst<(outs GPR32:$d), (ins G64:$a), "I2F"#ty#".RM $d, $a",
                        [(set f32:$d, (rm i64:$a))]>;
  def _RP : SASSVarInst<(outs GPR32:$d), (ins G64:$a), "I2F"#ty#".RP $d, $a",
                        [(set f32:$d, (rp i64:$a))]>;
  }
  def : Pat<(rn i64:$a), (!cast<Instruction>(NAME#"_RN") $a)>;
}
defm I2F_S64 : I2F64Round<".S64", sint_to_fp, int_nvvm_ll2f_rn, int_nvvm_ll2f_rz,
                          int_nvvm_ll2f_rm, int_nvvm_ll2f_rp>;
defm I2F_U64 : I2F64Round<".U64", uint_to_fp, int_nvvm_ull2f_rn, int_nvvm_ull2f_rz,
                          int_nvvm_ull2f_rm, int_nvvm_ull2f_rp>;

//===----------------------------------------------------------------------===//
// f32 arithmetic with an explicit rounding mode (no .FTZ: the nvvm forms
// without .ftz keep denormals)
//===----------------------------------------------------------------------===//

multiclass RoundedArith<string rnd, Intrinsic Add, Intrinsic Mul, Intrinsic Fma> {
  let LatMnem = LAT_FADD, SchedRW = [WriteFMA] in
  def ADD : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                     "FADD"#rnd#" $d, $a, $b", [(set f32:$d, (Add f32:$a, f32:$b))]>;
  let LatMnem = LAT_FMUL, SchedRW = [WriteFMA] in
  def MUL : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                     "FMUL"#rnd#" $d, $a, $b", [(set f32:$d, (Mul f32:$a, f32:$b))]>;
  let LatMnem = LAT_FFMA, SchedRW = [WriteFMA] in
  def FMA : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                     "FFMA"#rnd#" $d, $a, $b, $c",
                     [(set f32:$d, (Fma f32:$a, f32:$b, f32:$c))]>;
}
defm FRZ : RoundedArith<".RZ", int_nvvm_add_rz_f, int_nvvm_mul_rz_f, int_nvvm_fma_rz_f>;
defm FRM : RoundedArith<".RM", int_nvvm_add_rm_f, int_nvvm_mul_rm_f, int_nvvm_fma_rm_f>;
defm FRP : RoundedArith<".RP", int_nvvm_add_rp_f, int_nvvm_mul_rp_f, int_nvvm_fma_rp_f>;
// Round to nearest is the default: the plain instructions (not contracted
// across: these nodes are only ever their own operation).
def : Pat<(int_nvvm_add_rn_f f32:$a, f32:$b), (FADDrr $a, $b)>;
def : Pat<(int_nvvm_mul_rn_f f32:$a, f32:$b), (FMULrr $a, $b)>;
def : Pat<(int_nvvm_fma_rn_f f32:$a, f32:$b, f32:$c), (FFMArrr $a, $b, $c)>;

//===----------------------------------------------------------------------===//
// Division range check
//===----------------------------------------------------------------------===//

// FCHK P, a, b: set when a / b may not be computed exactly by the
// reciprocal + Newton fast path (special or denormal operands, quotient
// near overflow or underflow). Created from SASSLowerIEEE's calls to
// SASS::FChkFn (SASSTargetLowering::lowerBuiltinCall).
def SASSfchk : SDNode<"SASSISD::FCHK", SDTypeProfile<1, 2, [SDTCisVT<0, i1>,
                                                            SDTCisVT<1, f32>,
                                                            SDTCisVT<2, f32>]>>;
let LatMnem = LAT_F2I, SchedRW = [WriteSFU] in
def FCHK : SASSVarInst<(outs PRED:$p), (ins GPR32:$a, GPR32:$b), "FCHK $p, $a, $b",
                       [(set i1:$p, (SASSfchk f32:$a, f32:$b))]>;

//===----------------------------------------------------------------------===//
// f64: arithmetic on the fixed-latency FP64 pipe, conversions on MIO
//===----------------------------------------------------------------------===//

let LatMnem = LAT_DFMA, SchedRW = [WriteWide] in {
def DADDrr : SASSInst<(outs G64:$d), (ins G64:$a, G64:$b), "DADD $d, $a, $b",
                      [(set f64:$d, (fadd f64:$a, f64:$b))]>;
def DSUBrr : SASSInst<(outs G64:$d), (ins G64:$a, G64:$b), "DADD $d, $a, -$b",
                      [(set f64:$d, (fsub f64:$a, f64:$b))]>;
def DMULrr : SASSInst<(outs G64:$d), (ins G64:$a, G64:$b), "DMUL $d, $a, $b",
                      [(set f64:$d, (fmul f64:$a, f64:$b))]>;
def DFMArrr : SASSInst<(outs G64:$d), (ins G64:$a, G64:$b, G64:$c), "DFMA $d, $a, $b, $c",
                       [(set f64:$d, (fma f64:$a, f64:$b, f64:$c))]>;
def DFMArrn : SASSInst<(outs G64:$d), (ins G64:$a, G64:$b, G64:$c), "DFMA $d, $a, $b, -$c",
                       [(set f64:$d, (fma f64:$a, f64:$b, (fneg f64:$c)))]>;
def DFMAnrr : SASSInst<(outs G64:$d), (ins G64:$a, G64:$b, G64:$c), "DFMA $d, -$a, $b, $c",
                       [(set f64:$d, (fma (fneg f64:$a), f64:$b, f64:$c))]>;
}
let LatMnem = LAT_DFMA, SchedRW = [WritePred] in
def DSETPrr : SASSInst<(outs PRED:$p), (ins G64:$a, G64:$b, fcc:$cc),
                       "DSETP.${cc}.AND $p, PT, $a, $b, PT">;
foreach C = FpCCs in
  def : Pat<(setcc f64:$a, f64:$b, C.Code), (DSETPrr $a, $b, C.Val)>;

// Sign operations flip or clear the sign bit (no NaN canonicalization).
def : Pat<(fneg f64:$a),
          (REG_SEQUENCE GPR64, (EXTRACT_SUBREG $a, sub0), sub0,
             (LOP3rir (EXTRACT_SUBREG $a, sub1), 0x80000000, (i32 RZ), 0x3c), sub1)>;
def : Pat<(fabs f64:$a),
          (REG_SEQUENCE GPR64, (EXTRACT_SUBREG $a, sub0), sub0,
             (LOP3rir (EXTRACT_SUBREG $a, sub1), 0x7fffffff, (i32 RZ), 0xc0), sub1)>;
def : Pat<(select i1:$p, f64:$a, f64:$b),
          (REG_SEQUENCE GPR64,
             (SELrr (EXTRACT_SUBREG $a, sub0), (EXTRACT_SUBREG $b, sub0), $p, 0), sub0,
             (SELrr (EXTRACT_SUBREG $a, sub1), (EXTRACT_SUBREG $b, sub1), $p, 0), sub1)>;
def fpbits_lo : SDNodeXForm<fpimm, [{
  return CurDAG->getTargetConstant(
      N->getValueAPF().bitcastToAPInt().getZExtValue() & 0xffffffffu, SDLoc(N), MVT::i32);
}]>;
def fpbits_hi : SDNodeXForm<fpimm, [{
  return CurDAG->getTargetConstant(
      N->getValueAPF().bitcastToAPInt().getZExtValue() >> 32, SDLoc(N), MVT::i32);
}]>;
def : Pat<(f64 fpimm:$v), (REG_SEQUENCE GPR64, (MOVri (fpbits_lo $v)), sub0,
                                               (MOVri (fpbits_hi $v)), sub1)>;
def : Pat<(f64 (bitconvert i64:$a)), (COPY_TO_REGCLASS $a, GPR64)>;
def : Pat<(i64 (bitconvert f64:$a)), (COPY_TO_REGCLASS $a, GPR64)>;
def : Pat<(f64 (SASSldc timm:$o)), (LDC64 timm:$o)>;

let LatMnem = LAT_F2I, SchedRW = [WriteSFU] in {
def F2F_F32_F64 : SASSVarInst<(outs GPR32:$d), (ins G64:$a), "F2F.F32.F64 $d, $a",
                              [(set f32:$d, (fpround f64:$a))]>;
def F2F_F64_F32 : SASSVarInst<(outs G64:$d), (ins GPR32:$a), "F2F.F64.F32 $d, $a",
                              [(set f64:$d, (fpextend f32:$a))]>;
def I2F_F64_S32 : SASSVarInst<(outs G64:$d), (ins GPR32:$a), "I2F.F64 $d, $a",
                              [(set f64:$d, (sint_to_fp i32:$a))]>;
def I2F_F64_U32 : SASSVarInst<(outs G64:$d), (ins GPR32:$a), "I2F.F64.U32 $d, $a",
                              [(set f64:$d, (uint_to_fp i32:$a))]>;
def I2F_F64_S64 : SASSVarInst<(outs G64:$d), (ins G64:$a), "I2F.F64.S64 $d, $a",
                              [(set f64:$d, (sint_to_fp i64:$a))]>;
def I2F_F64_U64 : SASSVarInst<(outs G64:$d), (ins G64:$a), "I2F.F64.U64 $d, $a",
                              [(set f64:$d, (uint_to_fp i64:$a))]>;
def F2I_S32_F64 : SASSVarInst<(outs GPR32:$d), (ins G64:$a), "F2I.F64.TRUNC $d, $a",
                              [(set i32:$d, (fp_to_sint f64:$a))]>;
def F2I_U32_F64 : SASSVarInst<(outs GPR32:$d), (ins G64:$a), "F2I.U32.F64.TRUNC $d, $a",
                              [(set i32:$d, (fp_to_uint f64:$a))]>;
def F2I_S64_F64 : SASSVarInst<(outs G64:$d), (ins G64:$a), "F2I.S64.F64.TRUNC $d, $a",
                              [(set i64:$d, (fp_to_sint f64:$a))]>;
def F2I_U64_F64 : SASSVarInst<(outs G64:$d), (ins G64:$a), "F2I.U64.F64.TRUNC $d, $a",
                              [(set i64:$d, (fp_to_uint f64:$a))]>;
}
