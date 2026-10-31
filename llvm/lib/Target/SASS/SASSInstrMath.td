//===-- SASSInstrMath.td - MUFU, saturation and select idioms -*- tablegen -*-===//
//
// Approximate math primitives (SASSLowerMath reduces everything to these),
// and FP idioms ptxas folds into one instruction: clamp to [0, 1] as .SAT,
// `cmp ? 1.0 : 0.0` as FSET.BF, `p ? x : x op c` as a guarded op.
//
//===----------------------------------------------------------------------===//

def fpzero : FPImmLeaf<f32, [{ return Imm.isPosZero(); }]>;
def fpone : FPImmLeaf<f32, [{ return Imm.isExactlyValue(1.0); }]>;
// The 32-bit pattern of an f32 constant written in an output pattern.
def fbits32 : Operand<i32> { let PrintMethod = "printF32Imm"; }

//===----------------------------------------------------------------------===//
// MUFU: the flush-to-zero approx primitives
//===----------------------------------------------------------------------===//

def afn_ftanh : PatFrag<(ops node:$a), (ftanh node:$a), [{
  return N->getFlags().hasApproximateFuncs();
}]>;
let LatMnem = LAT_MUFU, SchedRW = [WriteSFU] in {
def MUFU_TANH : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.TANH $d, $a",
                            [(set f32:$d, (afn_ftanh f32:$a))]>;
def MUFU_RCP : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.RCP $d, $a",
                           [(set f32:$d, (int_nvvm_rcp_approx_ftz_f f32:$a))]>;
def MUFU_RSQ : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.RSQ $d, $a",
                           [(set f32:$d, (int_nvvm_rsqrt_approx_ftz_f f32:$a))]>;
def MUFU_SQRT : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.SQRT $d, $a",
                            [(set f32:$d, (int_nvvm_sqrt_approx_ftz_f f32:$a))]>;
def MUFU_EX2 : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.EX2 $d, $a",
                           [(set f32:$d, (int_nvvm_ex2_approx_ftz f32:$a))]>;
def MUFU_LG2 : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.LG2 $d, $a",
                           [(set f32:$d, (int_nvvm_lg2_approx_ftz_f f32:$a))]>;
// Inputs in turns: sin.approx(x) is MUFU.SIN(x * 1/(2 pi)), the product
// rounded toward zero.
def MUFU_SIN : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.SIN $d, $a">;
def MUFU_COS : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.COS $d, $a">;
}

let LatMnem = LAT_FMUL, SchedRW = [WriteFMA] in
def FMULRZri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, fbits32:$b),
                        "FMUL.RZ $d, $a, $b">;
def : Pat<(int_nvvm_sin_approx_ftz_f f32:$a),
          (MUFU_SIN (FMULRZri $a, 0x3e22f983))>;
def : Pat<(int_nvvm_cos_approx_ftz_f f32:$a),
          (MUFU_COS (FMULRZri $a, 0x3e22f983))>;

//===----------------------------------------------------------------------===//
// Compares and selects
//===----------------------------------------------------------------------===//

let LatMnem = LAT_FSETP, SchedRW = [WritePred], FTZ = 1 in {
def FSETPabsrr : SASSInst<(outs PRED:$p),
                          (ins GPR32:$a, GPR32:$b, fcc:$cc, PRED:$c, negflag:$cn),
                          "FSETP.${cc}.AND $p, PT, |$a|, $b, ${cn}$c">;
def FSETPabsri : SASSInst<(outs PRED:$p),
                          (ins GPR32:$a, fimm32:$b, fcc:$cc, PRED:$c, negflag:$cn),
                          "FSETP.${cc}.AND $p, PT, |$a|, $b, ${cn}$c">;
}
foreach C = FpCCs in {
  def : Pat<(setcc (fabs f32:$a), f32:$b, C.Code),
            (FSETPabsrr $a, $b, C.Val, (i1 PT), 0)>;
  def : Pat<(setcc (fabs f32:$a), fpimm:$b, C.Code),
            (FSETPabsri $a, fpimm:$b, C.Val, (i1 PT), 0)>;
}

let LatMnem = LAT_FSEL in
def FSELri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, fimm32:$b, PRED:$p, negflag:$pn),
                      "FSEL $d, $a, $b, ${pn}$p">;
def : Pat<(select i1:$p, f32:$a, fpimm:$b), (FSELri $a, fpimm:$b, $p, 0)>;
def : Pat<(select i1:$p, fpimm:$a, f32:$b), (FSELri $b, fpimm:$a, $p, 1)>;

// `cmp ? 1.0 : 0.0` in one instruction (Boolean Float result).
let LatMnem = LAT_FSET, FTZ = 1 in {
def FSETBFrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, fcc:$cc),
                        "FSET.BF.${cc}.AND $d, $a, $b, PT">;
def FSETBFri : SASSInst<(outs GPR32:$d), (ins GPR32:$a, fimm32:$b, fcc:$cc),
                        "FSET.BF.${cc}.AND $d, $a, $b, PT">;
}
foreach C = FpCCs in {
  def : Pat<(select (i1 (setcc f32:$a, f32:$b, C.Code)), fpone, fpzero),
            (FSETBFrr $a, $b, C.Val)>;
  def : Pat<(select (i1 (setcc f32:$a, fpimm:$b, C.Code)), fpone, fpzero),
            (FSETBFri $a, fpimm:$b, C.Val)>;
}

// `p ? x : y op z` as `y op z` guarded by !p and tied to x, as ptxas emits
// the approx range fix-ups (`@!P0 FMUL R0, R0, 0.5`).
let Constraints = "$d = $f", isPredicable = 0, FTZ = 1 in {
let LatMnem = LAT_FMUL, SchedRW = [WriteFMA] in {
def FMULrr_g : SASSInst<(outs GPR32:$d), (ins GPR32:$f, GPR32:$a, GPR32:$b),
                        "FMUL $d, $a, $b">;
def FMULri_g : SASSInst<(outs GPR32:$d), (ins GPR32:$f, GPR32:$a, fimm32:$b),
                        "FMUL $d, $a, $b">;
}
let LatMnem = LAT_FADD, SchedRW = [WriteFMA] in
def FADDri_g : SASSInst<(outs GPR32:$d), (ins GPR32:$f, GPR32:$a, fimm32:$b),
                        "FADD $d, $a, $b">;
}
def : Pat<(select i1:$p, f32:$f, (fmul f32:$a, f32:$b)), (FMULrr_g $f, $a, $b, $p, 1)>;
def : Pat<(select i1:$p, (fmul f32:$a, f32:$b), f32:$f), (FMULrr_g $f, $a, $b, $p, 0)>;
def : Pat<(select i1:$p, f32:$f, (fmul f32:$a, fpimm:$b)),
          (FMULri_g $f, $a, fpimm:$b, $p, 1)>;
def : Pat<(select i1:$p, (fmul f32:$a, fpimm:$b), f32:$f),
          (FMULri_g $f, $a, fpimm:$b, $p, 0)>;
def : Pat<(select i1:$p, f32:$f, (fadd f32:$a, fpimm:$b)),
          (FADDri_g $f, $a, fpimm:$b, $p, 1)>;
def : Pat<(select i1:$p, (fadd f32:$a, fpimm:$b), f32:$f),
          (FADDri_g $f, $a, fpimm:$b, $p, 0)>;

//===----------------------------------------------------------------------===//
// Saturation: min(max(x, 0), 1) and nvvm.saturate fold into .SAT
//===----------------------------------------------------------------------===//

let FTZ = 1 in {
let LatMnem = LAT_FADD, SchedRW = [WriteFMA] in {
def FADDrr_sat : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                          "FADD.SAT $d, $a, $b">;
def FADDri_sat : SASSInst<(outs GPR32:$d), (ins GPR32:$a, fimm32:$b),
                          "FADD.SAT $d, $a, $b">;
}
let LatMnem = LAT_FFMA, SchedRW = [WriteFMA] in {
def FFMArrr_sat : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                           "FFMA.SAT $d, $a, $b, $c">;
def FFMArir_sat : SASSInst<(outs GPR32:$d), (ins GPR32:$a, fimm32:$b, GPR32:$c),
                           "FFMA.SAT $d, $a, $b, $c">;
def FFMArri_sat : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, fimm32:$c),
                           "FFMA.SAT $d, $a, $b, $c">;
}
}
// fmaxnum(NaN, 0) = 0 and .SAT maps NaN to +0; either zero is a valid
// fmaxnum(-0, +0), so the clamp is exactly .SAT.
def clamp01 : PatFrag<(ops node:$x), (fminnum (fmaxnum node:$x, fpzero), fpone)>;
def nvvm_sat : PatFrag<(ops node:$x), (int_nvvm_saturate_f node:$x)>;
def nvvm_sat_ftz : PatFrag<(ops node:$x), (int_nvvm_saturate_ftz_f node:$x)>;
foreach Sat = [clamp01, nvvm_sat, nvvm_sat_ftz] in {
  def : Pat<(Sat (fadd f32:$a, f32:$b)), (FADDrr_sat $a, $b)>;
  def : Pat<(Sat (fadd f32:$a, fpimm:$b)), (FADDri_sat $a, fpimm:$b)>;
  def : Pat<(Sat (fma f32:$a, f32:$b, f32:$c)), (FFMArrr_sat $a, $b, $c)>;
  def : Pat<(Sat (fma f32:$a, fpimm:$b, f32:$c)), (FFMArir_sat $a, fpimm:$b, $c)>;
  def : Pat<(Sat (fma f32:$a, f32:$b, fpimm:$c)), (FFMArri_sat $a, $b, fpimm:$c)>;
  // ptxas's saturated product: a * b + 0.
  def : Pat<(Sat (fmul f32:$a, f32:$b)), (FFMArrr_sat $a, $b, (f32 RZ))>;
  def : Pat<(Sat f32:$a), (FADDrr_sat (f32 RZ), $a)>;
}
