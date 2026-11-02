//===-- SASSInstrTypes.td - f16/bf16/fp8, atomics, warp ops, math -*- tablegen -*-===//
//
// Scalar f16/bf16 values live in the low 16 bits of a 32-bit register; the
// high half is undefined. Instructions read them through the .H0_H0 swizzle
// (both lanes take the low half), so both result halves hold the result.
// Packed v2f16/v2bf16 use the whole register (low half = element 0).
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Half-precision arithmetic (fp16 pipe)
//===----------------------------------------------------------------------===//

// One arithmetic family: `ty` is the type modifier ("" for f16,
// ".BF16_V2" for bf16), `sw` the source swizzle (".H0_H0" for scalars).
multiclass HalfArith<string ty, string sw, ValueType VT> {
  let LatMnem = LAT_HFMA2, SchedRW = [WriteFMA] in {
  def ADD : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                     "HADD2"#ty#" $d, ${a}"#sw#", ${b}"#sw,
                     [(set VT:$d, (fadd VT:$a, VT:$b))]>;
  def SUB : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                     "HADD2"#ty#" $d, ${a}"#sw#", -${b}"#sw,
                     [(set VT:$d, (fsub VT:$a, VT:$b))]>;
  def MUL : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                     "HMUL2"#ty#" $d, ${a}"#sw#", ${b}"#sw,
                     [(set VT:$d, (fmul VT:$a, VT:$b))]>;
  def FMA : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                     "HFMA2"#ty#" $d, ${a}"#sw#", ${b}"#sw#", ${c}"#sw,
                     [(set VT:$d, (fma VT:$a, VT:$b, VT:$c))]>;
  def FMS : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                     "HFMA2"#ty#" $d, ${a}"#sw#", ${b}"#sw#", -${c}"#sw,
                     [(set VT:$d, (fma VT:$a, VT:$b, (fneg VT:$c)))]>;
  def NFMA : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                      "HFMA2"#ty#" $d, -${a}"#sw#", ${b}"#sw#", ${c}"#sw,
                      [(set VT:$d, (fma (fneg VT:$a), VT:$b, VT:$c))]>;
  def MIN : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                     "HMNMX2"#ty#" $d, ${a}"#sw#", ${b}"#sw#", PT",
                     [(set VT:$d, (fminnum VT:$a, VT:$b))]>;
  def MAX : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                     "HMNMX2"#ty#" $d, ${a}"#sw#", ${b}"#sw#", !PT",
                     [(set VT:$d, (fmaxnum VT:$a, VT:$b))]>;
  }
}
defm HF16 : HalfArith<"", ".H0_H0", f16>;
defm HBF16 : HalfArith<".BF16_V2", ".H0_H0", bf16>;
defm HF16X2 : HalfArith<"", "", v2f16>;
defm HBF16X2 : HalfArith<".BF16_V2", "", v2bf16>;

// Scalar compares: HSETP2 sets a predicate per lane; with .H0_H0 sources
// both agree and the second is discarded.
multiclass HalfCompare<string ty, ValueType VT> {
  let LatMnem = LAT_HFMA2, SchedRW = [WritePred] in
  def "" : SASSInst<(outs PRED:$p), (ins GPR32:$a, GPR32:$b, fcc:$cc),
                    "HSETP2"#ty#".${cc}.AND $p, PT, ${a}.H0_H0, ${b}.H0_H0, PT">;
  foreach C = FpCCs in
    def : Pat<(setcc VT:$a, VT:$b, C.Code),
              (!cast<Instruction>(NAME) $a, $b, C.Val)>;
}
defm HSETP2F16 : HalfCompare<"", f16>;
defm HSETP2BF16 : HalfCompare<".BF16_V2", bf16>;

// Sign operations are bit operations (IEEE negate/abs never round).
class SignMask<ValueType vt, int sign> {
  ValueType VT = vt;
  int Sign = sign;
}
foreach P = [SignMask<f16, 0x8000>, SignMask<bf16, 0x8000>,
             SignMask<v2f16, 0x80008000>, SignMask<v2bf16, 0x80008000>] in {
  def : Pat<(fneg P.VT:$a), (LOP3rir $a, P.Sign, (i32 RZ), 0x3c)>;
  def : Pat<(fabs P.VT:$a), (LOP3rir $a, !sub(0xffffffff, P.Sign), (i32 RZ), 0xc0)>;
}

foreach VT = [f16, bf16, v2f16, v2bf16] in
  def : Pat<(select i1:$p, VT:$a, VT:$b), (SELrr $a, $b, $p, 0)>;
foreach VT = [f16, bf16] in
  def : Pat<(VT fpimm:$v), (MOVri (fpbits $v))>;

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

let LatMnem = LAT_PRMT in
def PRMTrir : SASSInst<(outs GPR32:$d), (ins GPR32:$a, simm32:$s, GPR32:$c),
                       "PRMT $d, $a, $s, $c">;

let LatMnem = LAT_HFMA2, SchedRW = [WriteFMA] in
def HF16TOF32 : SASSInst<(outs GPR32:$d), (ins GPR32:$a),
                         "HADD2.F32 $d, -RZ, ${a}.H0_H0",
                         [(set f32:$d, (fpextend f16:$a))]>;
def : Pat<(f32 (fpextend bf16:$a)), (SHLri $a, 16)>;

// F2FP.PACK_AB: d.hi = round(a), d.lo = round(b).
let LatMnem = LAT_PRMT in {
def F2FP_F16 : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                        "F2FP.F16.F32.PACK_AB $d, $a, $b">;
def F2FP_BF16 : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                         "F2FP.BF16.F32.PACK_AB $d, $a, $b">;
}
def : Pat<(f16 (fpround f32:$a)), (F2FP_F16 (i32 RZ), $a)>;
def : Pat<(bf16 (fpround f32:$a)), (F2FP_BF16 (i32 RZ), $a)>;
let AddedComplexity = 1 in {
def : Pat<(v2f16 (build_vector (f16 (fpround f32:$lo)), (f16 (fpround f32:$hi)))),
          (F2FP_F16 $hi, $lo)>;
def : Pat<(v2bf16 (build_vector (bf16 (fpround f32:$lo)), (bf16 (fpround f32:$hi)))),
          (F2FP_BF16 $hi, $lo)>;
}

let LatMnem = LAT_F2I, SchedRW = [WriteSFU] in {
def I2F_F16 : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "I2F.F16 $d, $a",
                          [(set f16:$d, (sint_to_fp i32:$a))]>;
def I2F_F16U : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "I2F.F16.U32 $d, $a",
                           [(set f16:$d, (uint_to_fp i32:$a))]>;
def I2F_BF16 : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "I2F.BF16 $d, $a",
                           [(set bf16:$d, (sint_to_fp i32:$a))]>;
def I2F_BF16U : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "I2F.BF16.U32 $d, $a",
                            [(set bf16:$d, (uint_to_fp i32:$a))]>;
}
// Half -> integer: the f32 widening is exact.
def : Pat<(i32 (fp_to_sint f16:$a)), (F2IS32 (HF16TOF32 $a))>;
def : Pat<(i32 (fp_to_uint f16:$a)), (F2IU32 (HF16TOF32 $a))>;
def : Pat<(i32 (fp_to_sint bf16:$a)), (F2IS32 (SHLri $a, 16))>;
def : Pat<(i32 (fp_to_uint bf16:$a)), (F2IU32 (SHLri $a, 16))>;

// The 16 bits of a scalar half as the low half of an i32 (upper bits
// undefined), and back: bitcasts between f16/bf16 and the illegal i16.
def SASShalfbits : SDNode<"SASSISD::HALF_BITS", SDTypeProfile<1, 1, [SDTCisVT<0, i32>]>>;
def SASSbitshalf : SDNode<"SASSISD::BITS_HALF", SDTypeProfile<1, 1, [SDTCisVT<1, i32>]>>;

// Packed <-> scalar halves.
class HalfPair<ValueType v, ValueType s> {
  ValueType V = v;
  ValueType S = s;
}
foreach P = [HalfPair<v2f16, f16>, HalfPair<v2bf16, bf16>] in {
  defvar V = P.V;
  defvar S = P.S;
  def : Pat<(V (build_vector S:$lo, S:$hi)), (PRMTrir $lo, 0x5410, $hi)>;
  def : Pat<(S (extractelt V:$v, 0)), (COPY_TO_REGCLASS $v, GPR32)>;
  def : Pat<(S (extractelt V:$v, 1)), (PRMTrir $v, 0x7632, (i32 RZ))>;
  // Raw bits of a scalar half (SASSISD::HALF_BITS / BITS_HALF).
  def : Pat<(i32 (SASShalfbits S:$a)), (COPY_TO_REGCLASS $a, GPR32)>;
  def : Pat<(S (SASSbitshalf i32:$a)), (COPY_TO_REGCLASS $a, GPR32)>;
}

// Half loads and stores move the 16 bits (LD*.U16 / ST*.U16).
foreach VT = [f16, bf16] in {
  foreach NC = [0, 1] in
    def : Pat<(VT (GlobalLoad<load, NC> ADDRri:$a)),
              (!cast<Instruction>("LDGU16"#!if(NC, "_NC", "")) ADDRri:$a)>;
  def : Pat<(VT (SpaceLoad<load, 0> ADDRri:$a)), (LDU16 ADDRri:$a)>;
  def : Pat<(VT (SpaceLoad<load, 3> ADDRs:$a)), (LDSU16 ADDRs:$a)>;
  def : Pat<(SpaceStore<store, 1> VT:$v, ADDRri:$a), (STGU16 ADDRri:$a, $v)>;
  def : Pat<(SpaceStore<store, 0> VT:$v, ADDRri:$a), (STU16 ADDRri:$a, $v)>;
  def : Pat<(SpaceStore<store, 3> VT:$v, ADDRs:$a), (STSU16 ADDRs:$a, $v)>;
}

//===----------------------------------------------------------------------===//
// Atomics, strong (coherent) loads/stores and fences. Selected in C++
// (SASSISelAtomics.cpp): the opcode depends on the operation, the address
// space, the memory-model scope and whether the result is used.
//===----------------------------------------------------------------------===//

def scope : Operand<i32> { let PrintMethod = "printScope"; }
// Global address without the descriptor, as ATOMG.CAS takes it.
def memgr : Operand<i64> {
  let MIOperandInfo = (ops GPR64, i32imm);
  let PrintMethod = "printRawGlobalAddr";
}

// One read-modify-write operation `op` (the modifier after ATOM*.E.):
// G = ATOMG (global), F = ATOM (generic), RG/RF = REDG/RED (no result),
// S = ATOMS (shared) when the shared form exists.
multiclass AtomRMW<string op, RegisterOperand RC, bit HasRed, bit HasShared> {
  let mayLoad = 1, mayStore = 1, hasSideEffects = 1, LatMnem = LAT_LDG,
      SchedRW = [WriteLD], Uses = [URD2] in {
    def G : SASSVarInst<(outs RC:$d), (ins memg:$a, RC:$v, scope:$sc),
                        "ATOMG.E."#op#".STRONG.${sc} PT, $d, $a, $v">;
    def F : SASSVarInst<(outs RC:$d), (ins memg:$a, RC:$v, scope:$sc),
                        "ATOM.E."#op#".STRONG.${sc} PT, $d, $a, $v">;
    if HasRed then {
      def RG : SASSVarInst<(outs), (ins memg:$a, RC:$v, scope:$sc),
                           "REDG.E."#op#".STRONG.${sc} $a, $v">;
      def RF : SASSVarInst<(outs), (ins memg:$a, RC:$v, scope:$sc),
                           "RED.E."#op#".STRONG.${sc} $a, $v">;
    }
  }
  if HasShared then
  let mayLoad = 1, mayStore = 1, hasSideEffects = 1, LatMnem = LAT_LDS,
      SchedRW = [WriteLDS] in
  def S : SASSVarInst<(outs RC:$d), (ins mems:$a, RC:$v), "ATOMS."#op#" $d, $a, $v">;
}
defm AT_ADD : AtomRMW<"ADD", Op32, 1, 1>;
defm AT_SMIN : AtomRMW<"MIN.S32", Op32, 1, 1>;
defm AT_SMAX : AtomRMW<"MAX.S32", Op32, 1, 1>;
defm AT_UMIN : AtomRMW<"MIN", Op32, 1, 1>;
defm AT_UMAX : AtomRMW<"MAX", Op32, 1, 1>;
defm AT_AND : AtomRMW<"AND", Op32, 1, 1>;
defm AT_OR : AtomRMW<"OR", Op32, 1, 1>;
defm AT_XOR : AtomRMW<"XOR", Op32, 1, 1>;
defm AT_INC : AtomRMW<"INC", Op32, 1, 1>;
defm AT_DEC : AtomRMW<"DEC", Op32, 1, 1>;
defm AT_EXCH : AtomRMW<"EXCH", Op32, 0, 1>;
defm AT_FADD : AtomRMW<"ADD.F32.FTZ.RN", Op32, 1, 0>;
// sm_90 vector forms, atomic per element (SASSPipeBalance joins the scalars).
defm AT_FADDX2 : AtomRMW<"ADD.F32x2.FTZ.RN", G64, 1, 0>;
defm AT_FADDX4 : AtomRMW<"ADD.F32x4.FTZ.RN", G128, 1, 0>;
// Packed halves (f16x2/bf16x2 per 32-bit word; x4/x8 joined by SASSPipeBalance).
defm AT_HADD2 : AtomRMW<"ADD.F16x2.RN", Op32, 1, 0>;
defm AT_HADD4 : AtomRMW<"ADD.F16x4.RN", G64, 1, 0>;
defm AT_HADD8 : AtomRMW<"ADD.F16x8.RN", G128, 1, 0>;
defm AT_BADD2 : AtomRMW<"ADD.BF16x2.RN", Op32, 1, 0>;
defm AT_BADD4 : AtomRMW<"ADD.BF16x4.RN", G64, 1, 0>;
defm AT_BADD8 : AtomRMW<"ADD.BF16x8.RN", G128, 1, 0>;
defm AT_ADD64 : AtomRMW<"ADD.64", G64, 1, 0>;
defm AT_SMIN64 : AtomRMW<"MIN.S64", G64, 1, 0>;
defm AT_SMAX64 : AtomRMW<"MAX.S64", G64, 1, 0>;
defm AT_UMIN64 : AtomRMW<"MIN.64", G64, 1, 0>;
defm AT_UMAX64 : AtomRMW<"MAX.64", G64, 1, 0>;
defm AT_AND64 : AtomRMW<"AND.64", G64, 1, 0>;
defm AT_OR64 : AtomRMW<"OR.64", G64, 1, 0>;
defm AT_XOR64 : AtomRMW<"XOR.64", G64, 1, 0>;
defm AT_EXCH64 : AtomRMW<"EXCH.64", G64, 0, 1>;

// Compare-and-swap: d = [a]; if (d == c) [a] = n.
multiclass AtomCAS<string sz, RegisterOperand RC> {
  let mayLoad = 1, mayStore = 1, hasSideEffects = 1, LatMnem = LAT_LDG,
      SchedRW = [WriteLD], Uses = [URD2] in {
    def G : SASSVarInst<(outs RC:$d), (ins memgr:$a, RC:$c, RC:$n, scope:$sc),
                        "ATOMG.E.CAS"#sz#".STRONG.${sc} PT, $d, $a, $c, $n">;
    def F : SASSVarInst<(outs RC:$d), (ins memg:$a, RC:$c, RC:$n, scope:$sc),
                        "ATOM.E.CAS"#sz#".STRONG.${sc} PT, $d, $a, $c, $n">;
  }
  let mayLoad = 1, mayStore = 1, hasSideEffects = 1, LatMnem = LAT_LDS,
      SchedRW = [WriteLDS] in
  def S : SASSVarInst<(outs RC:$d), (ins mems:$a, RC:$c, RC:$n),
                      "ATOMS.CAS"#sz#" $d, $a, $c, $n">;
}
defm AT_CAS : AtomCAS<"", Op32>;
defm AT_CAS64 : AtomCAS<".64", G64>;

// Coherent accesses at a scope (atomic loads/stores, volatile).
multiclass StrongLoad<string mn, string sz, RegisterOperand RC> {
  let mayLoad = 1, hasSideEffects = 1, LatMnem = !if(!eq(mn, "LD.E"), LAT_LD, LAT_LDG),
      SchedRW = [WriteLD], Uses = [URD2] in
  def "" : SASSVarInst<(outs RC:$d), (ins memg:$addr, scope:$sc),
                       mn#sz#".STRONG.${sc} $d, $addr">;
}
multiclass StrongStore<string mn, string sz, RegisterOperand RC> {
  let mayStore = 1, hasSideEffects = 1, LatMnem = !if(!eq(mn, "ST.E"), LAT_ST, LAT_STG),
      SchedRW = [WriteST], Uses = [URD2] in
  def "" : SASSVarInst<(outs), (ins memg:$addr, RC:$v, scope:$sc),
                       mn#sz#".STRONG.${sc} $addr, $v">;
}
foreach Sp = [["LDG.E", "STG.E", "G"], ["LD.E", "ST.E", "F"]] in {
  defm SLD#Sp[2]#32 : StrongLoad<Sp[0], "", Op32>;
  defm SLD#Sp[2]#64 : StrongLoad<Sp[0], ".64", G64>;
  defm SLD#Sp[2]#128 : StrongLoad<Sp[0], ".128", G128>;
  defm SLD#Sp[2]#U8 : StrongLoad<Sp[0], ".U8", Op32>;
  defm SLD#Sp[2]#S8 : StrongLoad<Sp[0], ".S8", Op32>;
  defm SLD#Sp[2]#U16 : StrongLoad<Sp[0], ".U16", Op32>;
  defm SLD#Sp[2]#S16 : StrongLoad<Sp[0], ".S16", Op32>;
  defm SST#Sp[2]#32 : StrongStore<Sp[1], "", Op32>;
  defm SST#Sp[2]#64 : StrongStore<Sp[1], ".64", G64>;
  defm SST#Sp[2]#128 : StrongStore<Sp[1], ".128", G128>;
  defm SST#Sp[2]#U8 : StrongStore<Sp[1], ".U8", Op32>;
  defm SST#Sp[2]#U16 : StrongStore<Sp[1], ".U16", Op32>;
}

// Fences (MIO pipe). A fence at GPU/system scope is MEMBAR + ERRBAR +
// CGAERRBAR, and an acquiring one also invalidates L1 (CCTL.IVALL), as
// ptxas emits for fence.{sc,acq_rel}.{gpu,sys} and ld.acquire/st.release.
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, LatMnem = LAT_ST,
    SchedRW = [WriteST], isPredicable = 0 in {
foreach K = ["SC", "ALL"] in
  foreach S = ["CTA", "GPU", "SYS"] in
    def MEMBAR_#K#_#S : SASSInst<(outs), (ins), "MEMBAR."#K#"."#S>;
def ERRBAR : SASSInst<(outs), (ins), "ERRBAR">;
def CGAERRBAR : SASSInst<(outs), (ins), "CGAERRBAR">;
def CCTL_IVALL : SASSInst<(outs), (ins), "CCTL.IVALL">;
}
