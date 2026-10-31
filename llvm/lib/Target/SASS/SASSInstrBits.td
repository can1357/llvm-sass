//===-- SASSInstrBits.td - Bit, byte and warp-reduction instructions -*- tablegen -*-===//
//
// Native forms of the generic bit operations (ctpop, ctlz, bitreverse,
// bswap, abs) and the targets of SASSLiftIdioms: IDP.4A byte dot products
// and REDUX warp reductions.
//
//===----------------------------------------------------------------------===//

let LatMnem = LAT_POPC, SchedRW = [WriteSFU] in
def POPC : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "POPC $d, $a",
                       [(set i32:$d, (ctpop i32:$a))]>;
let LatMnem = LAT_BREV, SchedRW = [WriteSFU] in
def BREV : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "BREV $d, $a",
                       [(set i32:$d, (bitreverse i32:$a))]>;
// Bit index of the most significant one, -1 for zero.
let LatMnem = LAT_FLO, SchedRW = [WriteSFU] in
def FLO : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "FLO.U32 $d, $a">;
let LatMnem = LAT_IADD3 in
def ISUBir : SASSInst<(outs GPR32:$d), (ins simm32:$a, GPR32:$b),
                      "IADD3 $d, PT, PT, -$b, $a, RZ">;
// ctlz(x) = 31 - FLO(x), 32 for zero (as nvcc's __clz).
def : Pat<(ctlz i32:$a), (ISUBir 31, (FLO $a))>;
def : Pat<(ctlz_zero_poison i32:$a), (ISUBir 31, (FLO $a))>;

let LatMnem = LAT_IABS in
def IABS : SASSInst<(outs GPR32:$d), (ins GPR32:$a), "IABS $d, $a",
                    [(set i32:$d, (abs i32:$a))]>;
def : Pat<(bswap i32:$a), (PRMTrir $a, 0x0123, (i32 RZ))>;

// d = c + sum_k byte_k(a) * byte_k(b), bytes signed (S8) or unsigned (U8).
foreach T = [["S8.S8", "s_s"], ["S8.U8", "s_u"], ["U8.S8", "u_s"], ["U8.U8", "u_u"]] in {
  let LatMnem = LAT_IDP, SchedRW = [WriteIMAD] in
  def IDP4A_#T[1] : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b, GPR32:$c),
                             "IDP.4A."#T[0]#" $d, $a, $b, $c",
                             [(set i32:$d, (!cast<Intrinsic>("int_nvvm_idp4a_"#T[1])
                                              i32:$a, i32:$b, i32:$c))]>;
}

// Full-warp reductions into a uniform register.
foreach R = [["add", "SUM.S32"], ["min", "MIN.S32"], ["max", "MAX.S32"],
             ["umin", "MIN"], ["umax", "MAX"], ["and", "AND"], ["or", "OR"],
             ["xor", "XOR"]] in {
  // The intrinsics are modeled as touching inaccessible memory.
  let LatMnem = LAT_REDUX, SchedRW = [WriteLDC], hasSideEffects = 1,
      mayLoad = 1, mayStore = 1, isConvergent = 1, isPredicable = 0 in
  def REDUX_#R[0] : SASSVarInst<(outs UGPR32:$d), (ins GPR32:$a),
                                "REDUX."#R[1]#" $d, $a">;
  def : Pat<(!cast<Intrinsic>("int_nvvm_redux_sync_"#R[0]) i32:$a, (i32 -1)),
            (COPY_TO_REGCLASS (!cast<Instruction>("REDUX_"#R[0]) $a), GPR32)>;
}
