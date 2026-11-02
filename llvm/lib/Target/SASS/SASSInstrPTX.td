//===-- SASSInstrPTX.td - Instructions behind inline PTX ---*- tablegen -*-===//
//
// SASS for the PTX features kernels reach through inline asm. The asm is
// rewritten into NVVM intrinsics by SASSExpandInlinePTX; these are the
// instructions and patterns those intrinsics select to (sm_90 forms, as
// ptxas emits them). Intrinsics whose selection needs C++ (register
// tuples, uniform operands) are handled in SASSISelPTX.cpp.
//
//===----------------------------------------------------------------------===//

// PRMT d, a, sel, b: bytes of {b, a} picked by the selector's nibbles
// (PRMTrir is in SASSInstrTypes.td).
let LatMnem = LAT_PRMT in
def PRMTrrr : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$s, GPR32:$b),
                       "PRMT $d, $a, $s, $b">;
def : Pat<(int_nvvm_prmt i32:$a, i32:$b, i32:$c), (PRMTrrr $a, $c, $b)>;
def : Pat<(int_nvvm_prmt i32:$a, i32:$b, (i32 imm:$c)), (PRMTrir $a, imm:$c, $b)>;
let AddedComplexity = 1 in {
def : Pat<(int_nvvm_prmt i32:$a, (i32 0), i32:$c), (PRMTrrr $a, $c, (i32 RZ))>;
def : Pat<(int_nvvm_prmt i32:$a, (i32 0), (i32 imm:$c)), (PRMTrir $a, imm:$c, (i32 RZ))>;
}

// ex2.approx.f16x2 (tanh.approx.f32 is MUFU_TANH in SASSInstrMath.td).
let LatMnem = LAT_MUFU, SchedRW = [WriteSFU] in {
def MUFU_EX2_F16 : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MUFU.EX2.F16 $d, $a">;
def MUFU_EX2_F16_H1 : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a),
                                  "MUFU.EX2.F16.H1 $d, $a">;
}
// Each half separately, packed back with PRMT (as ptxas does).
def : Pat<(v2f16 (int_nvvm_ex2_approx v2f16:$a)),
          (COPY_TO_REGCLASS (PRMTrir (MUFU_EX2_F16 $a), 0x5410, (MUFU_EX2_F16_H1 $a)),
                            GPR32)>;

// Named barriers: bar.sync / barrier.sync / bar.arrive with an id and an
// optional thread count, each an immediate or a register. Register operands
// are read after issue: a read scoreboard holds back later writers (ptxas
// sets one on every BAR with a register; without it a barrier id register
// rewritten right after BAR.SYNC sends the warp to another barrier).
let LatMnem = LAT_BAR, MinStall = 6, hasSideEffects = 1, isConvergent = 1,
    isPredicable = 0, NoGuard = 1, ReadSB = 1 in {
def BAR_SYNC_I : SASSInst<(outs), (ins simm32:$id), "BAR.SYNC.DEFER_BLOCKING $id">;
def BAR_SYNC_R : SASSInst<(outs), (ins GPR32:$id), "BAR.SYNC.DEFER_BLOCKING $id">;
foreach Kind = [["SYNC", "BAR.SYNC.DEFER_BLOCKING"], ["ARV", "BAR.ARV"]] in {
def BAR_#Kind[0]#_II : SASSInst<(outs), (ins simm32:$id, simm32:$n), Kind[1]#" $id, $n">;
def BAR_#Kind[0]#_IR : SASSInst<(outs), (ins simm32:$id, GPR32:$n), Kind[1]#" $id, $n">;
def BAR_#Kind[0]#_RI : SASSInst<(outs), (ins GPR32:$id, simm32:$n), Kind[1]#" $id, $n">;
def BAR_#Kind[0]#_RR : SASSInst<(outs), (ins GPR32:$id, GPR32:$n), Kind[1]#" $id, $n">;
}
}
foreach Sync = [int_nvvm_barrier_cta_sync_aligned_all, int_nvvm_barrier_cta_sync_all] in {
def : Pat<(Sync (i32 imm:$id)), (BAR_SYNC_I imm:$id)>;
def : Pat<(Sync i32:$id), (BAR_SYNC_R $id)>;
}
foreach P = [["SYNC", "int_nvvm_barrier_cta_sync_aligned_count"],
             ["SYNC", "int_nvvm_barrier_cta_sync_count"],
             ["ARV", "int_nvvm_barrier_cta_arrive_aligned_count"],
             ["ARV", "int_nvvm_barrier_cta_arrive_count"]] in {
defvar Intr = !cast<Intrinsic>(P[1]);
def : Pat<(Intr (i32 imm:$id), (i32 imm:$n)),
          (!cast<Instruction>("BAR_"#P[0]#"_II") imm:$id, imm:$n)>;
def : Pat<(Intr (i32 imm:$id), i32:$n), (!cast<Instruction>("BAR_"#P[0]#"_IR") imm:$id, $n)>;
def : Pat<(Intr i32:$id, (i32 imm:$n)), (!cast<Instruction>("BAR_"#P[0]#"_RI") $id, imm:$n)>;
def : Pat<(Intr i32:$id, i32:$n), (!cast<Instruction>("BAR_"#P[0]#"_RR") $id, $n)>;
}

// Programmatic dependent launch, traps (control-flow unit).
let LatMnem = LAT_BRA, hasSideEffects = 1, isPredicable = 0, NoGuard = 1 in {
def ACQBULK : SASSInst<(outs), (ins), "ACQBULK", [(int_nvvm_griddepcontrol_wait)]>;
def PREEXIT : SASSInst<(outs), (ins), "PREEXIT", [(int_nvvm_griddepcontrol_launch_dependents)]>;
def BPT_TRAP : SASSInst<(outs), (ins), "BPT.TRAP 0x1", [(trap)]>;
}
// __nanosleep(t): NANOSLEEP on the control-flow unit.
let LatMnem = LAT_BRA, hasSideEffects = 1 in {
def NANOSLEEPi : SASSInst<(outs), (ins simm32:$t), "NANOSLEEP $t",
                          [(int_nvvm_nanosleep (i32 imm:$t))]>;
def NANOSLEEPr : SASSInst<(outs), (ins GPR32:$t), "NANOSLEEP $t",
                          [(int_nvvm_nanosleep i32:$t)]>;
}

// Special registers. CS2R reads 64-bit counters on the integer pipe.
let LatMnem = LAT_SEL, hasSideEffects = 1 in {
def CS2R_GLOBALTIMER : SASSInst<(outs G64:$d), (ins), "CS2R $d, SR_GLOBALTIMERLO",
                                [(set i64:$d, (int_nvvm_read_ptx_sreg_globaltimer))]>;
def CS2R_CLOCK64 : SASSInst<(outs G64:$d), (ins), "CS2R $d, SR_CLOCKLO",
                            [(set i64:$d, (int_nvvm_read_ptx_sreg_clock64))]>;
}
// Zero into a register pair (ptxas clears accumulators this way); formed
// after allocation by SASSZeroPairs from two MOV Rn, 0x0.
let LatMnem = LAT_SEL, isAsCheapAsAMove = 1 in
def CS2R_SRZ : SASSInst<(outs G64:$d), (ins), "CS2R $d, SRZ">;
def : Pat<(int_nvvm_read_ptx_sreg_cluster_ctarank), (S2R_CGA)>;
let Predicates = [NoBlackwellISA] in
def : Pat<(int_nvvm_read_ptx_sreg_cluster_nctarank), (LDC32 0x188)>;

// Prefetches into L1/L2 (CCTL on a 64-bit generic or global address). The
// address is read after issue, as for any MIO instruction, so it is released
// on a read scoreboard (ptxas: CCTL.E.PF2 [R4] ;[wr=7:rd=1]); without one a
// later write of the address register can redirect the prefetch.
let LatMnem = LAT_LDG, hasSideEffects = 1, mayLoad = 1, Uses = [URD2] in {
def CCTL_PF1 : SASSVarInst<(outs), (ins G64:$a), "CCTL.E.PF1 [$a]">;
def CCTL_PF2 : SASSVarInst<(outs), (ins G64:$a), "CCTL.E.PF2 [$a]">;
}
def : Pat<(int_nvvm_prefetch_global_L2 i64:$a), (CCTL_PF2 $a)>;
def : Pat<(int_nvvm_prefetch_L2 i64:$a), (CCTL_PF2 $a)>;
def : Pat<(int_nvvm_prefetch_global_L1 i64:$a), (CCTL_PF1 $a)>;
def : Pat<(int_nvvm_prefetch_L1 i64:$a), (CCTL_PF1 $a)>;
def : Pat<(int_nvvm_prefetch_global_L2_evict_last i64:$a), (CCTL_PF2 $a)>;
def : Pat<(int_nvvm_prefetch_global_L2_evict_normal i64:$a), (CCTL_PF2 $a)>;

//===----------------------------------------------------------------------===//
// FP8 conversions (cvt.*.e4m3x2 / e5m2x2). F2FP runs on the integer pipe.
//===----------------------------------------------------------------------===//

// The NVVM intrinsics carry FP8 pairs as i16, an illegal type here:
// SASSISelPTX.cpp rewrites them into these i32 nodes (packed pair in the
// low 16 bits).
def SDT_F2FP8F32 : SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisVT<1, f32>, SDTCisVT<2, f32>]>;
def SDT_F2FP8F16 : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, v2f16>]>;
def SDT_FP8F16 : SDTypeProfile<1, 1, [SDTCisVT<0, v2f16>, SDTCisVT<1, i32>]>;
// sm_90 has no FP8 mma.sync either: SASSExpandInlinePTX converts FP8
// fragments to f16x2 with these and uses HMMA.
multiclass F2FP8<string fmt, string node> {
  def _F32_NODE : SDNode<"SASSISD::F32_TO_"#node, SDT_F2FP8F32>;
  def _F16_NODE : SDNode<"SASSISD::F16_TO_"#node, SDT_F2FP8F16>;
  def _UNPACK_NODE : SDNode<"SASSISD::"#node#"_TO_F16", SDT_FP8F16>;
  let LatMnem = LAT_PRMT in {
  // d.lo16 = {cvt(a) high byte, cvt(b) low byte}
  def _F32 : SASSInst<(outs GPR32:$d), (ins GPR32:$a, GPR32:$b),
                      "F2FP.SATFINITE."#fmt#".F32.PACK_AB_MERGE_C $d, $a, $b, RZ",
                      [(set i32:$d, (!cast<SDNode>(NAME#"_F32_NODE") f32:$a, f32:$b))]>;
  def _F16 : SASSInst<(outs GPR32:$d), (ins GPR32:$a),
                      "F2FP.SATFINITE."#fmt#".F16.UNPACK_B_MERGE_C $d, $a, RZ",
                      [(set i32:$d, (!cast<SDNode>(NAME#"_F16_NODE") v2f16:$a))]>;
  def _UNPACK : SASSInst<(outs GPR32:$d), (ins GPR32:$a), "F2FP.F16."#fmt#".UNPACK_B $d, $a",
                         [(set v2f16:$d, (!cast<SDNode>(NAME#"_UNPACK_NODE") i32:$a))]>;
  def _UNPACK_H1 : SASSInst<(outs GPR32:$d), (ins GPR32:$a),
                            "F2FP.F16."#fmt#".UNPACK_B.H1 $d, $a",
                            [(set v2f16:$d, (!cast<SDNode>(NAME#"_UNPACK_NODE")
                                              (srl i32:$a, (i32 16))))]>;
  }
}
defm F2FP_E4M3 : F2FP8<"E4M3", "E4M3X2">;
defm F2FP_E5M2 : F2FP8<"E5M2", "E5M2X2">;

//===----------------------------------------------------------------------===//
// mbarrier (SYNCS) on a shared address [Ra + off]. On sm_90 the assembler
// spells the zero uniform register UR63 in these addresses.
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, LatMnem = LAT_LDS,
    SchedRW = [WriteLDS] in {
// arrive with count 1 (A1T0) or count register (ART0), or arrive + expect
// transaction bytes (A1TR); _X forms discard the returned state.
def SYNCS_ARRIVE : SASSVarInst<(outs G64:$d), (ins GPR32:$a, simm32:$o),
                               "SYNCS.ARRIVE.TRANS64.A1T0 $d, [$a+UR63+$o], RZ">;
def SYNCS_ARRIVE_X : SASSVarInst<(outs), (ins GPR32:$a, simm32:$o),
                                 "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [$a+UR63+$o], RZ">;
def SYNCS_ARRIVE_N : SASSVarInst<(outs G64:$d), (ins GPR32:$a, simm32:$o, GPR32:$n),
                                 "SYNCS.ARRIVE.TRANS64.ART0 $d, [$a+UR63+$o], $n">;
def SYNCS_ARRIVE_N_X : SASSVarInst<(outs), (ins GPR32:$a, simm32:$o, GPR32:$n),
                                   "SYNCS.ARRIVE.TRANS64.ART0 RZ, [$a+UR63+$o], $n">;
def SYNCS_ARRIVE_TX : SASSVarInst<(outs G64:$d), (ins GPR32:$a, simm32:$o, GPR32:$n),
                                  "SYNCS.ARRIVE.TRANS64 $d, [$a+UR63+$o], $n">;
def SYNCS_ARRIVE_TX_X : SASSVarInst<(outs), (ins GPR32:$a, simm32:$o, GPR32:$n),
                                    "SYNCS.ARRIVE.TRANS64 RZ, [$a+UR63+$o], $n">;
// Remote (shared::cluster) arrive: no state.
def SYNCS_ARRIVE_REMOTE : SASSVarInst<(outs), (ins GPR32:$a, simm32:$o),
                                      "SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [$a+UR63+$o], RZ">;
// expect_tx: add to the pending transaction bytes without arriving.
def SYNCS_EXPECT_TX : SASSVarInst<(outs), (ins GPR32:$a, simm32:$o, GPR32:$n),
                                  "SYNCS.ARRIVE.TRANS64.RED.A0TR RZ, [$a+UR63+$o], $n">;
// cp.async.mbarrier.arrive (counting form): one more pending transaction.
def SYNCS_EXPECT_ONE : SASSVarInst<(outs), (ins GPR32:$a, simm32:$o),
                                   "SYNCS.ARRIVE.TRANS64.RED.A0T1 RZ, [$a+UR63+$o], RZ">;
// Phase checks against bit 31 of $ph (the parity, or the state's high word).
def SYNCS_TRYWAIT : SASSVarInst<(outs PRED:$p), (ins GPR32:$a, simm32:$o, GPR32:$ph),
                                "SYNCS.PHASECHK.TRANS64.TRYWAIT $p, [$a+UR63+$o], $ph">;
def SYNCS_TESTWAIT : SASSVarInst<(outs PRED:$p), (ins GPR32:$a, simm32:$o, GPR32:$ph),
                                 "SYNCS.PHASECHK.TRANS64 $p, [$a+UR63+$o], $ph">;
def SYNCS_INVAL : SASSVarInst<(outs), (ins GPR32:$a, simm32:$o), "SYNCS.CCTL.IV [$a+UR63+$o]">;
}

//===----------------------------------------------------------------------===//
// Fences
//===----------------------------------------------------------------------===//

// A fence's completion is tracked by a write scoreboard on a dummy result
// that the following SBWAIT consumes (the hardware writes no register).
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isPredicable = 0, NoGuard = 1 in {
let LatMnem = LAT_LDS, SchedRW = [WriteLDS] in {
def FENCE_VIEW_ASYNC_S : SASSVarInst<(outs GPR32:$done), (ins), "FENCE.VIEW.ASYNC.S">;
def FENCE_VIEW_ASYNC_G : SASSVarInst<(outs GPR32:$done), (ins), "FENCE.VIEW.ASYNC.G">;
}
}
// MEMBAR_*, ERRBAR, CGAERRBAR and CCTL_IVALL are in SASSInstrTypes.td.
let LatMnem = LAT_NOP, hasSideEffects = 1, isPredicable = 0, NoGuard = 1 in
def SBWAIT : SASSInst<(outs), (ins GPR32:$done), "NOP">;

//===----------------------------------------------------------------------===//
// Async proxy and cluster instructions with uniform-register operands.
// The UNI_* pseudos take GPRs; SASSISelPTX.cpp expands each into a loop
// that elects one pending thread per iteration, moves its operands to
// uniform registers and issues the instruction once (as ptxas does).
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isPredicable = 0, NoGuard = 1,
    LatMnem = LAT_LDS, SchedRW = [WriteLDS] in {
// mbarrier.init; $after is the preceding FENCE.VIEW.ASYNC.S's dummy result.
def SYNCS_EXCH : SASSVarInst<(outs GPR32:$done), (ins UGPR32:$a, UG64:$v, GPR32:$after),
                             "SYNCS.EXCH.64 URZ, [$a], $v">;
def ARRIVES_ARVCNT : SASSVarInst<(outs), (ins UGPR32:$a), "ARRIVES.LDGSTSBAR.64.ARVCNT [RZ+$a]">;
def ARRIVES_TRANSCNT : SASSVarInst<(outs), (ins UGPR32:$a),
                                   "ARRIVES.LDGSTSBAR.64.TRANSCNT [RZ+$a]">;
// {dst, mbar}, src, size/16
def UBLKCP_SG : SASSVarInst<(outs), (ins UG64:$dm, UG64:$src, UGPR32:$n),
                            "UBLKCP.S.G $dm, $src, $n">;
// dst, src, size/16
def UBLKCP_GS : SASSVarInst<(outs), (ins UG64:$dst, UGPR32:$src, UGPR32:$n),
                            "UBLKCP.G.S $dst, $src, $n">;
def UBLKPF_L2 : SASSVarInst<(outs), (ins UG64:$src, UGPR32:$n), "UBLKPF.L2 $src, $n">;
// With an L2 cache policy (.L2::cache_hint) as the descriptor pair.
def UBLKCP_SG_D : SASSVarInst<(outs), (ins UG64:$dm, UG64:$src, UGPR32:$n, UG64:$pol),
                              "UBLKCP.S.G $dm, $src, $n, desc[$pol]">;
def UBLKCP_GS_D : SASSVarInst<(outs), (ins UG64:$dst, UGPR32:$src, UGPR32:$n, UG64:$pol),
                              "UBLKCP.G.S $dst, $src, $n, desc[$pol]">;
def UBLKPF_L2_D : SASSVarInst<(outs), (ins UG64:$src, UGPR32:$n, UG64:$pol),
                              "UBLKPF.L2 $src, $n, desc[$pol]">;
// Tensor copies: a group of 4 (2D) or 8 uniform registers {dst|src, mbar,
// coordinates...}, and the tensor map's address.
// The group is {smem, mbarrier (loads only), coordinates...} padded to 4 or
// 8 registers.
foreach D = [["1D", "4", "4"], ["2D", "4", "4"], ["3D", "8", "4"], ["4D", "8", "8"], ["5D", "8", "8"]] in {
defvar G4 = (ins UGPR32:$u0, UGPR32:$u1, UGPR32:$u2, UGPR32:$u3, UG64:$map);
defvar G8 = (ins UGPR32:$u0, UGPR32:$u1, UGPR32:$u2, UGPR32:$u3, UGPR32:$u4, UGPR32:$u5,
                 UGPR32:$u6, UGPR32:$u7, UG64:$map);
defvar S4 = "\\{$u0,$u1,$u2,$u3\\}";
defvar S8 = "\\{$u0,$u1,$u2,$u3,$u4,$u5,$u6,$u7\\}";
def UTMALDG_#D[0] : SASSVarInst<(outs), !if(!eq(D[1], "4"), G4, G8),
                                "UTMALDG."#D[0]#" "#!if(!eq(D[1], "4"), S4, S8)#", $map">;
def UTMASTG_#D[0] : SASSVarInst<(outs), !if(!eq(D[2], "4"), G4, G8),
                                "UTMASTG."#D[0]#" "#!if(!eq(D[2], "4"), S4, S8)#", $map">;
def UTMALDG_#D[0]#_D : SASSVarInst<(outs), !con(!if(!eq(D[1], "4"), G4, G8), (ins UG64:$pol)),
                                   "UTMALDG."#D[0]#" "#!if(!eq(D[1], "4"), S4, S8)#
                                   ", $map, desc[$pol]">;
def UTMASTG_#D[0]#_D : SASSVarInst<(outs), !con(!if(!eq(D[2], "4"), G4, G8), (ins UG64:$pol)),
                                   "UTMASTG."#D[0]#" "#!if(!eq(D[2], "4"), S4, S8)#
                                   ", $map, desc[$pol]">;
}
def UTMACCTL_PF : SASSVarInst<(outs), (ins UG64:$map), "UTMACCTL.PF $map">;
}
// Bulk groups: UTMACMDFLUSH commits on SB0 (SASSControlCodes), DEPBAR.LE
// SB0, N waits until at most N groups are pending.
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isPredicable = 0, NoGuard = 1, LatMnem = LAT_NOP in {
def UTMACMDFLUSH : SASSInst<(outs), (ins), "UTMACMDFLUSH">;
let MinStall = 4 in
def DEPBAR_LE_SB0 : SASSInst<(outs), (ins i32imm:$n), "DEPBAR.LE SB0, $n">;
}

let usesCustomInserter = 1, hasSideEffects = 1, mayLoad = 1, mayStore = 1,
    isPredicable = 0, NoGuard = 1 in {
// Hopper, operands warp-uniform (SDNode divergence): no waterfall loop.
def UNIU_SYNCS_EXCH : SASSInst<(outs), (ins GPR32:$a, GPR32:$lo, GPR32:$hi), "">;
def UNIU_UBLKCP_SG : SASSInst<(outs), (ins GPR32:$dst, GPR32:$mbar, G64:$src, GPR32:$n), "">;
def UNIU_ARRIVES_ARVCNT : SASSInst<(outs), (ins GPR32:$a), "">;
def UNIU_ARRIVES_TRANSCNT : SASSInst<(outs), (ins GPR32:$a), "">;
def UNI_SYNCS_EXCH : SASSInst<(outs), (ins GPR32:$a, GPR32:$lo, GPR32:$hi), "">;
def UNI_ARRIVES_ARVCNT : SASSInst<(outs), (ins GPR32:$a), "">;
def UNI_ARRIVES_TRANSCNT : SASSInst<(outs), (ins GPR32:$a), "">;
def UNI_UBLKCP_SG : SASSInst<(outs), (ins GPR32:$dst, GPR32:$mbar, G64:$src, GPR32:$n), "">;
def UNI_UBLKCP_GS : SASSInst<(outs), (ins G64:$dst, GPR32:$src, GPR32:$n), "">;
def UNI_UBLKPF_L2 : SASSInst<(outs), (ins G64:$src, GPR32:$n), "">;
def UNI_UTMACCTL_PF : SASSInst<(outs), (ins G64:$map), "">;
// Tensor copies: map, smem address, mbarrier (loads; RZ for stores), coords
// (RZ-padded to 5).
def UNI_UTMALDG : SASSInst<(outs), (ins i32imm:$dim, G64:$map, GPR32:$s, GPR32:$mbar,
                                        GPR32:$c0, GPR32:$c1, GPR32:$c2, GPR32:$c3, GPR32:$c4), "">;
def UNI_UTMASTG : SASSInst<(outs), (ins i32imm:$dim, G64:$map, GPR32:$s,
                                        GPR32:$c0, GPR32:$c1, GPR32:$c2, GPR32:$c3, GPR32:$c4), "">;
// _H: the same with an L2 cache policy, already in a uniform register pair
// (SASS::policyDesc), after the other operands.
def UNIU_UBLKCP_SG_H : SASSInst<(outs), (ins GPR32:$dst, GPR32:$mbar, G64:$src, GPR32:$n,
                                             UG64:$pol), "">;
def UNI_UBLKCP_SG_H : SASSInst<(outs), (ins GPR32:$dst, GPR32:$mbar, G64:$src, GPR32:$n,
                                            UG64:$pol), "">;
def UNI_UBLKCP_GS_H : SASSInst<(outs), (ins G64:$dst, GPR32:$src, GPR32:$n, UG64:$pol), "">;
def UNI_UBLKPF_L2_H : SASSInst<(outs), (ins G64:$src, GPR32:$n, UG64:$pol), "">;
def UNI_UTMALDG_H : SASSInst<(outs), (ins i32imm:$dim, G64:$map, GPR32:$s, GPR32:$mbar,
                                          GPR32:$c0, GPR32:$c1, GPR32:$c2, GPR32:$c3, GPR32:$c4,
                                          UG64:$pol), "">;
def UNI_UTMASTG_H : SASSInst<(outs), (ins i32imm:$dim, G64:$map, GPR32:$s,
                                          GPR32:$c0, GPR32:$c1, GPR32:$c2, GPR32:$c3, GPR32:$c4,
                                          UG64:$pol), "">;
}

// Loop pieces for the expansion.
let isPredicable = 0, NoGuard = 1, hasSideEffects = 1 in {
let LatMnem = LAT_ELECT in
def ELECT_P : SASSInst<(outs PRED:$lead), (ins PRED:$among), "ELECT $lead, URZ, $among">;
let LatMnem = LAT_BRA, SchedRW = [WriteBR], MinStall = 5, isBranch = 1, isTerminator = 1 in
def BRA_U_ANY : SASSInst<(outs), (ins brtarget:$t), "BRA.U.ANY $t">;
}
let LatMnem = LAT_R2UR in
def R2UR_G : SASSInst<(outs UGPR32:$d), (ins GPR32:$a), "R2UR PT, $d, $a">;

// elect.sync: the elected lane gets the predicate; every lane the leader's id.
let LatMnem = LAT_ELECT, isPredicable = 0, NoGuard = 1, hasSideEffects = 1 in
def ELECT : SASSInst<(outs PRED:$p, UGPR32:$lane), (ins), "ELECT $p, $lane, PT">;

//===----------------------------------------------------------------------===//
// Cluster barriers and shared::cluster accesses
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, isPredicable = 0, NoGuard = 1, isConvergent = 1, LatMnem = LAT_BAR, MinStall = 6 in {
def UCGABAR_ARV : SASSInst<(outs), (ins), "UCGABAR_ARV">;
def UCGABAR_WAIT : SASSInst<(outs), (ins), "UCGABAR_WAIT">;
}
// c[0x0][0x140] is 1 when the grid was launched with clusters.
let Predicates = [NoBlackwellISA] in
def : Pat<(int_nvvm_is_explicit_cluster), (ISETPri (LDC32 0x140), 1, 4, (i1 PT), 0)>;

// Shared::cluster addresses are 32-bit; with the shared window's high word
// they form a generic address: shared::cluster accesses are generic LD/ST.
def SASSswinhi : SDNode<"SASSISD::SWINHI", SDTypeProfile<1, 0, [SDTCisVT<0, i32>]>>;
let LatMnem = LAT_S2R, SchedRW = [WriteS2R], isReMaterializable = 1 in
def S2R_SWINHI : SASSVarInst<(outs GPR32:$d), (ins), "S2R $d, SR_SWINHI",
                             [(set i32:$d, (SASSswinhi))]>;
def ClusterGeneric : OutPatFrag<(ops node:$a),
                                (REG_SEQUENCE GPR64, $a, sub0, (S2R_SWINHI), sub1)>;
def : Pat<(i32 (SpaceLoad<load, 7> i32:$a)), (LD32 (ClusterGeneric $a), 0)>;
def : Pat<(f32 (SpaceLoad<load, 7> i32:$a)), (LD32 (ClusterGeneric $a), 0)>;
def : Pat<(i64 (SpaceLoad<load, 7> i32:$a)), (LD64 (ClusterGeneric $a), 0)>;
def : Pat<(SpaceStore<store, 7> i32:$v, i32:$a), (ST32 (ClusterGeneric $a), 0, $v)>;
def : Pat<(SpaceStore<store, 7> f32:$v, i32:$a), (ST32 (ClusterGeneric $a), 0, $v)>;
def : Pat<(SpaceStore<store, 7> i64:$v, i32:$a), (ST64 (ClusterGeneric $a), 0, $v)>;

// st.async to another CTA's shared memory, completing bytes on its
// mbarrier: STAS [{addr, mbar}].
let hasSideEffects = 1, mayStore = 1, LatMnem = LAT_STS, SchedRW = [WriteST] in {
def STAS32 : SASSVarInst<(outs), (ins G64:$am, GPR32:$v), "STAS [$am+UR63], $v">;
def STAS64 : SASSVarInst<(outs), (ins G64:$am, G64:$v), "STAS.64 [$am+UR63], $v">;
def STAS128 : SASSVarInst<(outs), (ins G64:$am, G128:$v), "STAS.128 [$am+UR63], $v">;
}
