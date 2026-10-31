//===-- SASSRegisterWGMMA.td - Registers for warpgroup MMA -*- tablegen -*-===//
//
// Hopper warpgroup MMA (HGMMA/QGMMA, SASSWGMMA.cpp) operands:
//
// * Accumulators: 4-aligned contiguous GPR tuples of 8..64 registers in
//   steps of 4 (GACC<n>; 4 is GPR128, 2 is GPR64). m64nNk* has N/2 f32 or
//   N/4 f16 accumulator registers per thread; f16 counts that are not a
//   multiple of 4 use the next larger tuple and leave its top lanes unused.
//   LaneBitmask has 64 bits, so no tuple can have more than 64 32-bit
//   lanes: f32 shapes with N > 128 are issued as two instructions
//   (SASSWGMMA.cpp).
// * Descriptors: a 4-aligned uniform quad (A and B descriptors, SS form) or
//   a 4-aligned uniform pair (B descriptor, RS form).
// * Uniform predicates UP0-UP6 (UPT reads as true): runtime scale-d and the
//   USETMAXREG.TRY_ALLOC result.
//
//===----------------------------------------------------------------------===//

let Namespace = "SASS" in {
foreach I = 4-63 in
  def sub#I : SubRegIndex<32, !mul(I, 32)>;
}

foreach S = !range(8, 65, 4) in {
  foreach B = !range(0, !sub(256, S), 4) in
    def RA#S#_#B : SASSTuple<B, "R"#B#"_x"#S,
                             !foreach(I, !range(S), !cast<Register>("R"#!add(B, I))),
                             !foreach(I, !range(S), !cast<SubRegIndex>("sub"#I))>;
  def GACC#S : RegisterClass<"SASS", [untyped], 128,
                             (sequence "RA"#S#"_%u", 0, !sub(255, S), 4)> {
    let Size = !mul(S, 32);
  }
}

foreach I = 0-14 in
  def URQ#I : SASSTuple<!mul(I, 4), "UR"#!mul(I, 4)#"_128",
                        [!cast<Register>("UR"#!mul(I, 4)),
                         !cast<Register>("UR"#!add(!mul(I, 4), 1)),
                         !cast<Register>("UR"#!add(!mul(I, 4), 2)),
                         !cast<Register>("UR"#!add(!mul(I, 4), 3))],
                        [sub0, sub1, sub2, sub3]>;
def UGPR128 : RegisterClass<"SASS", [untyped], 128, (sequence "URQ%u", 0, 14)> {
  let Size = 128;
}
// 4-aligned uniform pairs (RS-form B descriptor).
def UGPR64A4 : RegisterClass<"SASS", [i64], 64,
                             (decimate (sequence "URD%u", 0, 30), 2)>;

foreach I = 0-6 in
  def UP#I : SASSReg<I, "UP"#I>;
def UPT : SASSReg<7, "UPT">;
def UPRED : RegisterClass<"SASS", [i1], 8, (sequence "UP%u", 0, 6)> {
  let Size = 8;
}
// Operand class: also accepts UPT (never allocated: no vreg has this class).
def UPREDT : RegisterClass<"SASS", [i1], 8, (add UPRED, UPT)> {
  let Size = 8;
}
