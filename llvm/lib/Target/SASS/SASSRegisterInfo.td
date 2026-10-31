//===-- SASSRegisterInfo.td - SASS register file -----------*- tablegen -*-===//
//
// Vector registers R0-R254 (RZ reads as zero), even-aligned 64-bit pairs and
// 4-aligned 128-bit quads, predicates P0-P6 (PT reads as true), and uniform
// registers UR0-UR62 (URZ reads as zero).
//
//===----------------------------------------------------------------------===//

let Namespace = "SASS" in {
def sub0 : SubRegIndex<32>;
def sub1 : SubRegIndex<32, 32>;
def sub2 : SubRegIndex<32, 64>;
def sub3 : SubRegIndex<32, 96>;
}

class SASSReg<int Enc, string Name> : Register<Name> {
  let Namespace = "SASS";
  let HWEncoding{7-0} = Enc;
}

class SASSTuple<int Enc, string Name, list<Register> Subs,
                list<SubRegIndex> Idx> : RegisterWithSubRegs<Name, Subs> {
  let Namespace = "SASS";
  let HWEncoding{7-0} = Enc;
  let SubRegIndices = Idx;
  let CoveredBySubRegs = 1;
}

foreach I = 0-254 in
  def R#I : SASSReg<I, "R"#I>;
def RZ : SASSReg<255, "RZ">;

foreach I = 0-126 in
  def RD#I : SASSTuple<!mul(I, 2), "R"#!mul(I, 2)#"_64",
                       [!cast<Register>("R"#!mul(I, 2)),
                        !cast<Register>("R"#!add(!mul(I, 2), 1))],
                       [sub0, sub1]>;

foreach I = 0-62 in
  def RQ#I : SASSTuple<!mul(I, 4), "R"#!mul(I, 4)#"_128",
                       [!cast<Register>("R"#!mul(I, 4)),
                        !cast<Register>("R"#!add(!mul(I, 4), 1)),
                        !cast<Register>("R"#!add(!mul(I, 4), 2)),
                        !cast<Register>("R"#!add(!mul(I, 4), 3))],
                       [sub0, sub1, sub2, sub3]>;

foreach I = 0-6 in
  def P#I : SASSReg<I, "P"#I>;
def PT : SASSReg<7, "PT">;

foreach I = 0-62 in
  def UR#I : SASSReg<I, "UR"#I>;
def URZ : SASSReg<63, "URZ">;

foreach I = 0-30 in
  def URD#I : SASSTuple<!mul(I, 2), "UR"#!mul(I, 2)#"_64",
                        [!cast<Register>("UR"#!mul(I, 2)),
                         !cast<Register>("UR"#!add(!mul(I, 2), 1))],
                        [sub0, sub1]>;

// R1 (stack pointer by ABI) and RZ are reserved in SASSRegisterInfo; RZ stays
// in the class so patterns can name it as a zero source.
def GPR32 : RegisterClass<"SASS", [i32, f32, v2f16, v2bf16, f16, bf16], 32,
                          (add (sequence "R%u", 0, 254), RZ)>;
def GPR64 : RegisterClass<"SASS", [i64, f64], 64, (sequence "RD%u", 0, 126)>;
def GPR128 : RegisterClass<"SASS", [v4i32, v4f32], 128,
                           (sequence "RQ%u", 0, 62)>;
// Predicates spill as one byte (SASSRegisterInfo::eliminateFrameIndex).
def PRED : RegisterClass<"SASS", [i1], 8, (add (sequence "P%u", 0, 6), PT)> {
  let Size = 8;
}
def UGPR32 : RegisterClass<"SASS", [i32], 32,
                           (add (sequence "UR%u", 0, 62), URZ)>;
def UGPR64 : RegisterClass<"SASS", [i64], 64, (sequence "URD%u", 0, 30)>;

include "SASSRegisterWGMMA.td"
