//===-- SASSInstrGlobals.td - Constant bank loads ----------*- tablegen -*-===//
//
// addrspace(4) loads read __constant__ variables (bank 3) or by-value
// kernel parameters (bank 0); SASS::constBank picks the bank from the memory
// operand. Addresses are 32-bit bank offsets: c[bank][R+imm] or c[bank][imm].
//
//===----------------------------------------------------------------------===//

class ConstBankLoad<PatFrag base, int bank>
    : PatFrag<(ops node:$ptr), (base node:$ptr),
              "return isAddrSpace(N, 4) && "
              "SASS::constBank(*cast<MemSDNode>(N)) == " # bank # ";">;

multiclass ConstLoads<string sz, RegisterOperand RC> {
  let mayLoad = 1, LatMnem = LAT_LDC, SchedRW = [WriteLDC] in {
    def "" : SASSVarInst<(outs RC:$d), (ins simm32:$bank, mems:$a),
                         "LDC"#sz#" $d, c[$bank]$a">;
    def _I : SASSVarInst<(outs RC:$d), (ins simm32:$bank, simm32:$o),
                         "LDC"#sz#" $d, c[$bank][$o]">;
  }
}
defm LDCB32 : ConstLoads<"", Op32>;
defm LDCB64 : ConstLoads<".64", G64>;
defm LDCBU8 : ConstLoads<".U8", Op32>;
defm LDCBS8 : ConstLoads<".S8", Op32>;
defm LDCBU16 : ConstLoads<".U16", Op32>;
defm LDCBS16 : ConstLoads<".S16", Op32>;

multiclass ConstLoadPat<ValueType vt, PatFrag frag, string inst, int bank> {
  def : Pat<(vt (ConstBankLoad<frag, bank> ADDRs:$a)),
            (!cast<Instruction>(inst) bank, ADDRs:$a)>;
  // Above the ComplexPattern form, which would put the constant in a register.
  let AddedComplexity = 10 in
  def : Pat<(vt (ConstBankLoad<frag, bank> (i32 imm:$o))),
            (!cast<Instruction>(inst#"_I") bank, imm:$o)>;
}
foreach bank = [0, 3] in {
  defm : ConstLoadPat<i32, load, "LDCB32", bank>;
  defm : ConstLoadPat<f32, load, "LDCB32", bank>;
  defm : ConstLoadPat<i64, load, "LDCB64", bank>;
  defm : ConstLoadPat<i32, zextloadi8, "LDCBU8", bank>;
  defm : ConstLoadPat<i32, extloadi8, "LDCBU8", bank>;
  defm : ConstLoadPat<i32, sextloadi8, "LDCBS8", bank>;
  defm : ConstLoadPat<i32, zextloadi16, "LDCBU16", bank>;
  defm : ConstLoadPat<i32, extloadi16, "LDCBU16", bank>;
  defm : ConstLoadPat<i32, sextloadi16, "LDCBS16", bank>;
}
