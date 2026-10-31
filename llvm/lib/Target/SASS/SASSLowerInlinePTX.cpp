//===-- SASSLowerInlinePTX.cpp - Inline PTX to NVVM intrinsics -------------===//
//
// CUDA code reaches tensor-core and async-copy instructions through inline
// PTX (CUTLASS-style `asm volatile("mma.sync...")`). This pass rewrites the
// supported templates into the equivalent llvm.nvvm intrinsics, which the
// SASS instruction selector implements, so the same source builds with nvcc
// and with this backend. Any other inline asm is a fatal error naming it.
//
// Supported templates ($N operands; `.shared::cta` = `.shared`):
//   ldmatrix.sync.aligned.m8n8.{x1,x2,x4}[.trans].shared.b16 {d...}, [a]
//   stmatrix.sync.aligned.m8n8.{x1,x2,x4}[.trans].shared.b16 [a], {s...}
//   mma.sync.aligned.m16n8k16.row.col.f32.{f16.f16,bf16.bf16}.f32
//       {d0..d3}, {a0..a3}, {b0,b1}, {c0..c3}
//   cp.async.{ca,cg}.shared.global[.L2::{64,128,256}B] [s], [g], {4,8,16}
//   cp.async.commit_group;  cp.async.wait_group N;  cp.async.wait_all
//   ex2.approx.ftz.f32 d, a;  rcp.approx.ftz.f32 d, a
// Addresses may carry a constant offset ([$N+16]).
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include <regex>

using namespace llvm;

namespace {
class SASSLowerInlinePTX : public FunctionPass {
public:
  static char ID;
  SASSLowerInlinePTX() : FunctionPass(ID) {}
  StringRef getPassName() const override { return "SASS lower inline PTX"; }
  bool runOnFunction(Function &F) override;
};

/// One `asm` call: its operands by `$N` index.
class AsmCall {
  CallInst &CI;
  SmallVector<int, 8> OutOf;   // constraint index -> result element, or -1
  SmallVector<int, 8> ArgOf;   // constraint index -> call argument, or -1
  SmallVector<int, 8> TiedArg; // result element -> call argument tied to it
  unsigned NumOuts = 0;

public:
  explicit AsmCall(CallInst &CI) : CI(CI) {
    const auto *IA = cast<InlineAsm>(CI.getCalledOperand());
    unsigned Arg = 0;
    for (const InlineAsm::ConstraintInfo &C : IA->ParseConstraints()) {
      if (C.Type == InlineAsm::isClobber)
        continue;
      if (C.Type == InlineAsm::isOutput) {
        OutOf.push_back(NumOuts++);
        ArgOf.push_back(-1);
        TiedArg.push_back(-1);
        continue;
      }
      OutOf.push_back(-1);
      ArgOf.push_back(Arg);
      // A tied input ("+r" in the source) names its output by number;
      // outputs come first, so that is also the result element.
      if (!C.Codes.empty() && isDigit(C.Codes[0][0]))
        TiedArg[std::stoi(C.Codes[0])] = Arg;
      ++Arg;
    }
  }

  unsigned numOuts() const { return NumOuts; }

  /// The value read by operand $N (an input, or the input tied to an output).
  Value *src(unsigned N) const {
    if (N >= OutOf.size())
      report_fatal_error("SASS: inline PTX operand $" + Twine(N) + " out of range");
    if (ArgOf[N] >= 0)
      return CI.getArgOperand(ArgOf[N]);
    if (TiedArg[OutOf[N]] < 0)
      report_fatal_error("SASS: inline PTX reads output $" + Twine(N));
    return CI.getArgOperand(TiedArg[OutOf[N]]);
  }
  /// Result element written by operand $N.
  unsigned dst(unsigned N) const {
    if (N >= OutOf.size() || OutOf[N] < 0)
      report_fatal_error("SASS: inline PTX writes input $" + Twine(N));
    return OutOf[N];
  }
};

/// A parsed operand: register refs ($N, {$a,$b}), an address [$N+off] or an
/// integer literal.
struct Operand {
  SmallVector<unsigned, 4> Refs;
  int64_t Offset = 0;
  bool IsAddr = false;
  std::optional<int64_t> Imm;
};

std::optional<Operand> parseOperand(StringRef S) {
  Operand Op;
  S = S.trim();
  if (S.consume_front("[")) {
    if (!S.consume_back("]"))
      return std::nullopt;
    Op.IsAddr = true;
    auto [Base, Off] = S.split('+');
    if (!Off.empty() && Off.trim().getAsInteger(0, Op.Offset))
      return std::nullopt;
    S = Base.trim();
  } else if (S.consume_front("{")) {
    if (!S.consume_back("}"))
      return std::nullopt;
    SmallVector<StringRef, 4> Parts;
    S.split(Parts, ',');
    for (StringRef P : Parts) {
      unsigned N;
      P = P.trim();
      if (!P.consume_front("$") || P.getAsInteger(10, N))
        return std::nullopt;
      Op.Refs.push_back(N);
    }
    return Op;
  }
  unsigned N;
  int64_t V;
  if (S.consume_front("$") && !S.getAsInteger(10, N))
    Op.Refs.push_back(N);
  else if (!S.getAsInteger(0, V))
    Op.Imm = V;
  else
    return std::nullopt;
  return Op;
}

/// Splits `opcode a, {b, c}, [d]` into the opcode and top-level operands.
bool splitAsm(StringRef Asm, std::string &Opcode,
              SmallVectorImpl<Operand> &Ops) {
  std::string S = std::regex_replace(Asm.str(), std::regex("\\s+"), " ");
  StringRef R = StringRef(S).trim();
  R.consume_back(";");
  R = R.trim();
  if (R.contains(';') || R.contains('\n'))
    return false;
  auto [Op, Rest] = R.split(' ');
  Opcode = std::regex_replace(Op.str(), std::regex("\\.shared::cta"), ".shared");
  int Depth = 0;
  size_t Start = 0;
  for (size_t I = 0; I <= Rest.size(); ++I) {
    char C = I < Rest.size() ? Rest[I] : ',';
    Depth += (C == '{' || C == '[') - (C == '}' || C == ']');
    if (C == ',' && Depth == 0) {
      if (I > Start) {
        std::optional<Operand> O = parseOperand(Rest.slice(Start, I));
        if (!O)
          return false;
        Ops.push_back(*O);
      }
      Start = I + 1;
    }
  }
  return true;
}
} // namespace

char SASSLowerInlinePTX::ID = 0;

INITIALIZE_PASS(SASSLowerInlinePTX, "sass-lower-inline-ptx",
                "SASS lower inline PTX", false, false)

FunctionPass *llvm::createSASSLowerInlinePTXPass() {
  return new SASSLowerInlinePTX();
}

/// Integer register value as a pointer in address space \p AS, plus a byte
/// offset.
static Value *address(IRBuilder<> &B, Value *V, int64_t Off, unsigned AS) {
  if (V->getType()->isPointerTy())
    V = B.CreatePtrToInt(V, B.getIntNTy(AS == 3 ? 32 : 64));
  if (Off)
    V = B.CreateAdd(V, ConstantInt::get(V->getType(), Off));
  return B.CreateIntToPtr(V, B.getPtrTy(AS));
}

bool SASSLowerInlinePTX::runOnFunction(Function &F) {
  SmallVector<CallInst *, 16> Calls;
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I); CI && CI->isInlineAsm())
      Calls.push_back(CI);
  Module &M = *F.getParent();
  for (CallInst *CI : Calls) {
    StringRef Asm = cast<InlineAsm>(CI->getCalledOperand())->getAsmString();
    // Canonical wgmma.mma_async calls from SASSExpandInlinePTX are selected
    // as asm (they need accumulator tuples no intrinsic describes).
    if (Asm.starts_with("wgmma.mma_async"))
      continue;
    std::string Opc;
    SmallVector<Operand, 4> Ops;
    auto Fail = [&]() -> void {
      report_fatal_error("SASS: unsupported inline PTX in " + F.getName() +
                         ": \"" + Asm + "\"");
    };
    if (!splitAsm(Asm, Opc, Ops))
      Fail();
    AsmCall Call(*CI);
    IRBuilder<> B(CI);
    SmallVector<Value *, 4> Outs(Call.numOuts(), nullptr);
    auto Regs = [&](const Operand &O) {
      SmallVector<Value *, 4> V;
      for (unsigned N : O.Refs)
        V.push_back(Call.src(N));
      return V;
    };
    auto Define = [&](const Operand &O, ArrayRef<Value *> Vals) {
      if (O.Refs.size() != Vals.size())
        Fail();
      for (auto [N, V] : zip(O.Refs, Vals))
        Outs[Call.dst(N)] = V;
    };
    auto Results = [&](Value *R, unsigned N) {
      SmallVector<Value *, 4> V;
      if (N == 1)
        V.push_back(R);
      else
        for (unsigned I = 0; I < N; ++I)
          V.push_back(B.CreateExtractValue(R, I));
      return V;
    };
    std::smatch Mt;
    std::string Name;
    if (std::regex_match(Opc, Mt, std::regex(
            "(ld|st)matrix\\.sync\\.aligned\\.m8n8\\.x([124])(\\.trans)?\\.shared\\.b16"))) {
      bool Load = Mt[1] == "ld";
      unsigned N = std::stoi(Mt[2]);
      Name = std::string("llvm.nvvm.") + (Load ? "ld" : "st") +
             "matrix.sync.aligned.m8n8.x" + Mt[2].str() + (Mt[3].matched ? ".trans" : "") + ".b16";
      const Operand &Addr = Ops[Load ? 1 : 0], &Regs0 = Ops[Load ? 0 : 1];
      if (Ops.size() != 2 || !Addr.IsAddr || Regs0.Refs.size() != N)
        Fail();
      Value *P = address(B, Call.src(Addr.Refs[0]), Addr.Offset, 3);
      Intrinsic::ID IID = Intrinsic::lookupIntrinsicID(Name);
      Function *Fn = Intrinsic::getOrInsertDeclaration(&M, IID, {P->getType()});
      if (Load) {
        Define(Regs0, Results(B.CreateCall(Fn, {P}), N));
      } else {
        SmallVector<Value *, 5> Args = {P};
        append_range(Args, Regs(Regs0));
        B.CreateCall(Fn, Args);
      }
    } else if (std::regex_match(Opc, Mt, std::regex(
                   "mma\\.sync\\.aligned\\.m16n8k16\\.row\\.col\\.f32\\.(f16|bf16)\\.(f16|bf16)\\.f32"))) {
      bool F16 = Mt[1] == "f16";
      if (Mt[1] != Mt[2] || Ops.size() != 4 || Ops[0].Refs.size() != 4 ||
          Ops[1].Refs.size() != 4 || Ops[2].Refs.size() != 2 || Ops[3].Refs.size() != 4)
        Fail();
      SmallVector<Value *, 10> Args;
      Type *Frag = F16 ? static_cast<Type *>(FixedVectorType::get(B.getHalfTy(), 2))
                       : B.getInt32Ty();
      for (unsigned I : {1, 2})
        for (Value *V : Regs(Ops[I]))
          Args.push_back(B.CreateBitCast(V, Frag));
      append_range(Args, Regs(Ops[3]));
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, F16 ? Intrinsic::nvvm_mma_m16n8k16_row_col_f32_f32
                  : Intrinsic::nvvm_mma_m16n8k16_row_col_bf16);
      Define(Ops[0], Results(B.CreateCall(Fn, Args), 4));
    } else if (std::regex_match(Opc, Mt, std::regex(
                   "cp\\.async\\.(ca|cg)\\.shared\\.global(\\.L2::(64|128|256)B)?"))) {
      if (Ops.size() != 3 || !Ops[0].IsAddr || !Ops[1].IsAddr)
        Fail();
      int64_t Size = Ops[2].Imm ? *Ops[2].Imm : -1;
      if (!Ops[2].Imm && Ops[2].Refs.size() == 1)
        if (auto *C = dyn_cast<ConstantInt>(Call.src(Ops[2].Refs[0])))
          Size = C->getSExtValue();
      Intrinsic::ID IID =
          Mt[1] == "cg" ? (Size == 16 ? Intrinsic::nvvm_cp_async_cg_shared_global_16
                                      : Intrinsic::not_intrinsic)
          : Size == 16  ? Intrinsic::nvvm_cp_async_ca_shared_global_16
          : Size == 8   ? Intrinsic::nvvm_cp_async_ca_shared_global_8
          : Size == 4   ? Intrinsic::nvvm_cp_async_ca_shared_global_4
                        : Intrinsic::not_intrinsic;
      if (IID == Intrinsic::not_intrinsic)
        Fail();
      B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, IID),
                   {address(B, Call.src(Ops[0].Refs[0]), Ops[0].Offset, 3),
                    address(B, Call.src(Ops[1].Refs[0]), Ops[1].Offset, 1)});
    } else if (Opc == "cp.async.commit_group" && Ops.empty()) {
      B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_cp_async_commit_group));
    } else if (Opc == "cp.async.wait_all" && Ops.empty()) {
      B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_cp_async_wait_all));
    } else if (Opc == "cp.async.wait_group" && Ops.size() == 1) {
      int64_t N = -1;
      if (Ops[0].Imm)
        N = *Ops[0].Imm;
      else if (auto *C = dyn_cast<ConstantInt>(Call.src(Ops[0].Refs[0])))
        N = C->getSExtValue();
      if (N < 0)
        Fail();
      B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_cp_async_wait_group),
                   {B.getInt32(N)});
    } else if ((Opc == "ex2.approx.ftz.f32" || Opc == "rcp.approx.ftz.f32") &&
               Ops.size() == 2 && Ops[0].Refs.size() == 1 && Ops[1].Refs.size() == 1) {
      Function *Fn =
          Opc[0] == 'e'
              ? Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_ex2_approx_ftz,
                                                  {B.getFloatTy()})
              : Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_rcp_approx_ftz_f);
      Define(Ops[0], {B.CreateCall(Fn, {Call.src(Ops[1].Refs[0])})});
    } else {
      Fail();
    }
    // Rebuild the asm call's result from the operands it defined.
    if (!CI->getType()->isVoidTy()) {
      Value *R;
      if (Outs.size() == 1) {
        R = Outs[0];
      } else {
        R = PoisonValue::get(CI->getType());
        for (auto [I, V] : enumerate(Outs))
          R = B.CreateInsertValue(R, V, I);
      }
      if (is_contained(Outs, nullptr))
        Fail();
      CI->replaceAllUsesWith(R);
    }
    CI->eraseFromParent();
  }
  return !Calls.empty();
}
