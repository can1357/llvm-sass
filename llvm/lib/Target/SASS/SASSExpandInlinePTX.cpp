//===-- SASSExpandInlinePTX.cpp - Inline PTX programs to IR ---------------===//
//
// CUDA kernels reach PTX-only features through inline asm whose templates
// are small PTX programs: scoped `.reg` declarations, predicates, guarded
// instructions, labels and branches around one or more instructions. This
// pass parses every such template and rewrites it as IR:
//
// * Registers (asm operands $N and `.reg` locals) become allocas, promoted
//   to SSA values at the end; labels and branches become basic blocks.
// * Instructions with an IR or NVVM-intrinsic equivalent (integer/float
//   arithmetic, setp/selp/cvt/prmt/lop3, ld/st with state spaces, orderings
//   and vectors, atom/red, fences and barriers, mbarrier, bulk and tensor
//   copies, cluster barriers, special registers, ...) are emitted directly.
//   Cache hints (.L1::*, .L2::*, createpolicy) are performance hints and are
//   dropped.
// * The instructions SASSLowerInlinePTX implements (ldmatrix, stmatrix,
//   mma.sync.m16n8k16, cp.async, ex2/rcp.approx.ftz) are re-emitted as
//   single-instruction asm calls for that pass. mma.sync.m16n8k32 on FP8
//   becomes two m16n8k16 f16 MMAs on converted fragments, as ptxas does on
//   sm_90, which has no FP8 mma.sync.
//
// A template that is one unguarded instruction this pass does not know is
// left alone for SASSLowerInlinePTX; in any other template an unknown
// instruction is a fatal error naming it.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

using namespace llvm;

namespace {

class SASSExpandInlinePTX : public FunctionPass {
public:
  static char ID;
  SASSExpandInlinePTX() : FunctionPass(ID) {}
  StringRef getPassName() const override { return "SASS expand inline PTX"; }
  bool runOnFunction(Function &F) override;
};

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

/// One operand of a PTX instruction.
struct Opnd {
  enum Kind { Reg, Imm, FImm, Vec, Addr, Sink } K = Reg;
  std::string Name;   // Reg: "$N", a local or a special register ("%tid.x").
  std::string Name2;  // Second destination of `p|q`.
  bool Neg = false;   // `!p`
  int64_t Int = 0;    // Imm; Addr offset.
  double FP = 0;      // FImm given in decimal.
  uint64_t Bits = 0;  // FImm given as 0f/0d bits.
  bool HasBits = false, Is64 = false;
  std::vector<Opnd> Elts; // Vec elements; Addr: [base] or [base, {coords}].
};

/// A register operand named \p Name.
Opnd regOp(StringRef Name) {
  Opnd O;
  O.Name = Name.str();
  return O;
}

struct Stmt {
  bool Guarded = false, GuardNeg = false;
  std::string Guard;
  std::string Opc, Root;          // "ld.global.v4.u32", "ld"
  SmallVector<std::string, 8> Mods; // "global", "v4", "u32"
  SmallVector<Opnd, 8> Ops;
  std::string Text;
  bool has(StringRef M) const { return is_contained(Mods, M); }
};

/// A label, a `{`/`}` scope edge, a `.reg` declaration or an instruction.
struct Item {
  enum Kind { Label, Open, Close, Decl, Inst } K;
  std::string Name;               // Label name; Decl type (".pred").
  SmallVector<std::string, 4> Names; // Decl names
  Stmt S;
};

bool isIdentChar(char C) {
  return isAlnum(C) || C == '_' || C == '$' || C == '%' || C == '.';
}

std::optional<Opnd> parseOperand(StringRef S);

/// Splits at top-level commas (outside [] and {}).
SmallVector<StringRef, 8> splitTop(StringRef S) {
  SmallVector<StringRef, 8> Out;
  int Depth = 0;
  size_t Start = 0;
  for (size_t I = 0; I <= S.size(); ++I) {
    char C = I < S.size() ? S[I] : ',';
    if (C == '[' || C == '{')
      ++Depth;
    else if (C == ']' || C == '}')
      --Depth;
    else if (C == ',' && Depth == 0) {
      StringRef P = S.slice(Start, I).trim();
      if (!P.empty() || I < S.size())
        Out.push_back(P);
      Start = I + 1;
    }
  }
  return Out;
}

std::optional<Opnd> parseScalar(StringRef S) {
  Opnd O;
  S = S.trim();
  if (S.empty())
    return std::nullopt;
  if (S == "_") {
    O.K = Opnd::Sink;
    return O;
  }
  if (S.consume_front("!")) {
    O.Neg = true;
    S = S.trim();
  }
  // Hex float bits: 0f3F800000, 0d3FF0000000000000.
  if (S.size() > 2 && S[0] == '0' && (S[1] == 'f' || S[1] == 'F' || S[1] == 'd' || S[1] == 'D') &&
      all_of(S.drop_front(2), isHexDigit)) {
    O.K = Opnd::FImm;
    O.HasBits = true;
    O.Is64 = S[1] == 'd' || S[1] == 'D';
    if (S.drop_front(2).getAsInteger(16, O.Bits))
      return std::nullopt;
    return O;
  }
  if (isDigit(S[0]) || S[0] == '-' || S[0] == '+') {
    StringRef N = S;
    bool Minus = N.consume_front("-");
    N.consume_front("+");
    N.consume_back("U");
    uint64_t V;
    if (!N.getAsInteger(0, V)) {
      O.K = Opnd::Imm;
      O.Int = Minus ? -(int64_t)V : (int64_t)V;
      return O;
    }
    double D;
    if (!S.getAsDouble(D)) {
      O.K = Opnd::FImm;
      O.FP = D;
      return O;
    }
    return std::nullopt;
  }
  if (!all_of(S, isIdentChar))
    return std::nullopt;
  O.K = Opnd::Reg;
  O.Name = S.str();
  return O;
}

std::optional<Opnd> parseOperand(StringRef S) {
  S = S.trim();
  if (S.consume_front("[")) {
    if (!S.consume_back("]"))
      return std::nullopt;
    Opnd O;
    O.K = Opnd::Addr;
    SmallVector<StringRef, 8> Parts = splitTop(S);
    if (Parts.empty())
      return std::nullopt;
    // base[+-off]
    StringRef Base = Parts[0];
    size_t Plus = Base.find_last_of("+-");
    if (Plus != StringRef::npos && Plus > 0) {
      StringRef Off = Base.substr(Plus).trim();
      bool Minus = Off.consume_front("-");
      Off.consume_front("+");
      uint64_t V;
      if (Off.trim().getAsInteger(0, V))
        return std::nullopt;
      O.Int = Minus ? -(int64_t)V : (int64_t)V;
      Base = Base.substr(0, Plus).trim();
    }
    std::optional<Opnd> B = parseScalar(Base);
    if (!B)
      return std::nullopt;
    O.Elts.push_back(*B);
    for (StringRef P : drop_begin(Parts)) {
      std::optional<Opnd> E = parseOperand(P);
      if (!E)
        return std::nullopt;
      O.Elts.push_back(*E);
    }
    return O;
  }
  if (S.consume_front("{")) {
    if (!S.consume_back("}"))
      return std::nullopt;
    Opnd O;
    O.K = Opnd::Vec;
    for (StringRef P : splitTop(S)) {
      std::optional<Opnd> E = parseScalar(P);
      if (!E)
        return std::nullopt;
      O.Elts.push_back(*E);
    }
    return O;
  }
  // p|q (setp), d|p (elect.sync)
  auto [A, Bar] = S.split('|');
  std::optional<Opnd> O = parseScalar(A);
  if (O && !Bar.empty()) {
    O->Name2 = Bar.trim().str();
    if (O->Name2.empty())
      return std::nullopt;
  }
  return O;
}

/// Parses an asm template (after `${:uid}`/`$$` substitution) into items.
/// Returns an error message on failure.
std::string parseProgram(StringRef Src, std::vector<Item> &Items) {
  // Strip comments.
  std::string T;
  for (size_t I = 0; I < Src.size(); ++I) {
    if (Src.substr(I).starts_with("//")) {
      while (I < Src.size() && Src[I] != '\n')
        ++I;
    } else if (Src.substr(I).starts_with("/*")) {
      size_t E = Src.find("*/", I + 2);
      I = E == StringRef::npos ? Src.size() : E + 1;
    } else {
      T += Src[I];
    }
  }
  StringRef S(T);
  while (true) {
    S = S.ltrim(" \t\r\n;");
    if (S.empty())
      return "";
    if (S.consume_front("{")) {
      Items.push_back({Item::Open, "", {}, {}});
      continue;
    }
    if (S.consume_front("}")) {
      Items.push_back({Item::Close, "", {}, {}});
      continue;
    }
    // Label: identifier followed by ':' (not '::').
    size_t N = 0;
    while (N < S.size() && (isAlnum(S[N]) || S[N] == '_' || S[N] == '$' || S[N] == '%'))
      ++N;
    if (N > 0) {
      size_t C = N;
      while (C < S.size() && (S[C] == ' ' || S[C] == '\t'))
        ++C;
      if (C < S.size() && S[C] == ':' && (C + 1 >= S.size() || S[C + 1] != ':')) {
        Items.push_back({Item::Label, S.substr(0, N).str(), {}, {}});
        S = S.drop_front(C + 1);
        continue;
      }
    }
    // A statement runs to ';' outside brackets.
    size_t End = 0;
    int Depth = 0;
    for (; End < S.size(); ++End) {
      char C = S[End];
      if (C == '[' || C == '{')
        ++Depth;
      else if (C == ']' || C == '}') {
        if (Depth == 0)
          break; // closing scope brace without ';'
        --Depth;
      } else if (C == ';' && Depth == 0)
        break;
    }
    StringRef Text = S.substr(0, End).trim();
    S = S.drop_front(End);
    Item It{Item::Inst, "", {}, {}};
    Stmt &St = It.S;
    St.Text = Text.str();
    StringRef R = Text;
    if (R.consume_front("@")) {
      St.Guarded = true;
      R = R.ltrim();
      if (R.consume_front("!"))
        St.GuardNeg = true;
      size_t G = 0;
      while (G < R.size() && isIdentChar(R[G]))
        ++G;
      St.Guard = R.substr(0, G).str();
      R = R.drop_front(G).ltrim();
    }
    size_t Sp = R.find_first_of(" \t\r\n");
    StringRef Opc = R.substr(0, Sp);
    StringRef Rest = Sp == StringRef::npos ? "" : R.substr(Sp).trim();
    if (Opc.starts_with(".")) {
      if (Opc == ".reg") {
        // .reg .type name[, name...]  (name<N> declares name0..nameN-1)
        auto [Ty, Names] = Rest.split(' ');
        It.K = Item::Decl;
        It.Name = Ty.trim().str();
        while (Names.trim().starts_with(".")) { // .reg .v2 .b32 x (vectors)
          return "vector .reg declarations are not supported";
        }
        for (StringRef Nm : splitTop(Names)) {
          Nm = Nm.trim();
          size_t Lt = Nm.find('<');
          if (Lt != StringRef::npos && Nm.ends_with(">")) {
            unsigned Cnt;
            if (Nm.slice(Lt + 1, Nm.size() - 1).getAsInteger(10, Cnt))
              return "bad register declaration: " + Text.str();
            for (unsigned I = 0; I < Cnt; ++I)
              It.Names.push_back((Nm.substr(0, Lt) + Twine(I)).str());
          } else {
            It.Names.push_back(Nm.str());
          }
        }
        Items.push_back(It);
        continue;
      }
      if (Opc == ".pragma" || Opc == ".loc")
        continue;
      return "unsupported directive '" + Opc.str() + "'";
    }
    std::string O = Opc.str();
    // .shared::cta is .shared; keep .shared::cluster.
    for (size_t P; (P = O.find(".shared::cta")) != std::string::npos;)
      O.replace(P, 12, ".shared");
    St.Opc = O;
    SmallVector<StringRef, 8> Parts;
    StringRef(St.Opc).split(Parts, '.');
    St.Root = Parts[0].str();
    for (StringRef P : drop_begin(Parts))
      St.Mods.push_back(P.str());
    if (!Rest.empty())
      for (StringRef P : splitTop(Rest)) {
        std::optional<Opnd> Op = parseOperand(P);
        if (!Op)
          return "cannot parse operand '" + P.str() + "' of '" + Text.str() + "'";
        St.Ops.push_back(*Op);
      }
    Items.push_back(It);
  }
}

//===----------------------------------------------------------------------===//
// Code generation
//===----------------------------------------------------------------------===//

/// PTX type of an instruction or register.
struct PTy {
  char Kind = 'b'; // b u s f (float) p (pred) h (half) B (bf16) 8 (fp8)
  unsigned Bits = 32; // Total width.
  unsigned Lanes = 1; // f16x2 / bf16x2: 2
  bool isSigned() const { return Kind == 's'; }
  bool isFloat() const { return Kind == 'f' || Kind == 'h' || Kind == 'B'; }
};

std::optional<PTy> parseType(StringRef M) {
  static const StringMap<PTy> Types = {
      {"b8", {'b', 8}},       {"b16", {'b', 16}},      {"b32", {'b', 32}},
      {"b64", {'b', 64}},     {"b128", {'b', 128}},    {"u8", {'u', 8}},
      {"u16", {'u', 16}},     {"u32", {'u', 32}},      {"u64", {'u', 64}},
      {"s8", {'s', 8}},       {"s16", {'s', 16}},      {"s32", {'s', 32}},
      {"s64", {'s', 64}},     {"f16", {'h', 16}},      {"f16x2", {'h', 32, 2}},
      {"bf16", {'B', 16}},    {"bf16x2", {'B', 32, 2}}, {"f32", {'f', 32}},
      {"f64", {'f', 64}},     {"pred", {'p', 1}},      {"e4m3x2", {'8', 16, 2}},
      {"e5m2x2", {'8', 16, 2}}, {"tf32", {'b', 32}}};
  auto It = Types.find(M);
  if (It == Types.end())
    return std::nullopt;
  return It->second;
}

class PTXProgram {
  CallInst &CI;
  Function &F;
  Module &M;
  LLVMContext &Ctx;
  const DataLayout &DL;
  IRBuilder<> B;
  SmallVectorImpl<AllocaInst *> &Slots;
  std::string Asm;

  struct Var {
    AllocaInst *Slot = nullptr;
    IntegerType *Ty = nullptr;
    std::optional<int64_t> Const; // Known value of an unwritten immediate input.
  };
  SmallVector<Var, 16> Ops;          // $N
  SmallVector<Type *, 16> OpTypes;   // IR type of $N
  SmallVector<int, 16> OutIndex;     // $N -> result element or -1
  std::vector<StringMap<Var>> Scopes;
  StringMap<BasicBlock *> Labels;
  BasicBlock *Tail = nullptr;
  Value *Guard = nullptr; // Guard of the pure instruction being emitted.
  const Stmt *Cur = nullptr;

public:
  PTXProgram(CallInst &CI, SmallVectorImpl<AllocaInst *> &Slots)
      : CI(CI), F(*CI.getFunction()), M(*F.getParent()), Ctx(F.getContext()),
        DL(M.getDataLayout()), B(&CI), Slots(Slots),
        Asm(cast<InlineAsm>(CI.getCalledOperand())->getAsmString()) {}

  void run(std::vector<Item> &Items);

  [[noreturn]] void fail(const Twine &Msg) const {
    report_fatal_error("SASS: inline PTX in " + F.getName() + ": " + Msg +
                       (Cur ? "\n  instruction: " + Cur->Text : std::string()) +
                       "\n  asm: \"" + Asm + "\"");
  }
  [[noreturn]] void unsupported(const Stmt &S) const {
    fail("unsupported PTX instruction '" + S.Opc + "'");
  }

private:
  // ---- registers ----
  AllocaInst *newSlot(Type *Ty, const Twine &Name) {
    IRBuilder<> EB(&*F.getEntryBlock().getFirstInsertionPt());
    AllocaInst *A = EB.CreateAlloca(Ty, nullptr, Name);
    Slots.push_back(A);
    return A;
  }
  Var *lookup(StringRef Name) {
    if (Name.starts_with("$")) {
      unsigned N;
      if (Name.drop_front().getAsInteger(10, N) || N >= Ops.size())
        fail("operand " + Name + " out of range");
      return &Ops[N];
    }
    for (auto &Sc : reverse(Scopes))
      if (auto It = Sc.find(Name); It != Sc.end())
        return &It->second;
    return nullptr;
  }
  IntegerType *intTy(unsigned Bits) { return IntegerType::get(Ctx, Bits); }
  /// \p V resized to \p Bits (truncated or extended per \p Signed).
  Value *resize(Value *V, unsigned Bits, bool Signed) {
    unsigned Have = V->getType()->getIntegerBitWidth();
    if (Have == Bits)
      return V;
    if (Have > Bits)
      return B.CreateTrunc(V, intTy(Bits));
    return Signed ? B.CreateSExt(V, intTy(Bits)) : B.CreateZExt(V, intTy(Bits));
  }
  /// Value of special register \p Name ("%tid.x") as an integer.
  Value *specialReg(StringRef Name) {
    // %clock64 and %globaltimer are 64-bit, the rest 32.
    Intrinsic::ID IID = nvvm("read_ptx_sreg_" + Name.drop_front().str());
    if (IID == Intrinsic::not_intrinsic)
      fail("unsupported special register " + Name);
    return call(IID, {});
  }
  /// Integer bits of operand \p O as \p Bits wide. Floats in decimal are
  /// converted per \p T.
  Value *read(const Opnd &O, unsigned Bits, PTy T = {}) {
    switch (O.K) {
    case Opnd::Imm:
      if (T.isFloat() && T.Lanes == 1) {
        Opnd F = O;
        F.K = Opnd::FImm;
        F.FP = (double)O.Int;
        return read(F, Bits, T);
      }
      return ConstantInt::get(intTy(Bits), O.Int, /*IsSigned=*/true);
    case Opnd::FImm: {
      if (O.HasBits) {
        if (O.Is64 && Bits == 32) {
          APFloat D(APFloat::IEEEdouble(), APInt(64, O.Bits));
          bool Lost;
          D.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &Lost);
          return ConstantInt::get(intTy(32), D.bitcastToAPInt().getZExtValue());
        }
        if (!O.Is64 && Bits == 64) {
          APFloat D(APFloat::IEEEsingle(), APInt(32, O.Bits));
          bool Lost;
          D.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &Lost);
          return ConstantInt::get(intTy(64), D.bitcastToAPInt().getZExtValue());
        }
        return ConstantInt::get(intTy(Bits), O.Bits);
      }
      APFloat D(O.FP);
      bool Lost;
      const fltSemantics &Sem = Bits == 64   ? APFloat::IEEEdouble()
                                : Bits == 32 ? APFloat::IEEEsingle()
                                : T.Kind == 'B' ? APFloat::BFloat()
                                                : APFloat::IEEEhalf();
      D.convert(Sem, APFloat::rmNearestTiesToEven, &Lost);
      return ConstantInt::get(intTy(Bits), D.bitcastToAPInt().getZExtValue());
    }
    case Opnd::Reg: {
      Value *V;
      if (Var *Vr = lookup(O.Name)) {
        V = B.CreateLoad(Vr->Ty, Vr->Slot);
      } else if (O.Name[0] == '%') {
        V = specialReg(O.Name);
      } else {
        fail("unknown register '" + O.Name + "'");
      }
      if (O.Neg) {
        if (V->getType()->isIntegerTy(1))
          V = B.CreateNot(V);
        else
          fail("'!' on a non-predicate operand");
      }
      return resize(V, Bits, T.isSigned());
    }
    case Opnd::Vec: {
      // {a, b}: elements packed low to high.
      unsigned EB = Bits / O.Elts.size();
      Value *R = ConstantInt::get(intTy(Bits), 0);
      for (auto [I, E] : enumerate(O.Elts)) {
        if (E.K == Opnd::Sink)
          continue;
        Value *Ev = B.CreateZExt(read(E, EB), intTy(Bits));
        R = B.CreateOr(R, B.CreateShl(Ev, I * EB));
      }
      return R;
    }
    default:
      fail("unexpected operand kind");
    }
  }
  std::optional<int64_t> constant(const Opnd &O) {
    if (O.K == Opnd::Imm)
      return O.Int;
    if (O.K == Opnd::Reg)
      if (Var *V = lookup(O.Name))
        return V->Const;
    return std::nullopt;
  }
  int64_t immediate(const Opnd &O, StringRef What) {
    if (std::optional<int64_t> C = constant(O))
      return *C;
    fail(What + " must be a constant");
  }
  /// Stores integer bits \p V to destination \p O (sign-extending per \p T).
  void write(const Opnd &O, Value *V, PTy T = {}) {
    if (O.K == Opnd::Sink)
      return;
    if (O.K == Opnd::Vec) {
      unsigned EB = V->getType()->getIntegerBitWidth() / O.Elts.size();
      for (auto [I, E] : enumerate(O.Elts))
        write(E, B.CreateTrunc(B.CreateLShr(V, I * EB), intTy(EB)));
      return;
    }
    if (O.K != Opnd::Reg)
      fail("destination is not a register");
    Var *Vr = lookup(O.Name);
    if (!Vr)
      fail("unknown destination register '" + O.Name + "'");
    V = resize(V, Vr->Ty->getBitWidth(), T.isSigned());
    if (Guard)
      V = B.CreateSelect(Guard, V, B.CreateLoad(Vr->Ty, Vr->Slot));
    B.CreateStore(V, Vr->Slot);
    Vr->Const.reset();
  }
  Value *readF(const Opnd &O, Type *Ty, PTy T) {
    return B.CreateBitCast(read(O, DL.getTypeSizeInBits(Ty), T), Ty);
  }
  void writeF(const Opnd &O, Value *V) {
    write(O, B.CreateBitCast(V, intTy(DL.getTypeSizeInBits(V->getType()))));
  }
  Type *floatTy(PTy T) {
    switch (T.Kind) {
    case 'h':
      return T.Lanes == 2 ? (Type *)FixedVectorType::get(B.getHalfTy(), 2) : B.getHalfTy();
    case 'B':
      return T.Lanes == 2 ? (Type *)FixedVectorType::get(B.getBFloatTy(), 2) : B.getBFloatTy();
    default:
      return T.Bits == 64 ? B.getDoubleTy() : B.getFloatTy();
    }
  }

  // ---- control flow ----
  BasicBlock *label(StringRef Name) {
    BasicBlock *&BB = Labels[Name];
    if (!BB)
      BB = BasicBlock::Create(Ctx, "ptx." + Name, &F, Tail);
    return BB;
  }
  BasicBlock *newBlock(const Twine &Name) {
    return BasicBlock::Create(Ctx, Name, &F, Tail);
  }

  // ---- helpers ----
  Value *call(Intrinsic::ID IID, ArrayRef<Value *> Args, ArrayRef<Type *> Tys = {}) {
    return B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, IID, Tys), Args);
  }
  /// Pointer in address space \p AS for address operand \p O ([r+off]).
  Value *address(const Opnd &O, unsigned AS) {
    if (O.K != Opnd::Addr)
      fail("expected an address operand");
    const Opnd &Base = O.Elts[0];
    unsigned PtrBits = DL.getPointerSizeInBits(AS);
    Value *P = nullptr;
    if (Base.K == Opnd::Reg) {
      Value *V = read(Base, lookupBits(Base));
      P = asPointer(V, AS, PtrBits);
    } else if (Base.K == Opnd::Imm) {
      P = B.CreateIntToPtr(ConstantInt::get(intTy(PtrBits), Base.Int), B.getPtrTy(AS));
    } else {
      fail("bad address operand");
    }
    if (O.Int)
      P = B.CreateConstGEP1_64(B.getInt8Ty(), P, O.Int);
    return P;
  }
  unsigned lookupBits(const Opnd &O) {
    if (Var *V = lookup(O.Name))
      return V->Ty->getBitWidth();
    return 64;
  }
  /// Turns integer address \p V into a pointer; looks through the
  /// ptrtoint (and trunc/zext) clang emits for "l"(ptr) and
  /// __cvta_generic_to_shared so alias analysis keeps the provenance.
  Value *asPointer(Value *V, unsigned AS, unsigned PtrBits) {
    Value *S = V;
    while (true) {
      if (auto *L = dyn_cast<LoadInst>(S)) {
        // A slot written once with the incoming operand.
        Value *Init = nullptr;
        for (User *U : L->getPointerOperand()->users())
          if (auto *St = dyn_cast<StoreInst>(U)) {
            if (Init) {
              Init = nullptr;
              break;
            }
            Init = St->getValueOperand();
          }
        if (!Init)
          break;
        S = Init;
        continue;
      }
      if (auto *C = dyn_cast<CastInst>(S);
          C && (isa<TruncInst>(C) || isa<ZExtInst>(C) || isa<PtrToIntInst>(C))) {
        if (isa<PtrToIntInst>(C)) {
          S = C->getOperand(0);
          break;
        }
        S = C->getOperand(0);
        continue;
      }
      break;
    }
    if (S->getType()->isPointerTy()) {
      unsigned SAS = S->getType()->getPointerAddressSpace();
      if (SAS == AS)
        return S;
      if (SAS == 0 && (AS == 1 || AS == 3 || AS == 5))
        return B.CreateAddrSpaceCast(S, B.getPtrTy(AS));
      if (AS == 0 && (SAS == 1 || SAS == 3))
        return B.CreateAddrSpaceCast(S, B.getPtrTy(AS));
    }
    return B.CreateIntToPtr(resize(V, PtrBits, false), B.getPtrTy(AS));
  }
  /// State space of a memory instruction: IR address space.
  unsigned space(const Stmt &S) {
    if (S.has("global"))
      return 1;
    if (S.has("shared"))
      return 3;
    if (S.has("shared::cluster"))
      return 7;
    if (S.has("local"))
      return 5;
    if (S.has("const"))
      return 4;
    if (S.has("param"))
      fail("ld/st.param is not supported");
    return 0;
  }
  /// Instruction type: the last type modifier.
  PTy type(const Stmt &S) {
    for (StringRef Mo : reverse(S.Mods))
      if (std::optional<PTy> T = parseType(Mo))
        return *T;
    fail("missing type in '" + S.Opc + "'");
  }
  SyncScope::ID scope(const Stmt &S, StringRef Default = "gpu") {
    StringRef Sc = Default;
    for (StringRef Mo : S.Mods)
      if (Mo == "cta" || Mo == "cluster" || Mo == "gpu" || Mo == "sys")
        Sc = Mo;
    if (Sc == "sys")
      return SyncScope::System;
    return Ctx.getOrInsertSyncScopeID(Sc == "cta" ? "block" : Sc == "cluster" ? "cluster" : "device");
  }
  void need(const Stmt &S, unsigned N) {
    if (S.Ops.size() != N)
      fail("'" + S.Opc + "' expects " + Twine(N) + " operands");
  }
  /// NVVM intrinsic by record suffix: "mbarrier_arrive_scope_cta_space_cta"
  /// is llvm.nvvm.mbarrier.arrive.scope.cta.space.cta.
  Intrinsic::ID nvvm(std::string Suffix) {
    std::replace(Suffix.begin(), Suffix.end(), '_', '.');
    return Intrinsic::lookupIntrinsicID("llvm.nvvm." + Suffix);
  }

  // ---- instructions ----
  bool isPure(const Stmt &S);
  void emit(const Stmt &S);
  bool emitMove(const Stmt &S);
  bool emitInt(const Stmt &S);
  bool emitFloat(const Stmt &S);
  bool emitCompare(const Stmt &S);
  bool emitConvert(const Stmt &S);
  bool emitMemory(const Stmt &S);
  bool emitAtomic(const Stmt &S);
  bool emitSync(const Stmt &S);
  bool emitMBarrier(const Stmt &S);
  bool emitBulk(const Stmt &S);
  bool emitMatrix(const Stmt &S);
  bool emitMisc(const Stmt &S);
  /// Re-emits \p S as a one-instruction asm call with \p Dirs giving each
  /// operand's direction and \p Cons its constraint letter (r, f, l).
  /// Operand directions for emitLeaf.
  enum { In = 0, Out = 1, InOut = 2 };
  void emitLeaf(const Stmt &S, ArrayRef<int> Dirs, ArrayRef<char> Cons);
};

char SASSExpandInlinePTX::ID = 0;
} // namespace

INITIALIZE_PASS(SASSExpandInlinePTX, "sass-expand-inline-ptx",
                "SASS expand inline PTX", false, false)

FunctionPass *llvm::createSASSExpandInlinePTXPass() {
  return new SASSExpandInlinePTX();
}

void PTXProgram::run(std::vector<Item> &Items) {
  // Operands: outputs first (result elements), then inputs.
  auto *IA = cast<InlineAsm>(CI.getCalledOperand());
  Type *RetTy = CI.getType();
  unsigned NumOuts = 0, Arg = 0;
  SmallVector<std::pair<unsigned, Value *>, 8> Init; // operand -> initial value
  for (const InlineAsm::ConstraintInfo &C : IA->ParseConstraints()) {
    if (C.Type == InlineAsm::isClobber)
      continue;
    if (C.isIndirect)
      fail("memory operands are not supported");
    Type *Ty;
    if (C.Type == InlineAsm::isOutput) {
      Ty = RetTy->isStructTy() ? RetTy->getStructElementType(NumOuts) : RetTy;
      OutIndex.push_back(NumOuts++);
    } else {
      Value *V = CI.getArgOperand(Arg++);
      Ty = V->getType();
      OutIndex.push_back(-1);
      if (!C.Codes.empty() && isDigit(C.Codes[0][0])) {
        unsigned Tied = std::stoi(C.Codes[0]);
        Init.push_back({Tied, V});
      }
      Init.push_back({(unsigned)OpTypes.size(), V});
    }
    OpTypes.push_back(Ty);
    unsigned Bits = Ty->isPointerTy() ? DL.getPointerSizeInBits(Ty->getPointerAddressSpace())
                                      : DL.getTypeSizeInBits(Ty);
    Var V;
    V.Ty = intTy(Bits);
    V.Slot = newSlot(V.Ty, "ptx.op" + Twine(OpTypes.size() - 1));
    Ops.push_back(V);
  }
  auto Bits = [&](Value *V, IntegerType *Ty) -> Value * {
    if (V->getType()->isPointerTy())
      return B.CreatePtrToInt(V, Ty);
    if (!V->getType()->isIntegerTy())
      V = B.CreateBitCast(V, intTy(DL.getTypeSizeInBits(V->getType())));
    return resize(V, Ty->getBitWidth(), false);
  };
  for (auto [N, V] : Init) {
    B.CreateStore(Bits(V, Ops[N].Ty), Ops[N].Slot);
    if (auto *C = dyn_cast<ConstantInt>(V))
      Ops[N].Const = C->getSExtValue();
  }

  // Region: Head ... Tail, with the call at the start of Tail.
  BasicBlock *Head = CI.getParent();
  Tail = Head->splitBasicBlock(&CI, "ptx.tail");
  Head->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Head);
  Scopes.emplace_back();
  for (Item &It : Items) {
    switch (It.K) {
    case Item::Open:
      Scopes.emplace_back();
      break;
    case Item::Close:
      if (Scopes.size() == 1)
        fail("unbalanced '}'");
      Scopes.pop_back();
      break;
    case Item::Label: {
      BasicBlock *BB = label(It.Name);
      B.CreateBr(BB);
      B.SetInsertPoint(BB);
      break;
    }
    case Item::Decl: {
      std::optional<PTy> T = parseType(StringRef(It.Name).drop_front());
      if (!T)
        fail("unsupported register type " + It.Name);
      for (const std::string &N : It.Names) {
        Var V;
        V.Ty = intTy(T->Bits);
        V.Slot = newSlot(V.Ty, "ptx." + N);
        Scopes.back()[N] = V;
      }
      break;
    }
    case Item::Inst:
      Cur = &It.S;
      emit(It.S);
      Cur = nullptr;
      break;
    }
  }
  B.CreateBr(Tail);
  for (auto &[Name, BB] : Labels)
    if (BB->empty())
      fail("branch to undefined label '" + Name + "'");

  // Results.
  B.SetInsertPoint(&CI);
  if (!RetTy->isVoidTy()) {
    SmallVector<Value *, 8> Vals;
    for (auto [N, Out] : enumerate(OutIndex)) {
      if (Out < 0)
        continue;
      Type *Ty = OpTypes[N];
      Value *V = B.CreateLoad(Ops[N].Ty, Ops[N].Slot);
      if (Ty->isPointerTy())
        V = B.CreateIntToPtr(V, Ty);
      else if (!Ty->isIntegerTy())
        V = B.CreateBitCast(resize(V, DL.getTypeSizeInBits(Ty), false), Ty);
      else
        V = resize(V, Ty->getIntegerBitWidth(), false);
      Vals.push_back(V);
    }
    Value *R;
    if (RetTy->isStructTy()) {
      R = PoisonValue::get(RetTy);
      for (auto [I, V] : enumerate(Vals))
        R = B.CreateInsertValue(R, V, I);
    } else {
      R = Vals[0];
    }
    CI.replaceAllUsesWith(R);
  }
  CI.eraseFromParent();
}

bool PTXProgram::isPure(const Stmt &S) {
  static const StringSet<> Pure = {
      "mov",  "add",  "sub",  "mul",  "mad",  "fma",  "min",  "max",  "neg",
      "abs",  "not",  "cnot", "and",  "or",   "xor",  "shl",  "shr",  "selp",
      "setp", "cvt",  "prmt", "lop3", "bfe",  "bfi",  "popc", "clz",  "brev",
      "ex2",  "lg2",  "rcp",  "sqrt", "rsqrt", "tanh", "div", "rem",  "mapa",
      "createpolicy", "copysign", "testp", "set"};
  return Pure.contains(S.Root) && !S.Ops.empty();
}

void PTXProgram::emit(const Stmt &S) {
  if (S.Root == "bra") {
    need(S, 1);
    if (S.Ops[0].K != Opnd::Reg)
      fail("bad branch target");
    BasicBlock *Target = label(S.Ops[0].Name);
    BasicBlock *Next = newBlock("ptx.next");
    if (S.Guarded) {
      Value *P = read(regOp(S.Guard), 1);
      if (S.GuardNeg)
        B.CreateCondBr(P, Next, Target);
      else
        B.CreateCondBr(P, Target, Next);
    } else {
      B.CreateBr(Target);
    }
    B.SetInsertPoint(Next);
    return;
  }
  BasicBlock *Join = nullptr;
  if (S.Guarded) {
    Opnd G = regOp(S.Guard);
    G.Neg = S.GuardNeg;
    Value *P = read(G, 1);
    if (isPure(S)) {
      Guard = P;
    } else {
      BasicBlock *Then = newBlock("ptx.then");
      Join = newBlock("ptx.join");
      B.CreateCondBr(P, Then, Join);
      B.SetInsertPoint(Then);
    }
  }
  bool Done = emitMove(S) || emitInt(S) || emitFloat(S) || emitCompare(S) ||
              emitConvert(S) || emitMemory(S) || emitAtomic(S) || emitSync(S) ||
              emitMBarrier(S) || emitBulk(S) || emitMatrix(S) || emitMisc(S);
  if (!Done)
    unsupported(S);
  Guard = nullptr;
  if (Join) {
    B.CreateBr(Join);
    B.SetInsertPoint(Join);
  }
}

bool PTXProgram::emitMove(const Stmt &S) {
  if (S.Root != "mov")
    return false;
  need(S, 2);
  PTy T = type(S);
  const Opnd &D = S.Ops[0], &A = S.Ops[1];
  if (T.Kind == 'p') {
    write(D, read(A, 1));
    return true;
  }
  if (A.K == Opnd::Reg && !lookup(A.Name) && A.Name[0] != '%' && A.Name[0] != '$')
    fail("mov of symbol '" + A.Name + "' is not supported");
  write(D, read(A, T.Bits, T), T);
  return true;
}

bool PTXProgram::emitInt(const Stmt &S) {
  static const StringSet<> Ops = {"add", "sub", "mul", "mad", "min", "max", "neg", "abs",
                                  "not", "cnot", "and", "or", "xor", "shl", "shr", "div",
                                  "rem", "popc", "clz", "brev", "bfe", "bfi"};
  if (!Ops.contains(S.Root))
    return false;
  PTy T = type(S);
  if (T.isFloat())
    return false;
  if (S.has("cc") || S.Root == "addc" || S.Root == "subc")
    fail("carry chains are not supported");
  const std::string &R = S.Root;
  if (T.Kind == 'p') { // and/or/xor/not on predicates
    Value *A = read(S.Ops[1], 1);
    Value *V = R == "not" ? B.CreateNot(A)
               : R == "and" ? B.CreateAnd(A, read(S.Ops[2], 1))
               : R == "or"  ? B.CreateOr(A, read(S.Ops[2], 1))
               : R == "xor" ? B.CreateXor(A, read(S.Ops[2], 1))
                            : nullptr;
    if (!V)
      unsupported(S);
    write(S.Ops[0], V);
    return true;
  }
  unsigned W = T.Bits;
  bool Sg = T.isSigned();
  auto In = [&](unsigned I, unsigned Bits = 0) { return read(S.Ops[I], Bits ? Bits : W, T); };
  Value *V = nullptr;
  if (R == "add" || R == "sub") {
    need(S, 3);
    if (S.has("sat")) {
      if (!Sg)
        unsupported(S);
      V = call(R == "add" ? Intrinsic::sadd_sat : Intrinsic::ssub_sat, {In(1), In(2)}, {intTy(W)});
    } else {
      V = R == "add" ? B.CreateAdd(In(1), In(2)) : B.CreateSub(In(1), In(2));
    }
  } else if (R == "mul" || R == "mad") {
    need(S, R == "mul" ? 3 : 4);
    if (S.has("wide")) {
      Value *A = resize(In(1), 2 * W, Sg), *Bv = resize(In(2), 2 * W, Sg);
      V = B.CreateMul(A, Bv);
      if (R == "mad")
        V = B.CreateAdd(V, read(S.Ops[3], 2 * W, T));
      write(S.Ops[0], V, T);
      return true;
    }
    if (S.has("hi")) {
      Value *A = resize(In(1), 2 * W, Sg), *Bv = resize(In(2), 2 * W, Sg);
      V = B.CreateTrunc(B.CreateLShr(B.CreateMul(A, Bv), W), intTy(W));
    } else {
      V = B.CreateMul(In(1), In(2));
    }
    if (R == "mad")
      V = B.CreateAdd(V, In(3));
  } else if (R == "min" || R == "max") {
    need(S, 3);
    Intrinsic::ID IID = R == "min" ? (Sg ? Intrinsic::smin : Intrinsic::umin)
                                   : (Sg ? Intrinsic::smax : Intrinsic::umax);
    V = call(IID, {In(1), In(2)}, {intTy(W)});
  } else if (R == "neg") {
    V = B.CreateNeg(In(1));
  } else if (R == "abs") {
    V = call(Intrinsic::abs, {In(1), B.getFalse()}, {intTy(W)});
  } else if (R == "not") {
    V = B.CreateNot(In(1));
  } else if (R == "cnot") {
    V = B.CreateZExt(B.CreateICmpEQ(In(1), ConstantInt::get(intTy(W), 0)), intTy(W));
  } else if (R == "and" || R == "or" || R == "xor") {
    need(S, 3);
    V = B.CreateBinOp(R == "and" ? Instruction::And : R == "or" ? Instruction::Or : Instruction::Xor,
                      In(1), In(2));
  } else if (R == "shl" || R == "shr") {
    need(S, 3);
    // PTX clamps shift amounts at the type width.
    Value *A = In(1), *Sh = read(S.Ops[2], 32);
    std::optional<int64_t> C = constant(S.Ops[2]);
    Value *ShW = resize(Sh, W, false);
    Value *Big = B.CreateICmpUGE(Sh, B.getInt32(W));
    if (R == "shl") {
      V = B.CreateShl(A, ShW);
      if (!C || *C >= W)
        V = B.CreateSelect(Big, ConstantInt::get(intTy(W), 0), V);
    } else if (Sg) {
      Value *Cl = C ? ShW : B.CreateSelect(Big, ConstantInt::get(intTy(W), W - 1), ShW);
      if (C && *C >= W)
        Cl = ConstantInt::get(intTy(W), W - 1);
      V = B.CreateAShr(A, Cl);
    } else {
      V = B.CreateLShr(A, ShW);
      if (!C || *C >= W)
        V = B.CreateSelect(Big, ConstantInt::get(intTy(W), 0), V);
    }
  } else if (R == "div" || R == "rem") {
    need(S, 3);
    V = R == "div" ? (Sg ? B.CreateSDiv(In(1), In(2)) : B.CreateUDiv(In(1), In(2)))
                   : (Sg ? B.CreateSRem(In(1), In(2)) : B.CreateURem(In(1), In(2)));
  } else if (R == "popc" || R == "clz") {
    V = call(R == "popc" ? Intrinsic::ctpop : Intrinsic::ctlz,
             R == "popc" ? ArrayRef<Value *>{In(1)} : ArrayRef<Value *>{In(1), B.getFalse()},
             {intTy(W)});
    write(S.Ops[0], resize(V, 32, false));
    return true;
  } else if (R == "brev") {
    V = call(Intrinsic::bitreverse, {In(1)}, {intTy(W)});
  } else if (R == "bfe") {
    // d[i] = i < len && pos+i <= msb ? a[pos+i] : sbit, where pos and len
    // are the low bytes of b and c and sbit (signed types) is
    // a[min(pos+len-1, msb)]. Computed at twice the width so every shift
    // is in range.
    need(S, 4);
    IntegerType *W2 = intTy(2 * W);
    auto C2 = [&](uint64_t V) { return ConstantInt::get(W2, V); };
    Value *A = resize(In(1), 2 * W, Sg);
    Value *Pos = B.CreateAnd(resize(read(S.Ops[2], 32), 2 * W, false), C2(0xff));
    Value *Len = B.CreateAnd(resize(read(S.Ops[3], 32), 2 * W, false), C2(0xff));
    Value *P = call(Intrinsic::umin, {Pos, C2(W)}, {W2});
    Value *L = call(Intrinsic::umin, {Len, C2(W)}, {W2});
    Value *Field = Sg ? B.CreateAShr(A, P) : B.CreateLShr(A, P);
    Value *Mask = B.CreateSub(B.CreateShl(C2(1), L), C2(1));
    Field = B.CreateAnd(Field, Mask);
    if (Sg) {
      Value *Top = B.CreateSub(B.CreateAdd(Pos, Len), C2(1));
      Top = B.CreateSelect(B.CreateICmpEQ(B.CreateAdd(Pos, Len), C2(0)), C2(W - 1),
                           call(Intrinsic::umin, {Top, C2(W - 1)}, {W2}));
      Value *SBit = B.CreateAnd(B.CreateLShr(A, Top), C2(1));
      Value *Ext = B.CreateSelect(B.CreateICmpNE(SBit, C2(0)), B.CreateNot(Mask), C2(0));
      Field = B.CreateOr(Field, Ext);
    }
    V = B.CreateTrunc(Field, intTy(W));
  } else if (R == "bfi") {
    // d = b with bits [pos, pos+len) replaced by the low bits of a.
    need(S, 5);
    Value *A = In(1), *Bv = In(2), *Pos = resize(read(S.Ops[3], 32), W, false),
          *Len = resize(read(S.Ops[4], 32), W, false);
    Value *Wc = ConstantInt::get(intTy(W), W);
    Value *Lc = call(Intrinsic::umin, {Len, Wc}, {intTy(W)});
    Value *Ones = ConstantInt::getAllOnesValue(intTy(W));
    Value *Mask = B.CreateSelect(B.CreateICmpEQ(Lc, Wc), Ones,
                                 B.CreateSub(B.CreateShl(ConstantInt::get(intTy(W), 1), Lc),
                                             ConstantInt::get(intTy(W), 1)));
    Value *InRange = B.CreateICmpULT(Pos, Wc);
    Value *PosS = B.CreateSelect(InRange, Pos, ConstantInt::get(intTy(W), 0));
    Mask = B.CreateSelect(InRange, B.CreateShl(Mask, PosS), ConstantInt::get(intTy(W), 0));
    V = B.CreateOr(B.CreateAnd(Bv, B.CreateNot(Mask)), B.CreateAnd(B.CreateShl(A, PosS), Mask));
  }
  if (!V)
    unsupported(S);
  write(S.Ops[0], V, T);
  return true;
}

bool PTXProgram::emitFloat(const Stmt &S) {
  static const StringSet<> Ops = {"add",  "sub",   "mul",  "fma", "mad",  "min",
                                  "max",  "neg",   "abs",  "div", "rcp",  "sqrt",
                                  "rsqrt", "ex2",  "lg2",  "tanh", "copysign"};
  if (!Ops.contains(S.Root))
    return false;
  PTy T = type(S);
  if (!T.isFloat())
    return false;
  const std::string &R = S.Root;
  Type *FT = floatTy(T);
  auto In = [&](unsigned I) { return readF(S.Ops[I], FT, T); };
  bool Approx = S.has("approx");
  bool Ftz = S.has("ftz");
  Value *V = nullptr;
  FastMathFlags FMF;
  if (R == "add" || R == "sub" || R == "mul") {
    need(S, 3);
    // Without a rounding modifier ptxas may contract into FMA.
    bool Rounded = S.has("rn") || S.has("rz") || S.has("rm") || S.has("rp");
    if (S.has("rz") || S.has("rm") || S.has("rp"))
      fail("only round-to-nearest float arithmetic is supported");
    if (!Rounded)
      FMF.setAllowContract();
    V = B.CreateBinOp(R == "add" ? Instruction::FAdd : R == "sub" ? Instruction::FSub : Instruction::FMul,
                      In(1), In(2));
    cast<Instruction>(V)->setFastMathFlags(FMF);
  } else if (R == "fma" || R == "mad") {
    need(S, 4);
    V = call(Intrinsic::fma, {In(1), In(2), In(3)}, {FT});
  } else if (R == "min" || R == "max") {
    need(S, 3);
    V = call(R == "min" ? Intrinsic::minnum : Intrinsic::maxnum, {In(1), In(2)}, {FT});
  } else if (R == "neg") {
    V = B.CreateFNeg(In(1));
  } else if (R == "abs") {
    V = call(Intrinsic::fabs, {In(1)}, {FT});
  } else if (R == "copysign") {
    V = call(Intrinsic::copysign, {In(2), In(1)}, {FT});
  } else if (R == "div") {
    need(S, 3);
    V = B.CreateFDiv(In(1), In(2));
    if (Approx || S.has("full")) {
      FMF.setAllowReciprocal();
      cast<Instruction>(V)->setFastMathFlags(FMF);
    }
  } else if (R == "rcp") {
    if (Approx && Ftz && T.Bits == 32 && T.Lanes == 1) {
      V = call(Intrinsic::nvvm_rcp_approx_ftz_f, {In(1)});
    } else {
      V = B.CreateFDiv(ConstantFP::get(FT, 1.0), In(1));
      if (Approx) {
        FMF.setAllowReciprocal();
        cast<Instruction>(V)->setFastMathFlags(FMF);
      }
    }
  } else if (R == "sqrt" || R == "lg2" || R == "tanh") {
    Intrinsic::ID IID = R == "sqrt" ? Intrinsic::sqrt : R == "lg2" ? Intrinsic::log2 : Intrinsic::tanh;
    if (!Approx && R != "sqrt")
      unsupported(S);
    V = call(IID, {In(1)}, {FT});
    if (Approx) {
      FMF.setApproxFunc();
      cast<Instruction>(V)->setFastMathFlags(FMF);
    }
  } else if (R == "rsqrt") {
    if (!Approx || T.Bits != 32)
      unsupported(S);
    V = call(Ftz ? Intrinsic::nvvm_rsqrt_approx_ftz_f : Intrinsic::nvvm_rsqrt_approx_f, {In(1)});
  } else if (R == "ex2") {
    if (!Approx)
      unsupported(S);
    if (T.Kind == 'f' && T.Bits == 32 && !Ftz) {
      // ptxas: inputs below -126 would give denormals that MUFU.EX2
      // flushes; halve the input and square the result.
      Value *X = In(1);
      Value *Small = B.CreateFCmpULT(X, ConstantFP::get(FT, -126.0));
      Value *Xs = B.CreateSelect(Small, B.CreateFMul(X, ConstantFP::get(FT, 0.5)), X);
      Value *E = call(Intrinsic::nvvm_ex2_approx_ftz, {Xs}, {FT});
      V = B.CreateSelect(Small, B.CreateFMul(E, E), E);
    } else {
      V = call(Ftz ? Intrinsic::nvvm_ex2_approx_ftz : Intrinsic::nvvm_ex2_approx, {In(1)}, {FT});
    }
  }
  if (!V)
    unsupported(S);
  if (S.has("sat")) {
    V = call(Intrinsic::maxnum, {V, ConstantFP::get(FT, 0.0)}, {FT});
    V = call(Intrinsic::minnum, {V, ConstantFP::get(FT, 1.0)}, {FT});
  }
  if (S.has("relu"))
    V = call(Intrinsic::maxnum, {V, ConstantFP::get(FT, 0.0)}, {FT});
  writeF(S.Ops[0], V);
  return true;
}

bool PTXProgram::emitCompare(const Stmt &S) {
  if (S.Root == "selp") {
    need(S, 4);
    PTy T = type(S);
    Value *P = read(S.Ops[3], 1);
    write(S.Ops[0], B.CreateSelect(P, read(S.Ops[1], T.Bits, T), read(S.Ops[2], T.Bits, T)), T);
    return true;
  }
  if (S.Root != "setp")
    return false;
  PTy T = type(S);
  StringRef Cmp = S.Mods[0];
  std::optional<StringRef> BoolOp;
  if (S.Mods.size() > 1 && (S.Mods[1] == "and" || S.Mods[1] == "or" || S.Mods[1] == "xor"))
    BoolOp = S.Mods[1];
  need(S, BoolOp ? 4 : 3);
  Value *C;
  if (T.isFloat()) {
    if (T.Lanes != 1)
      unsupported(S);
    Type *FT = floatTy(T);
    Value *A = readF(S.Ops[1], FT, T), *Bv = readF(S.Ops[2], FT, T);
    CmpInst::Predicate P = StringSwitch<CmpInst::Predicate>(Cmp)
                               .Case("eq", CmpInst::FCMP_OEQ).Case("ne", CmpInst::FCMP_ONE)
                               .Case("lt", CmpInst::FCMP_OLT).Case("le", CmpInst::FCMP_OLE)
                               .Case("gt", CmpInst::FCMP_OGT).Case("ge", CmpInst::FCMP_OGE)
                               .Case("equ", CmpInst::FCMP_UEQ).Case("neu", CmpInst::FCMP_UNE)
                               .Case("ltu", CmpInst::FCMP_ULT).Case("leu", CmpInst::FCMP_ULE)
                               .Case("gtu", CmpInst::FCMP_UGT).Case("geu", CmpInst::FCMP_UGE)
                               .Case("num", CmpInst::FCMP_ORD).Case("nan", CmpInst::FCMP_UNO)
                               .Default(CmpInst::BAD_FCMP_PREDICATE);
    if (P == CmpInst::BAD_FCMP_PREDICATE)
      unsupported(S);
    C = B.CreateFCmp(P, A, Bv);
  } else {
    bool Sg = T.isSigned();
    Value *A = read(S.Ops[1], T.Bits, T), *Bv = read(S.Ops[2], T.Bits, T);
    CmpInst::Predicate P = StringSwitch<CmpInst::Predicate>(Cmp)
                               .Case("eq", CmpInst::ICMP_EQ).Case("ne", CmpInst::ICMP_NE)
                               .Case("lt", Sg ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT)
                               .Case("le", Sg ? CmpInst::ICMP_SLE : CmpInst::ICMP_ULE)
                               .Case("gt", Sg ? CmpInst::ICMP_SGT : CmpInst::ICMP_UGT)
                               .Case("ge", Sg ? CmpInst::ICMP_SGE : CmpInst::ICMP_UGE)
                               .Case("lo", CmpInst::ICMP_ULT).Case("ls", CmpInst::ICMP_ULE)
                               .Case("hi", CmpInst::ICMP_UGT).Case("hs", CmpInst::ICMP_UGE)
                               .Default(CmpInst::BAD_ICMP_PREDICATE);
    if (P == CmpInst::BAD_ICMP_PREDICATE)
      unsupported(S);
    C = B.CreateICmp(P, A, Bv);
  }
  Value *NC = B.CreateNot(C);
  if (BoolOp) {
    Value *X = read(S.Ops[3], 1);
    auto Comb = [&](Value *V) {
      return *BoolOp == "and" ? B.CreateAnd(V, X) : *BoolOp == "or" ? B.CreateOr(V, X) : B.CreateXor(V, X);
    };
    C = Comb(C);
    NC = Comb(NC);
  }
  write(S.Ops[0], C);
  if (!S.Ops[0].Name2.empty()) {
    Opnd Q = regOp(S.Ops[0].Name2);
    write(Q, NC);
  }
  return true;
}

bool PTXProgram::emitConvert(const Stmt &S) {
  if (S.Root == "prmt") {
    need(S, 4);
    StringRef Mode;
    for (StringRef Mo : S.Mods)
      if (Mo == "f4e" || Mo == "b4e" || Mo == "rc8" || Mo == "ecl" || Mo == "ecr" || Mo == "rc16")
        Mode = Mo;
    Value *A = read(S.Ops[1], 32), *Bv = read(S.Ops[2], 32), *C = read(S.Ops[3], 32);
    Value *V;
    if (Mode.empty())
      V = call(Intrinsic::nvvm_prmt, {A, Bv, C});
    else if (Mode == "f4e")
      V = call(Intrinsic::nvvm_prmt_f4e, {A, Bv, C});
    else if (Mode == "b4e")
      V = call(Intrinsic::nvvm_prmt_b4e, {A, Bv, C});
    else
      unsupported(S);
    write(S.Ops[0], V);
    return true;
  }
  if (S.Root == "lop3") {
    need(S, 5);
    if (!S.has("b32") || !S.Ops[0].Name2.empty())
      unsupported(S);
    unsigned Lut = immediate(S.Ops[4], "lop3 lookup table") & 0xff;
    Value *A = read(S.Ops[1], 32), *Bv = read(S.Ops[2], 32), *C = read(S.Ops[3], 32);
    // f = c ? f1(a,b) : f0(a,b), with a=0xf0, b=0xcc, c=0xaa.
    auto Two = [&](unsigned L) -> Value * { // L: 4-bit table over (a,b)
      Value *Z = B.getInt32(0), *O = B.getInt32(-1);
      switch (L & 0xf) {
      case 0x0: return Z;
      case 0xf: return O;
      case 0x8: return B.CreateAnd(A, Bv);
      case 0xe: return B.CreateOr(A, Bv);
      case 0x6: return B.CreateXor(A, Bv);
      case 0x9: return B.CreateNot(B.CreateXor(A, Bv));
      case 0xc: return A;
      case 0xa: return Bv;
      case 0x3: return B.CreateNot(A);
      case 0x5: return B.CreateNot(Bv);
      case 0x4: return B.CreateAnd(A, B.CreateNot(Bv));
      case 0x2: return B.CreateAnd(B.CreateNot(A), Bv);
      case 0x1: return B.CreateNot(B.CreateOr(A, Bv));
      case 0x7: return B.CreateNot(B.CreateAnd(A, Bv));
      case 0xd: return B.CreateOr(A, B.CreateNot(Bv));
      case 0xb: return B.CreateOr(B.CreateNot(A), Bv);
      }
      llvm_unreachable("4-bit table");
    };
    // Bit i of the 8-bit table is f(a=i>>2&1, b=i>>1&1, c=i&1); the 4-bit
    // table index is a*2+b... built as bit (a<<1|b) of L.
    unsigned L0 = 0, L1 = 0;
    for (unsigned Ab = 0; Ab < 4; ++Ab) {
      unsigned Av = Ab >> 1, Bb = Ab & 1;
      // index into LUT: a contributes 4, b 2, c 1.
      L0 |= ((Lut >> (Av * 4 + Bb * 2 + 0)) & 1) << (Av * 2 + Bb);
      L1 |= ((Lut >> (Av * 4 + Bb * 2 + 1)) & 1) << (Av * 2 + Bb);
    }
    // Two() expects tables indexed like 0xc=a, 0xa=b: bit (a*2+b) — matches.
    Value *V;
    if (L0 == L1)
      V = Two(L0);
    else if (L0 == 0)
      V = B.CreateAnd(C, Two(L1));
    else if (L1 == 0)
      V = B.CreateAnd(B.CreateNot(C), Two(L0));
    else {
      Value *F0 = Two(L0), *F1 = Two(L1);
      V = B.CreateXor(F0, B.CreateAnd(B.CreateXor(F0, F1), C));
    }
    write(S.Ops[0], V);
    return true;
  }
  if (S.Root != "cvt" && S.Root != "cvta")
    return false;
  if (S.Root == "cvta") {
    // cvta[.to].space.size d, a: generic <-> state-space addresses.
    need(S, 2);
    unsigned AS = space(S);
    PTy T = type(S);
    Value *A = read(S.Ops[1], T.Bits);
    Value *V;
    if (S.has("to")) {
      Value *P = B.CreateIntToPtr(resize(A, 64, false), B.getPtrTy(0));
      V = B.CreatePtrToInt(B.CreateAddrSpaceCast(P, B.getPtrTy(AS)),
                           intTy(DL.getPointerSizeInBits(AS)));
    } else {
      Value *P = B.CreateIntToPtr(resize(A, DL.getPointerSizeInBits(AS), false), B.getPtrTy(AS));
      V = B.CreatePtrToInt(B.CreateAddrSpaceCast(P, B.getPtrTy(0)), intTy(64));
    }
    write(S.Ops[0], V);
    return true;
  }
  SmallVector<PTy, 2> Ts;
  for (StringRef Mo : S.Mods)
    if (std::optional<PTy> T = parseType(Mo))
      Ts.push_back(*T);
  if (Ts.size() != 2)
    fail("cvt needs a destination and a source type");
  PTy D = Ts[0], A = Ts[1];
  bool Relu = S.has("relu");
  StringRef DName, AName;
  for (StringRef Mo : S.Mods)
    if (parseType(Mo))
      (DName.empty() ? DName : AName) = Mo;
  auto Fp8 = [&](StringRef Name) -> std::string {
    return Name == "e4m3x2" ? "e4m3x2" : "e5m2x2";
  };
  // FP8 pairs.
  if (D.Kind == '8' && A.Kind == 'f' && A.Bits == 32) {
    need(S, 3);
    Intrinsic::ID IID = Intrinsic::lookupIntrinsicID(
        "llvm.nvvm.ff.to." + Fp8(DName) + ".rn" + (Relu ? ".relu" : ""));
    write(S.Ops[0], call(IID, {readF(S.Ops[1], B.getFloatTy(), A), readF(S.Ops[2], B.getFloatTy(), A)}));
    return true;
  }
  if (D.Kind == '8' && A.Kind == 'h' && A.Lanes == 2) {
    need(S, 2);
    Intrinsic::ID IID = Intrinsic::lookupIntrinsicID(
        "llvm.nvvm.f16x2.to." + Fp8(DName) + ".rn" + (Relu ? ".relu" : ""));
    write(S.Ops[0], call(IID, {readF(S.Ops[1], floatTy(A), A)}));
    return true;
  }
  if (D.Kind == 'h' && D.Lanes == 2 && A.Kind == '8') {
    need(S, 2);
    Intrinsic::ID IID = Intrinsic::lookupIntrinsicID(
        "llvm.nvvm." + Fp8(AName) + ".to.f16x2.rn" + (Relu ? ".relu" : ""));
    writeF(S.Ops[0], call(IID, {read(S.Ops[1], 16)}));
    return true;
  }
  // f16x2/bf16x2 from two f32: d = {cvt(a) high, cvt(b) low}.
  if ((D.Kind == 'h' || D.Kind == 'B') && D.Lanes == 2 && A.Kind == 'f' && A.Bits == 32) {
    need(S, 3);
    Type *HT = D.Kind == 'h' ? B.getHalfTy() : B.getBFloatTy();
    auto Half = [&](const Opnd &O) {
      Value *X = readF(O, B.getFloatTy(), A);
      if (Relu)
        X = call(Intrinsic::maxnum, {X, ConstantFP::get(B.getFloatTy(), 0.0)}, {B.getFloatTy()});
      return B.CreateZExt(B.CreateBitCast(B.CreateFPTrunc(X, HT), B.getInt16Ty()), B.getInt32Ty());
    };
    if (!S.has("rn"))
      unsupported(S);
    Value *Hi = Half(S.Ops[1]), *Lo = Half(S.Ops[2]);
    write(S.Ops[0], B.CreateOr(B.CreateShl(Hi, 16), Lo));
    return true;
  }
  need(S, 2);
  if (D.Lanes != 1 || A.Lanes != 1)
    unsupported(S);
  if (!D.isFloat() && !A.isFloat()) {
    // Integer resize: extension follows the source's signedness, then
    // the destination register gets its own.
    Value *V = read(S.Ops[1], A.Bits, A);
    if (S.has("sat"))
      unsupported(S);
    V = resize(resize(V, std::max(A.Bits, D.Bits), A.isSigned()), D.Bits, false);
    write(S.Ops[0], V, D);
    return true;
  }
  Type *DT = D.isFloat() ? floatTy(D) : nullptr, *AT = A.isFloat() ? floatTy(A) : nullptr;
  bool Rni = S.has("rni"), Rzi = S.has("rzi"), Rmi = S.has("rmi"), Rpi = S.has("rpi");
  if (D.isFloat() && A.isFloat()) {
    Value *X = readF(S.Ops[1], AT, A), *V;
    unsigned DB = DL.getTypeSizeInBits(DT), AB = DL.getTypeSizeInBits(AT);
    if (Rni || Rzi || Rmi || Rpi) {
      Intrinsic::ID IID = Rni ? Intrinsic::roundeven : Rzi ? Intrinsic::trunc
                          : Rmi ? Intrinsic::floor : Intrinsic::ceil;
      X = call(IID, {X}, {AT});
    }
    if (DB < AB) {
      if (S.has("rz") || S.has("rm") || S.has("rp"))
        unsupported(S);
      V = B.CreateFPTrunc(X, DT);
    } else if (DB > AB) {
      V = B.CreateFPExt(X, DT);
    } else {
      V = X;
    }
    if (Relu)
      V = call(Intrinsic::maxnum, {V, ConstantFP::get(DT, 0.0)}, {DT});
    if (S.has("sat")) {
      V = call(Intrinsic::maxnum, {V, ConstantFP::get(DT, 0.0)}, {DT});
      V = call(Intrinsic::minnum, {V, ConstantFP::get(DT, 1.0)}, {DT});
    }
    writeF(S.Ops[0], V);
    return true;
  }
  if (D.isFloat()) { // int -> float
    if (S.has("rz") || S.has("rm") || S.has("rp"))
      unsupported(S);
    Value *X = read(S.Ops[1], A.Bits, A);
    writeF(S.Ops[0], A.isSigned() ? B.CreateSIToFP(X, DT) : B.CreateUIToFP(X, DT));
    return true;
  }
  // float -> int: PTX saturates (NaN -> 0).
  Value *X = readF(S.Ops[1], AT, A);
  if (Rni || Rmi || Rpi)
    X = call(Rni ? Intrinsic::roundeven : Rmi ? Intrinsic::floor : Intrinsic::ceil, {X}, {AT});
  else if (!Rzi)
    unsupported(S);
  Value *V = call(D.isSigned() ? Intrinsic::fptosi_sat : Intrinsic::fptoui_sat, {X},
                  {intTy(D.Bits), AT});
  write(S.Ops[0], V, D);
  return true;
}

bool PTXProgram::emitMemory(const Stmt &S) {
  bool Load = S.Root == "ld" || S.Root == "ldu";
  if (!Load && S.Root != "st")
    return false;
  if (S.has("async"))
    return false; // st.async: emitBulk
  PTy T = type(S);
  unsigned AS = space(S);
  unsigned N = S.has("v2") ? 2 : S.has("v4") ? 4 : S.has("v8") ? 8 : 1;
  bool HasHint = S.has("L2::cache_hint");
  need(S, 2 + HasHint);
  const Opnd &Addr = S.Ops[Load ? 1 : 0], &Val = S.Ops[Load ? 0 : 1];
  if ((N > 1) != (Val.K == Opnd::Vec) || (N > 1 && Val.Elts.size() != N))
    fail("vector width does not match the operand");
  Value *P = address(Addr, AS);
  unsigned EB = T.Bits;
  Type *ET = intTy(EB);
  if (AS == 7 && N > 1) {
    // shared::cluster accesses are generic LD/ST on {address, window}:
    // one per element.
    if (S.has("acquire") || S.has("release") || S.has("relaxed"))
      fail("ordered shared::cluster vector accesses are not supported");
    for (unsigned I = 0; I < N; ++I) {
      Value *EP = B.CreateConstGEP1_64(B.getInt8Ty(), P, I * EB / 8);
      if (Load)
        write(Val.Elts[I], B.CreateAlignedLoad(ET, EP, Align(EB / 8), S.has("volatile")), T);
      else
        B.CreateAlignedStore(read(Val.Elts[I], EB, T), EP, Align(EB / 8), S.has("volatile"));
    }
    return true;
  }
  Type *Ty = N == 1 ? ET : (Type *)FixedVectorType::get(ET, N);
  Align Al(std::max(1u, EB * N / 8));
  std::optional<AtomicOrdering> Ord;
  if (S.has("acquire"))
    Ord = AtomicOrdering::Acquire;
  else if (S.has("release"))
    Ord = AtomicOrdering::Release;
  else if (S.has("relaxed"))
    Ord = AtomicOrdering::Monotonic;
  if (Ord && N > 1)
    fail("ordered vector accesses are not supported");
  bool Volatile = S.has("volatile") || S.has("mmio");
  // CTA-scope ordering on shared memory: the shared memory pipe keeps a
  // thread's accesses in order, so (as ptxas) a release store only needs a
  // CTA membar before it and an acquire load nothing; compiler ordering
  // comes from volatile and a single-thread fence.
  if (Ord && AS == 3 && scope(S) == Ctx.getOrInsertSyncScopeID("block")) {
    if (*Ord == AtomicOrdering::Release)
      B.CreateFence(AtomicOrdering::Release, scope(S));
    Ord.reset();
    Volatile = true;
    if (Load) {
      LoadInst *L = B.CreateAlignedLoad(Ty, P, Al, true);
      B.CreateFence(AtomicOrdering::Acquire, SyncScope::SingleThread);
      write(Val, L, T);
      return true;
    }
  }
  if (Load) {
    LoadInst *L = B.CreateAlignedLoad(Ty, P, Al, Volatile);
    if (Ord)
      L->setAtomic(*Ord, scope(S));
    if ((S.has("nc") || S.Root == "ldu") && !Ord && !Volatile)
      L->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(Ctx, {}));
    if (N == 1) {
      write(Val, L, T);
    } else {
      for (unsigned I = 0; I < N; ++I)
        write(Val.Elts[I], B.CreateExtractElement(L, I), T);
    }
    return true;
  }
  Value *V;
  if (N == 1) {
    V = read(Val, EB, T);
  } else {
    V = PoisonValue::get(Ty);
    for (unsigned I = 0; I < N; ++I)
      V = B.CreateInsertElement(V, read(Val.Elts[I], EB, T), I);
  }
  StoreInst *St = B.CreateAlignedStore(V, P, Al, Volatile);
  if (Ord)
    St->setAtomic(*Ord, scope(S));
  return true;
}

bool PTXProgram::emitAtomic(const Stmt &S) {
  bool Red = S.Root == "red";
  if (!Red && S.Root != "atom")
    return false;
  if (S.has("async"))
    return false;
  PTy T = type(S);
  unsigned AS = space(S);
  unsigned N = S.has("v2") ? 2 : S.has("v4") ? 4 : S.has("v8") ? 8 : 1;
  AtomicOrdering Ord = S.has("acquire") ? AtomicOrdering::Acquire
                       : S.has("release") ? AtomicOrdering::Release
                       : S.has("acq_rel") ? AtomicOrdering::AcquireRelease
                                          : AtomicOrdering::Monotonic;
  SyncScope::ID SSID = scope(S);
  unsigned First = Red ? 0 : 1;
  const Opnd &Addr = S.Ops[First];
  StringRef Op;
  for (StringRef Mo : S.Mods)
    if (Mo == "add" || Mo == "min" || Mo == "max" || Mo == "inc" || Mo == "dec" || Mo == "and" ||
        Mo == "or" || Mo == "xor" || Mo == "exch" || Mo == "cas")
      Op = Mo;
  if (Op.empty())
    unsupported(S);
  Value *P = address(Addr, AS);
  if (Op == "cas") {
    need(S, 4);
    if (Red || N != 1)
      unsupported(S);
    Value *Cmp = read(S.Ops[2], T.Bits), *New = read(S.Ops[3], T.Bits);
    AtomicOrdering Fail = Ord == AtomicOrdering::Release ? AtomicOrdering::Monotonic
                          : Ord == AtomicOrdering::AcquireRelease ? AtomicOrdering::Acquire
                                                                  : Ord;
    Value *R = B.CreateAtomicCmpXchg(P, Cmp, New, Align(T.Bits / 8), Ord, Fail, SSID);
    write(S.Ops[0], B.CreateExtractValue(R, 0), T);
    return true;
  }
  need(S, First + 2);
  bool FP = T.isFloat();
  AtomicRMWInst::BinOp BO;
  if (Op == "add")
    BO = FP ? AtomicRMWInst::FAdd : AtomicRMWInst::Add;
  else if (Op == "min")
    BO = FP ? AtomicRMWInst::FMin : T.isSigned() ? AtomicRMWInst::Min : AtomicRMWInst::UMin;
  else if (Op == "max")
    BO = FP ? AtomicRMWInst::FMax : T.isSigned() ? AtomicRMWInst::Max : AtomicRMWInst::UMax;
  else if (Op == "inc")
    BO = AtomicRMWInst::UIncWrap;
  else if (Op == "dec")
    BO = AtomicRMWInst::UDecWrap;
  else if (Op == "and")
    BO = AtomicRMWInst::And;
  else if (Op == "or")
    BO = AtomicRMWInst::Or;
  else if (Op == "xor")
    BO = AtomicRMWInst::Xor;
  else
    BO = AtomicRMWInst::Xchg;
  Type *ET = FP ? floatTy(T) : intTy(T.Bits);
  unsigned ESize = DL.getTypeStoreSize(ET);
  const Opnd &Val = S.Ops[First + 1];
  if (N > 1) {
    // .v2/.v4 reductions are atomic per element.
    if (Val.K != Opnd::Vec || Val.Elts.size() != N || !Red)
      unsupported(S);
    for (unsigned I = 0; I < N; ++I) {
      Value *EP = B.CreateConstGEP1_64(B.getInt8Ty(), P, I * ESize);
      Value *V = FP ? readF(Val.Elts[I], ET, T) : read(Val.Elts[I], T.Bits, T);
      B.CreateAtomicRMW(BO, EP, V, Align(ESize), Ord, SSID);
    }
    return true;
  }
  Value *V = FP ? readF(Val, ET, T) : read(Val, T.Bits, T);
  Value *R = B.CreateAtomicRMW(BO, P, V, Align(ESize), Ord, SSID);
  if (!Red) {
    if (FP)
      writeF(S.Ops[0], R);
    else
      write(S.Ops[0], R, T);
  }
  return true;
}

bool PTXProgram::emitSync(const Stmt &S) {
  const std::string &R = S.Root;
  if (R == "fence" || R == "membar") {
    if (S.has("proxy")) {
      std::string Kind;
      if (S.has("async"))
        Kind = S.has("global") ? "async.global" : S.has("shared") ? "async.shared_cta"
               : S.has("shared::cluster") ? "async.shared_cluster" : "async";
      else if (S.has("alias"))
        Kind = "alias";
      else
        unsupported(S);
      call(Intrinsic::lookupIntrinsicID("llvm.nvvm.fence.proxy." + Kind), {});
      return true;
    }
    if (S.has("mbarrier_init")) {
      call(Intrinsic::nvvm_fence_mbarrier_init_release_cluster, {});
      return true;
    }
    AtomicOrdering Ord = S.has("acq_rel") ? AtomicOrdering::AcquireRelease
                         : S.has("acquire") ? AtomicOrdering::Acquire
                         : S.has("release") ? AtomicOrdering::Release
                                            : AtomicOrdering::SequentiallyConsistent;
    SyncScope::ID SSID;
    if (R == "membar")
      SSID = S.has("cta") ? Ctx.getOrInsertSyncScopeID("block")
             : S.has("sys") ? SyncScope::System : Ctx.getOrInsertSyncScopeID("device");
    else
      SSID = scope(S);
    B.CreateFence(Ord, SSID);
    return true;
  }
  if (R == "bar" || R == "barrier") {
    if (S.has("cluster")) {
      bool Aligned = S.has("aligned");
      Intrinsic::ID IID;
      if (S.has("arrive"))
        IID = S.has("relaxed") ? (Aligned ? Intrinsic::nvvm_barrier_cluster_arrive_relaxed_aligned
                                          : Intrinsic::nvvm_barrier_cluster_arrive_relaxed)
                               : (Aligned ? Intrinsic::nvvm_barrier_cluster_arrive_aligned
                                          : Intrinsic::nvvm_barrier_cluster_arrive);
      else if (S.has("wait"))
        IID = Aligned ? Intrinsic::nvvm_barrier_cluster_wait_aligned
                      : Intrinsic::nvvm_barrier_cluster_wait;
      else
        unsupported(S);
      // A grid launched without clusters is a cluster of one CTA: as ptxas,
      // skip the arrive and wait at the CTA barrier instead.
      BasicBlock *Then = newBlock("ptx.cluster"), *Else = newBlock("ptx.nocluster"),
                 *Join = newBlock("ptx.join");
      B.CreateCondBr(call(Intrinsic::nvvm_is_explicit_cluster, {}), Then, Else);
      B.SetInsertPoint(Then);
      call(IID, {});
      B.CreateBr(Join);
      B.SetInsertPoint(Else);
      if (S.has("wait"))
        call(Intrinsic::nvvm_barrier_cta_sync_aligned_all, {B.getInt32(0)});
      B.CreateBr(Join);
      B.SetInsertPoint(Join);
      return true;
    }
    if (S.has("warp")) {
      need(S, 1);
      call(Intrinsic::nvvm_bar_warp_sync, {read(S.Ops[0], 32)});
      return true;
    }
    // bar.sync / bar.arrive are aligned; barrier.* only with .aligned.
    bool Aligned = R == "bar" || S.has("aligned");
    bool Arrive = S.has("arrive");
    if (!Arrive && !S.has("sync"))
      unsupported(S);
    if (S.Ops.empty() || S.Ops.size() > 2)
      fail("bar expects 1 or 2 operands");
    Value *Id = read(S.Ops[0], 32);
    if (S.Ops.size() == 2) {
      Intrinsic::ID IID = Arrive ? (Aligned ? Intrinsic::nvvm_barrier_cta_arrive_aligned_count
                                            : Intrinsic::nvvm_barrier_cta_arrive_count)
                                 : (Aligned ? Intrinsic::nvvm_barrier_cta_sync_aligned_count
                                            : Intrinsic::nvvm_barrier_cta_sync_count);
      call(IID, {Id, read(S.Ops[1], 32)});
    } else {
      if (Arrive)
        fail("bar.arrive needs a thread count");
      call(Aligned ? Intrinsic::nvvm_barrier_cta_sync_aligned_all : Intrinsic::nvvm_barrier_cta_sync_all,
           {Id});
    }
    return true;
  }
  if (R == "griddepcontrol") {
    call(S.has("wait") ? Intrinsic::nvvm_griddepcontrol_wait
                       : Intrinsic::nvvm_griddepcontrol_launch_dependents, {});
    return true;
  }
  return false;
}

bool PTXProgram::emitMBarrier(const Stmt &S) {
  const std::string &R = S.Root;
  if (R == "cp" && S.Mods.size() >= 3 && S.Mods[0] == "async" && S.Mods[1] == "mbarrier") {
    // cp.async.mbarrier.arrive[.noinc].shared.b64 [a]
    need(S, 1);
    bool Shared = S.has("shared");
    Value *P = address(S.Ops[0], Shared ? 3 : 0);
    Intrinsic::ID IID = S.has("noinc")
                            ? (Shared ? Intrinsic::nvvm_cp_async_mbarrier_arrive_noinc_shared
                                      : Intrinsic::nvvm_cp_async_mbarrier_arrive_noinc)
                            : (Shared ? Intrinsic::nvvm_cp_async_mbarrier_arrive_shared
                                      : Intrinsic::nvvm_cp_async_mbarrier_arrive);
    call(IID, {P});
    return true;
  }
  if (R != "mbarrier")
    return false;
  if (S.Mods.empty())
    unsupported(S);
  const std::string &Op = S.Mods[0];
  bool Cluster = S.has("shared::cluster");
  if (!Cluster && !S.has("shared"))
    fail("mbarrier needs the .shared state space");
  std::string Scope = S.has("cluster") ? "scope_cluster" : "scope_cta";
  std::string Space = Cluster ? "space_cluster" : "space_cta";
  unsigned AS = Cluster ? 7 : 3;
  bool Relaxed = S.has("relaxed");
  if (Op == "init") {
    need(S, 2);
    call(Intrinsic::nvvm_mbarrier_init_shared, {address(S.Ops[0], 3), read(S.Ops[1], 32)});
    return true;
  }
  if (Op == "inval") {
    need(S, 1);
    call(Intrinsic::nvvm_mbarrier_inval_shared, {address(S.Ops[0], 3)});
    return true;
  }
  if (Op == "arrive" || Op == "arrive_drop") {
    // mbarrier.arrive[.expect_tx][.sem][.scope].space.b64 state|_, [a][, n]
    bool Tx = S.has("expect_tx");
    if (S.Ops.size() < 2 || S.Ops.size() > 3)
      fail("mbarrier.arrive expects 2 or 3 operands");
    if (Tx && S.Ops.size() != 3)
      fail("mbarrier.arrive.expect_tx needs a transaction count");
    Intrinsic::ID IID = nvvm("mbarrier_" + Op + (Tx ? "_expect_tx" : "") +
                             (Relaxed ? "_relaxed" : "") + "_" + Scope + "_" + Space);
    if (IID == Intrinsic::not_intrinsic)
      unsupported(S);
    Value *Cnt = S.Ops.size() == 3 ? read(S.Ops[2], 32) : B.getInt32(1);
    Value *St = call(IID, {address(S.Ops[1], AS), Cnt});
    if (S.Ops[0].K != Opnd::Sink) {
      if (Cluster)
        fail("mbarrier.arrive on a shared::cluster barrier returns no state");
      write(S.Ops[0], St);
    }
    return true;
  }
  if (Op == "expect_tx" || Op == "complete_tx") {
    need(S, 2);
    call(nvvm("mbarrier_" + Op + "_" + Scope + "_" + Space),
         {address(S.Ops[0], AS), read(S.Ops[1], 32)});
    return true;
  }
  if (Op == "test_wait" || Op == "try_wait") {
    bool Parity = S.has("parity");
    bool Tl = Op == "try_wait" && S.Ops.size() == 4;
    if (S.Ops.size() != 3 && !Tl)
      fail("mbarrier wait expects 3 or 4 operands");
    if (Cluster)
      unsupported(S);
    Intrinsic::ID IID = nvvm("mbarrier_" + Op + (Parity ? "_parity" : "") + (Tl ? "_tl" : "") +
                             (Relaxed ? "_relaxed" : "") + "_" + Scope + "_space_cta");
    if (IID == Intrinsic::not_intrinsic)
      unsupported(S);
    SmallVector<Value *, 4> Args = {address(S.Ops[1], 3),
                                    Parity ? read(S.Ops[2], 32) : read(S.Ops[2], 64)};
    if (Tl)
      Args.push_back(read(S.Ops[3], 32));
    write(S.Ops[0], call(IID, Args));
    return true;
  }
  if (Op == "pending_count") {
    need(S, 2);
    write(S.Ops[0], call(Intrinsic::nvvm_mbarrier_pending_count, {read(S.Ops[1], 64)}));
    return true;
  }
  unsupported(S);
}

bool PTXProgram::emitBulk(const Stmt &S) {
  const std::string &R = S.Root;
  if (R == "st" && S.has("async")) {
    // st.async[.weak].shared::cluster.mbarrier::complete_tx::bytes[.vN].T [a], v, [mbar]
    need(S, 3);
    PTy T = type(S);
    unsigned N = S.has("v2") ? 2 : S.has("v4") ? 4 : 1;
    Value *A = address(S.Ops[0], 7), *Bar = address(S.Ops[2], 7);
    if (N * T.Bits <= 64) {
      unsigned Bits = T.Bits * N;
      call(Intrinsic::nvvm_st_async, {A, read(S.Ops[1], Bits, T), Bar}, {intTy(Bits)});
      return true;
    }
    // 128 bits: two 64-bit halves, completing the same bytes on the barrier.
    const Opnd &V = S.Ops[1];
    if (V.K != Opnd::Vec || V.Elts.size() != N)
      fail("bad st.async vector");
    unsigned Half = N / 2;
    for (unsigned H = 0; H < 2; ++H) {
      Opnd Part;
      Part.K = Opnd::Vec;
      Part.Elts.assign(V.Elts.begin() + H * Half, V.Elts.begin() + (H + 1) * Half);
      Value *PA = B.CreateConstGEP1_64(B.getInt8Ty(), A, H * 8);
      call(Intrinsic::nvvm_st_async, {PA, read(Part, 64, T), Bar}, {intTy(64)});
    }
    return true;
  }
  if (R != "cp" || S.Mods.size() < 2 || S.Mods[0] != "async" || S.Mods[1] != "bulk")
    return false;
  bool Hint = S.has("L2::cache_hint");
  bool Multicast = S.has("multicast::cluster");
  Value *NoHint = B.getInt64(0);
  auto Flag = [&](bool F) { return B.getInt1(F); };
  if (S.has("commit_group")) {
    call(Intrinsic::nvvm_cp_async_bulk_commit_group, {});
    return true;
  }
  if (S.has("wait_group")) {
    need(S, 1);
    int64_t N = immediate(S.Ops[0], "cp.async.bulk.wait_group count");
    call(S.has("read") ? Intrinsic::nvvm_cp_async_bulk_wait_group_read
                       : Intrinsic::nvvm_cp_async_bulk_wait_group,
         {B.getInt32(N)});
    return true;
  }
  // Cache hints are dropped (they only tune L2 residency).
  if (S.has("tensor")) {
    unsigned Dim = 0;
    for (StringRef Mo : S.Mods)
      if (Mo.size() == 2 && Mo[1] == 'd' && isDigit(Mo[0]))
        Dim = Mo[0] - '0';
    if (!Dim || S.has("im2col") || S.has("gather4") || S.has("scatter4"))
      unsupported(S);
    auto Coords = [&](const Opnd &A, SmallVectorImpl<Value *> &Args) {
      if (A.K != Opnd::Addr || A.Elts.size() != 2 || A.Elts[1].K != Opnd::Vec ||
          A.Elts[1].Elts.size() != Dim)
        fail("bad tensor coordinates");
      Opnd Map = A;
      Map.Elts.resize(1);
      Args.push_back(address(Map, 0));
      for (const Opnd &C : A.Elts[1].Elts)
        Args.push_back(read(C, 32));
    };
    // cp.async.bulk.tensor.Nd.<dst>.<src>...
    StringRef TDst = S.Mods.size() > 4 ? StringRef(S.Mods[4]) : "";
    StringRef TSrc = S.Mods.size() > 5 ? StringRef(S.Mods[5]) : "";
    if (TDst == "shared::cluster" && TSrc == "global") {
      // global -> shared::cluster: [dst], [map, {c...}], [mbar][, mask][, hint]
      need(S, 3 + Multicast + Hint);
      // dst, mbar, map, coords..., cta_mask, cache_hint, flags, cta_group
      SmallVector<Value *, 12> Args = {address(S.Ops[0], 7), address(S.Ops[2], 3)};
      Coords(S.Ops[1], Args);
      Args.push_back(Multicast ? read(S.Ops[3], 16) : B.getInt16(0));
      Args.push_back(NoHint);
      Args.append({Flag(Multicast), Flag(false), B.getInt32(0)});
      call(nvvm("cp_async_bulk_tensor_g2s_tile_" + std::to_string(Dim) + "d"), Args);
      return true;
    }
    if (TDst == "global" && TSrc == "shared" && S.has("bulk_group")) {
      // shared::cta -> global: [map, {c...}], [src][, hint]
      need(S, 2 + Hint);
      SmallVector<Value *, 8> Args = {address(S.Ops[1], 3)};
      Coords(S.Ops[0], Args);
      Args.push_back(NoHint);
      Args.push_back(Flag(false));
      call(nvvm("cp_async_bulk_tensor_s2g_tile_" + std::to_string(Dim) + "d"), Args);
      return true;
    }
    unsupported(S);
  }
  if (S.has("prefetch")) {
    // cp.async.bulk.prefetch.L2.global[.L2::cache_hint] [a], size[, hint]
    need(S, 2 + Hint);
    call(Intrinsic::nvvm_cp_async_bulk_prefetch_L2,
         {address(S.Ops[0], 1), read(S.Ops[1], 32), NoHint, Flag(false)});
    return true;
  }
  // Destination and source spaces follow "bulk".
  StringRef Dst = S.Mods.size() > 2 ? StringRef(S.Mods[2]) : "";
  StringRef Src = S.Mods.size() > 3 ? StringRef(S.Mods[3]) : "";
  if ((Dst == "shared::cluster" || Dst == "shared") && Src == "global") {
    // [dst], [src], size, [mbar][, mask][, hint]
    need(S, 4 + Multicast + Hint);
    if (Dst == "shared" && !Multicast) {
      call(Intrinsic::nvvm_cp_async_bulk_global_to_shared_cta,
           {address(S.Ops[0], 3), address(S.Ops[3], 3), address(S.Ops[1], 1),
            read(S.Ops[2], 32), NoHint, Flag(false)});
      return true;
    }
    call(Intrinsic::nvvm_cp_async_bulk_global_to_shared_cluster,
         {address(S.Ops[0], 7), address(S.Ops[3], 3), address(S.Ops[1], 1), read(S.Ops[2], 32),
          Multicast ? read(S.Ops[4], 16) : B.getInt16(0), NoHint, Flag(Multicast), Flag(false)});
    return true;
  }
  if (Dst == "global" && Src == "shared") {
    need(S, 3 + Hint);
    call(Intrinsic::nvvm_cp_async_bulk_shared_cta_to_global,
         {address(S.Ops[0], 1), address(S.Ops[1], 3), read(S.Ops[2], 32), NoHint, Flag(false)});
    return true;
  }
  if (Dst == "shared::cluster" && Src == "shared") {
    need(S, 4);
    call(Intrinsic::nvvm_cp_async_bulk_shared_cta_to_cluster,
         {address(S.Ops[0], 7), address(S.Ops[3], 3), address(S.Ops[1], 3), read(S.Ops[2], 32)});
    return true;
  }
  unsupported(S);
}

void PTXProgram::emitLeaf(const Stmt &S, ArrayRef<int> Dirs, ArrayRef<char> Cons) {
  // Operands are renumbered: outputs (and in-outs) first, then inputs, then
  // the inputs tied to the in-outs.
  struct Ref {
    const Opnd *O;
    char C;
    int Dir;
  };
  SmallVector<Ref, 16> Refs;
  auto Collect = [&](const Opnd &O, char C, int Dir) {
    if (O.K == Opnd::Vec) {
      for (const Opnd &E : O.Elts)
        Refs.push_back({&E, C, Dir});
    } else if (O.K == Opnd::Addr) {
      Refs.push_back({&O.Elts[0], 'r', In});
    } else if (O.K == Opnd::Reg) {
      Refs.push_back({&O, C, Dir});
    }
  };
  for (auto [I, O] : enumerate(S.Ops))
    if (Dirs[I] != In)
      Collect(O, Cons[I], Dirs[I]);
  for (auto [I, O] : enumerate(S.Ops))
    if (Dirs[I] == In)
      Collect(O, Cons[I], In);
  std::string Text = S.Opc + " ";
  DenseMap<const Opnd *, unsigned> Num;
  for (auto [I, R] : enumerate(Refs))
    Num[R.O] = I;
  auto Render = [&](const Opnd &O) -> std::string {
    switch (O.K) {
    case Opnd::Reg:
      return "$" + std::to_string(Num.lookup(&O));
    case Opnd::Imm:
      return std::to_string(O.Int);
    default:
      fail("unexpected operand in " + S.Opc);
    }
  };
  for (auto [I, O] : enumerate(S.Ops)) {
    if (I)
      Text += ", ";
    if (O.K == Opnd::Vec) {
      Text += "{";
      for (auto [J, E] : enumerate(O.Elts))
        Text += (J ? ", " : "") + Render(E);
      Text += "}";
    } else if (O.K == Opnd::Addr) {
      Text += "[" + Render(O.Elts[0]) + (O.Int ? "+" + std::to_string(O.Int) : "") + "]";
    } else {
      Text += Render(O);
    }
  }
  Text += ";";
  // Constraint letters and IR types; predicates travel as 0/1 in "r".
  SmallVector<Type *, 8> OutTys;
  SmallVector<Value *, 8> Args;
  SmallVector<Type *, 8> ArgTys;
  std::string Cs;
  auto TyOf = [&](char C, const Opnd &O) -> Type * {
    if (C == 'f')
      return B.getFloatTy();
    if (C == 'l')
      return B.getInt64Ty();
    if (O.K == Opnd::Reg)
      if (Var *V = lookup(O.Name); V && V->Ty->getBitWidth() == 64)
        return B.getInt64Ty();
    return B.getInt32Ty();
  };
  auto AddIn = [&](const Ref &R, std::string C) {
    Type *Ty = TyOf(R.C, *R.O);
    Cs += (Cs.empty() ? "" : ",") + C;
    Value *V = read(*R.O, DL.getTypeSizeInBits(Ty));
    Args.push_back(Ty->isFloatTy() ? B.CreateBitCast(V, Ty) : V);
    ArgTys.push_back(Ty);
  };
  for (const Ref &R : Refs) {
    Type *Ty = TyOf(R.C, *R.O);
    char C = Ty->isIntegerTy(64) ? 'l' : R.C;
    if (R.Dir != In) {
      Cs += (Cs.empty() ? "=" : ",=") + std::string(1, C);
      OutTys.push_back(Ty);
    } else {
      AddIn(R, std::string(1, C));
    }
  }
  for (auto [I, R] : enumerate(Refs))
    if (R.Dir == InOut)
      AddIn(R, std::to_string(I));
  Cs += Cs.empty() ? "~{memory}" : ",~{memory}";
  Type *RetTy = OutTys.empty() ? B.getVoidTy()
                : OutTys.size() == 1 ? OutTys[0]
                                     : (Type *)StructType::get(Ctx, OutTys);
  auto *IA = InlineAsm::get(FunctionType::get(RetTy, ArgTys, false), Text, Cs,
                            /*hasSideEffects=*/true);
  Value *Res = B.CreateCall(IA, Args);
  unsigned K = 0;
  for (const Ref &R : Refs) {
    if (R.Dir == In)
      continue;
    Value *V = OutTys.size() == 1 ? Res : B.CreateExtractValue(Res, K);
    ++K;
    if (!V->getType()->isIntegerTy())
      V = B.CreateBitCast(V, intTy(32));
    write(*R.O, V);
  }
}

bool PTXProgram::emitMatrix(const Stmt &S) {
  const std::string &R = S.Root;
  if (R == "ldmatrix" || R == "stmatrix") {
    need(S, 2);
    bool Ld = R == "ldmatrix";
    emitLeaf(S, {Ld ? Out : In, In}, {'r', 'r'});
    return true;
  }
  if (R == "cp" && S.Mods.size() >= 2 && S.Mods[0] == "async" && S.Mods[1] != "bulk" &&
      S.Mods[1] != "mbarrier") {
    // cp.async.{ca,cg}..., commit_group, wait_group N, wait_all
    SmallVector<int, 4> Dirs(S.Ops.size(), In);
    SmallVector<char, 4> Cons(S.Ops.size(), 'r');
    emitLeaf(S, Dirs, Cons);
    return true;
  }
  if ((R == "ex2" || R == "rcp") && S.has("approx") && S.has("ftz") && S.has("f32")) {
    need(S, 2);
    emitLeaf(S, {Out, In}, {'f', 'f'});
    return true;
  }
  if (R != "mma")
    return false;
  need(S, 4);
  StringRef Shape = S.Mods.size() > 2 ? StringRef(S.Mods[2]) : "";
  if (Shape == "m16n8k16") {
    emitLeaf(S, {Out, In, In, In}, {'f', 'r', 'r', 'f'});
    return true;
  }
  // m16n8k32 FP8 (sm_89 PTX): sm_90 has no FP8 mma.sync. Each A/B register
  // holds four FP8 values; converting its low and high pairs to f16x2 gives
  // the fragments of two k16 MMAs over the k-halves, accumulated in order.
  if (Shape == "m16n8k32" && S.Mods.size() == 9 && S.Mods[3] == "row" && S.Mods[4] == "col" &&
      S.Mods[5] == "f32" && S.Mods[8] == "f32" &&
      (S.Mods[6] == "e4m3" || S.Mods[6] == "e5m2") &&
      (S.Mods[7] == "e4m3" || S.Mods[7] == "e5m2")) {
    const Opnd &D = S.Ops[0], &A = S.Ops[1], &Bo = S.Ops[2], &C = S.Ops[3];
    if (D.Elts.size() != 4 || A.Elts.size() != 4 || Bo.Elts.size() != 2 || C.Elts.size() != 4)
      fail("bad m16n8k32 fragments");
    auto Unpack = [&](const Opnd &O, StringRef Ty, bool Hi) {
      Value *V = read(O, 32);
      if (Hi)
        V = B.CreateLShr(V, 16);
      Value *P = call(Intrinsic::lookupIntrinsicID("llvm.nvvm." + Ty.str() + "x2.to.f16x2.rn"),
                      {B.CreateTrunc(V, B.getInt16Ty())});
      return B.CreateBitCast(P, B.getInt32Ty());
    };
    // Temporaries for the halves and the intermediate accumulator.
    SmallVector<std::string, 16> Tmp;
    StringMap<Var> &Sc = Scopes.back();
    auto Temp = [&](Value *V) {
      std::string N = "%ptx.mma" + std::to_string(Sc.size());
      Var Vr;
      Vr.Ty = intTy(32);
      Vr.Slot = newSlot(Vr.Ty, N);
      B.CreateStore(V, Vr.Slot);
      Sc[N] = Vr;
      Opnd O;
      O.Name = N;
      return O;
    };
    Opnd D1;
    D1.K = Opnd::Vec;
    for (unsigned I = 0; I < 4; ++I)
      D1.Elts.push_back(Temp(B.getInt32(0)));
    for (bool Hi : {false, true}) {
      Stmt Half;
      Half.Opc = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32";
      Half.Root = "mma";
      Half.Text = S.Text;
      Opnd Av, Bv;
      Av.K = Bv.K = Opnd::Vec;
      for (const Opnd &E : A.Elts)
        Av.Elts.push_back(Temp(Unpack(E, S.Mods[6], Hi)));
      for (const Opnd &E : Bo.Elts)
        Bv.Elts.push_back(Temp(Unpack(E, S.Mods[7], Hi)));
      Half.Ops = {Hi ? D : D1, Av, Bv, Hi ? D1 : C};
      emitLeaf(Half, {Out, In, In, In}, {'f', 'r', 'r', 'f'});
    }
    return true;
  }
  unsupported(S);
}

bool PTXProgram::emitMisc(const Stmt &S) {
  const std::string &R = S.Root;
  if (R == "prefetch" || R == "prefetchu") {
    need(S, 1);
    if (S.has("tensormap")) {
      call(Intrinsic::nvvm_prefetch_tensormap, {address(S.Ops[0], 0)}, {B.getPtrTy(0)});
      return true;
    }
    bool Global = S.has("global"), L1 = S.has("L1");
    if (R == "prefetchu") {
      call(Intrinsic::nvvm_prefetchu_L1, {address(S.Ops[0], 0)});
    } else if (S.has("L2::evict_last")) {
      call(Intrinsic::nvvm_prefetch_global_L2_evict_last, {address(S.Ops[0], 1)});
    } else if (S.has("L2::evict_normal")) {
      call(Intrinsic::nvvm_prefetch_global_L2_evict_normal, {address(S.Ops[0], 1)});
    } else if (S.has("local")) {
      call(L1 ? Intrinsic::nvvm_prefetch_local_L1 : Intrinsic::nvvm_prefetch_local_L2,
           {address(S.Ops[0], 5)});
    } else {
      call(Global ? (L1 ? Intrinsic::nvvm_prefetch_global_L1 : Intrinsic::nvvm_prefetch_global_L2)
                  : (L1 ? Intrinsic::nvvm_prefetch_L1 : Intrinsic::nvvm_prefetch_L2),
           {address(S.Ops[0], Global ? 1 : 0)});
    }
    return true;
  }
  if (R == "trap") {
    call(Intrinsic::trap, {});
    return true;
  }
  if (R == "nanosleep") {
    need(S, 1);
    call(Intrinsic::nvvm_nanosleep, {read(S.Ops[0], 32)});
    return true;
  }
  if (R == "setmaxnreg") {
    need(S, 1);
    int64_t N = immediate(S.Ops[0], "setmaxnreg count");
    call(S.has("inc") ? Intrinsic::nvvm_setmaxnreg_inc_sync_aligned_u32
                      : Intrinsic::nvvm_setmaxnreg_dec_sync_aligned_u32,
         {B.getInt32(N)});
    return true;
  }
  if (R == "elect") {
    need(S, 2);
    Value *Res = call(Intrinsic::nvvm_elect_sync, {read(S.Ops[1], 32)});
    write(S.Ops[0], B.CreateExtractValue(Res, 0));
    if (!S.Ops[0].Name2.empty())
      write(regOp(S.Ops[0].Name2), B.CreateExtractValue(Res, 1));
    return true;
  }
  if (R == "mapa") {
    // Shared windows are (CTA rank << 24) + offset (SASSISD::SMEM_BASE):
    // the cluster address of a CTA's variable swaps in the target rank.
    need(S, 3);
    if (!S.has("shared::cluster") || !S.has("u32"))
      unsupported(S);
    Value *A = read(S.Ops[1], 32), *Rank = read(S.Ops[2], 32);
    write(S.Ops[0], B.CreateOr(B.CreateAnd(A, B.getInt32(0xffffff)), B.CreateShl(Rank, 24)));
    return true;
  }
  if (R == "createpolicy") {
    // Policies only feed .L2::cache_hint operands, which are dropped.
    write(S.Ops[0], B.getInt64(0));
    return true;
  }
  if (R == "discard" || R == "applypriority")
    return true; // Hints without architectural effect on values.
  if (R == "wgmma") {
    if (S.has("fence")) {
      call(Intrinsic::nvvm_wgmma_fence_sync_aligned, {});
      return true;
    }
    if (S.has("commit_group")) {
      call(Intrinsic::nvvm_wgmma_commit_group_sync_aligned, {});
      return true;
    }
    if (S.has("wait_group")) {
      need(S, 1);
      call(Intrinsic::nvvm_wgmma_wait_group_sync_aligned,
           {B.getInt64(immediate(S.Ops[0], "wgmma.wait_group count"))});
      return true;
    }
    if (S.has("mma_async")) {
      // Canonical one-instruction form for the wgmma selection:
      //   d (in-out, =f or =r), a (desc "l" or {regs} "r"), b (desc "l"),
      //   scale-d ("r", 0 or 1, or an immediate), immediates...
      if (S.Ops.size() < 4 || S.Ops[0].K != Opnd::Vec)
        fail("bad wgmma.mma_async operands");
      bool F32 = S.Mods.size() > 4 && S.Mods[4] == "f32";
      SmallVector<int, 9> Dirs(S.Ops.size(), In);
      SmallVector<char, 9> Cons(S.Ops.size(), 'r');
      Dirs[0] = InOut;
      Cons[0] = F32 ? 'f' : 'r';
      Cons[1] = S.Ops[1].K == Opnd::Vec ? 'r' : 'l';
      Cons[2] = 'l';
      emitLeaf(S, Dirs, Cons);
      return true;
    }
    return false;
  }
  if (R == "activemask") {
    need(S, 1);
    write(S.Ops[0], call(Intrinsic::nvvm_activemask, {}));
    return true;
  }
  return false;
}

/// Expands `${:uid}` and `$$` in an LLVM asm string.
static std::string expandAsm(StringRef Asm, unsigned Uid) {
  std::string Out;
  for (size_t I = 0; I < Asm.size(); ++I) {
    if (Asm.substr(I).starts_with("${:uid}")) {
      Out += std::to_string(Uid);
      I += 6;
    } else if (Asm.substr(I).starts_with("$$")) {
      Out += '$';
      ++I;
    } else {
      Out += Asm[I];
    }
  }
  return Out;
}

bool SASSExpandInlinePTX::runOnFunction(Function &F) {
  SmallVector<CallInst *, 16> Calls;
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I); CI && CI->isInlineAsm())
      Calls.push_back(CI);
  if (Calls.empty())
    return false;
  SmallVector<AllocaInst *, 32> Slots;
  bool Changed = false;
  unsigned Uid = 0;
  for (CallInst *CI : Calls) {
    StringRef Raw = cast<InlineAsm>(CI->getCalledOperand())->getAsmString();
    std::string Asm = expandAsm(Raw, Uid++);
    std::vector<Item> Items;
    std::string Err = parseProgram(Asm, Items);
    if (!Err.empty())
      report_fatal_error("SASS: inline PTX in " + F.getName() + ": " + Err + "\n  asm: \"" +
                         Raw + "\"");
    // An empty template (asm volatile("" ::: "memory")) is a compiler
    // barrier: a single-thread fence.
    // With operands ("+r"(x)) it also passes its tied inputs through.
    if (Items.empty()) {
      new FenceInst(F.getContext(), AtomicOrdering::SequentiallyConsistent,
                    SyncScope::SingleThread, CI->getIterator());
      Changed = true;
      if (CI->getType()->isVoidTy() && CI->arg_empty()) {
        CI->eraseFromParent();
        continue;
      }
    }
    // One unguarded instruction SASSLowerInlinePTX implements: leave it.
    if (Items.size() == 1 && Items[0].K == Item::Inst && !Items[0].S.Guarded) {
      const Stmt &S = Items[0].S;
      static const StringSet<> Leaves = {"ldmatrix", "stmatrix"};
      bool Leaf = Leaves.contains(S.Root) ||
                  (S.Root == "mma" && S.Mods.size() > 2 && S.Mods[2] == "m16n8k16") ||
                  (S.Root == "cp" && S.Mods.size() >= 2 && S.Mods[0] == "async" &&
                   S.Mods[1] != "bulk" && S.Mods[1] != "mbarrier") ||
                  ((S.Root == "ex2" || S.Root == "rcp") && StringRef(S.Opc).ends_with(".approx.ftz.f32"));
      if (Leaf)
        continue;
    }
    PTXProgram(*CI, Slots).run(Items);
    Changed = true;
  }
  if (Changed) {
    removeUnreachableBlocks(F);
    // Slots in removed blocks are gone; promote the rest.
    SmallVector<AllocaInst *, 32> Live;
    for (AllocaInst *A : Slots)
      if (isAllocaPromotable(A))
        Live.push_back(A);
    DominatorTree DT(F);
    PromoteMemToReg(Live, DT);
    for (AllocaInst *A : Slots)
      if (!is_contained(Live, A))
        report_fatal_error("SASS: inline PTX in " + F.getName() +
                           ": a register could not be promoted");
  }
  return Changed;
}
