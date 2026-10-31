//===-- SASSLiftIdioms.cpp - Scalar idioms to SIMD/warp instructions -------===//
//
// Recognizes scalar code that one specialized instruction computes, the way
// a vectorizer recognizes lanes in scalar code, and rewrites it as the NVVM
// intrinsic the instruction selector implements:
//
// * A sum of the four byte-wise products of two 32-bit words (each byte
//   extracted by shifts, masks or casts, all signed or all unsigned per
//   word) is IDP.4A: `acc + sum_k byte_k(a) * byte_k(b)`, as
//   llvm.nvvm.idp4a.{s,u}.{s,u}. Wrapping i32 addition makes the regrouped
//   sum exact.
// * A butterfly all-reduce over the full warp,
//   `v = op(v, shfl.sync.bfly(-1, v, m, 31))` for m = 16, 8, 4, 2, 1 in any
//   order, leaves every lane with op over all 32 lanes: for integer add,
//   min, max, and, or and xor that is REDUX (llvm.nvvm.redux.sync.*), one
//   instruction instead of ten. Both require all 32 lanes, so the rewrite
//   adds no requirement.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
class SASSLiftIdioms : public FunctionPass {
public:
  static char ID;
  SASSLiftIdioms() : FunctionPass(ID) {}
  StringRef getPassName() const override { return "SASS lift idioms"; }
  bool runOnFunction(Function &F) override;
};

/// Byte K of a 32-bit word, sign- or zero-extended to i32.
struct Byte {
  Value *Word = nullptr;
  unsigned K = 0;
  bool Signed = false;
};

/// One `byte_k(a) * byte_k(b)` term.
struct Product {
  Byte A, B;
};
} // namespace

char SASSLiftIdioms::ID = 0;

INITIALIZE_PASS(SASSLiftIdioms, "sass-lift-idioms", "SASS lift idioms", false,
                false)

static bool isI32(const Value *V) { return V->getType()->isIntegerTy(32); }

/// \p V as an extended byte of an i32 word: sext/zext of an i8 truncation or
/// <4 x i8> element, `ashr (shl w, 24 - 8k), 24`, `and (lshr w, 8k), 255`,
/// `lshr w, 24`.
static std::optional<Byte> matchByte(Value *V) {
  Value *X, *W;
  uint64_t S = 0, T = 0;
  bool Signed = isa<SExtInst>(V);
  if (match(V, m_CombineOr(m_SExt(m_Value(X)), m_ZExt(m_Value(X)))) &&
      X->getType()->isIntegerTy(8)) {
    // trunc (shr w, 8k) / trunc w.
    if (match(X, m_Trunc(m_Value(W))) && isI32(W)) {
      if (match(W, m_Shr(m_Value(X), m_ConstantInt(S))) && S % 8 == 0 && S < 32)
        return Byte{X, unsigned(S / 8), Signed};
      return Byte{W, 0, Signed};
    }
    // extractelement (bitcast w to <4 x i8>), k: little-endian bytes.
    ConstantInt *Idx;
    if (match(X, m_ExtractElt(m_Value(W), m_ConstantInt(Idx))) &&
        cast<VectorType>(W->getType())->getElementCount().getFixedValue() == 4) {
      Value *Word;
      if (match(W, m_BitCast(m_Value(Word))) && isI32(Word))
        return Byte{Word, unsigned(Idx->getZExtValue()), Signed};
      return Byte{W, unsigned(Idx->getZExtValue()), Signed};
    }
    return std::nullopt;
  }
  if (!isI32(V))
    return std::nullopt;
  // Signed: ashr (shl w, 24 - 8k), 24 or ashr w, 24.
  if (match(V, m_AShr(m_Value(X), m_SpecificInt(24)))) {
    if (match(X, m_Shl(m_Value(W), m_ConstantInt(S)))) {
      if (S % 8 || S > 24)
        return std::nullopt;
      return Byte{W, unsigned(3 - S / 8), true};
    }
    return Byte{X, 3, true};
  }
  // Unsigned: and (lshr w, 8k), 255, and w, 255, lshr w, 24.
  if (match(V, m_And(m_Value(X), m_SpecificInt(255)))) {
    if (match(X, m_LShr(m_Value(W), m_ConstantInt(T))) && T % 8 == 0 && T < 24)
      return Byte{W, unsigned(T / 8), false};
    return Byte{X, 0, false};
  }
  if (match(V, m_LShr(m_Value(X), m_SpecificInt(24))))
    return Byte{X, 3, false};
  return std::nullopt;
}

/// \p V as a sum of bytes, all of index K: appends them to \p Out.
static bool matchByteSum(Value *V, SmallVectorImpl<Byte> &Out) {
  Value *L, *R;
  if (V->hasOneUse() && match(V, m_Add(m_Value(L), m_Value(R))))
    return matchByteSum(L, Out) && matchByteSum(R, Out);
  auto B = matchByte(V);
  if (!B || (!Out.empty() && Out.front().K != B->K))
    return false;
  Out.push_back(*B);
  return true;
}

/// \p V as a sum of byte products `byte_k(a) * byte_k(b)`: a product of two
/// byte sums distributes (LLVM factors `a*b + a*c` into `a*(b + c)`).
static bool matchProducts(Value *V, SmallVectorImpl<Product> &Out) {
  Value *L, *R;
  SmallVector<Byte, 4> Ls, Rs;
  if (!match(V, m_Mul(m_Value(L), m_Value(R))) || !matchByteSum(L, Ls) ||
      !matchByteSum(R, Rs) || Ls.front().K != Rs.front().K)
    return false;
  for (const Byte &A : Ls)
    for (const Byte &B : Rs)
      Out.push_back({A, B});
  return true;
}

/// The leaves of the single-use add tree under \p Root.
static void addLeaves(Value *V, Value *Root, SmallVectorImpl<Value *> &Out) {
  Value *L, *R;
  if ((V == Root || V->hasOneUse()) && match(V, m_Add(m_Value(L), m_Value(R)))) {
    addLeaves(L, Root, Out);
    addLeaves(R, Root, Out);
    return;
  }
  Out.push_back(V);
}

/// Rewrites the add tree \p Root as IDP.4A over every complete set of four
/// byte products; true if it did.
static bool liftDot(BinaryOperator &Root) {
  SmallVector<Value *, 16> Leaves;
  addLeaves(&Root, &Root, Leaves);
  if (Leaves.size() < 4)
    return false;

  // Group the products by (word a, word b, signedness), with the pair
  // ordered so each group has one canonical key; a group is complete with
  // one product per byte index.
  struct Group {
    Byte A, B;
    SmallVector<unsigned, 4> Leaf; // index into Leaves, by byte
  };
  SmallVector<Group, 4> Groups;
  SmallVector<unsigned, 16> Terms(Leaves.size(), 0); // products per leaf
  for (unsigned I = 0; I < Leaves.size(); ++I) {
    SmallVector<Product, 4> Ps;
    if (!matchProducts(Leaves[I], Ps))
      continue;
    Terms[I] = Ps.size();
    for (Product &P : Ps) {
      if (P.B.Word < P.A.Word)
        std::swap(P.A, P.B);
      auto *G = find_if(Groups, [&](const Group &G) {
        return G.A.Word == P.A.Word && G.B.Word == P.B.Word &&
               G.A.Signed == P.A.Signed && G.B.Signed == P.B.Signed;
      });
      if (G == Groups.end()) {
        Groups.push_back({P.A, P.B, {~0u, ~0u, ~0u, ~0u}});
        G = &Groups.back();
      }
      if (G->Leaf[P.A.K] != ~0u)
        Terms[I] = 0; // A repeated term: leave this leaf alone.
      G->Leaf[P.A.K] = I;
    }
  }
  // A leaf is replaced only if every one of its products is in a complete
  // group; drop groups holding a product of another leaf until stable.
  for (bool Again = true; Again;) {
    Again = false;
    SmallVector<unsigned, 16> Covered(Leaves.size(), 0);
    for (const Group &G : Groups)
      if (!is_contained(G.Leaf, ~0u))
        for (unsigned I : G.Leaf)
          ++Covered[I];
    for (Group &G : Groups)
      if (!is_contained(G.Leaf, ~0u) &&
          any_of(G.Leaf, [&](unsigned I) { return Covered[I] != Terms[I]; })) {
        G.Leaf[0] = ~0u;
        Again = true;
      }
  }
  erase_if(Groups, [](const Group &G) { return is_contained(G.Leaf, ~0u); });
  if (Groups.empty())
    return false;
  SmallSet<unsigned, 16> Used;
  for (const Group &G : Groups)
    for (unsigned I : G.Leaf)
      Used.insert(I);

  // The remaining terms seed the accumulator of the first dot product.
  IRBuilder<> B(&Root);
  Value *Acc = nullptr;
  for (unsigned I = 0; I < Leaves.size(); ++I)
    if (!Used.contains(I))
      Acc = Acc ? B.CreateAdd(Acc, Leaves[I]) : Leaves[I];
  auto Word = [&](const Byte &X) {
    return isI32(X.Word) ? X.Word : B.CreateBitCast(X.Word, B.getInt32Ty());
  };
  for (const Group &G : Groups) {
    Intrinsic::ID ID = G.A.Signed ? (G.B.Signed ? Intrinsic::nvvm_idp4a_s_s
                                                : Intrinsic::nvvm_idp4a_s_u)
                                  : (G.B.Signed ? Intrinsic::nvvm_idp4a_u_s
                                                : Intrinsic::nvvm_idp4a_u_u);
    Acc = B.CreateIntrinsic(ID, {}, {Word(G.A), Word(G.B), Acc ? Acc : B.getInt32(0)});
  }
  Acc->takeName(&Root);
  Root.replaceAllUsesWith(Acc);
  RecursivelyDeleteTriviallyDeadInstructions(&Root);
  return true;
}

/// One butterfly step `op(x, shfl.sync.bfly(-1, x, m, 31))`: x and m.
static bool matchStep(Value *V, Intrinsic::ID &Op, Value *&X, unsigned &M) {
  auto Shfl = [&](Value *S, Value *Src) {
    auto *C = dyn_cast<IntrinsicInst>(S);
    if (!C || C->getIntrinsicID() != Intrinsic::nvvm_shfl_sync_bfly_i32 ||
        C->getArgOperand(1) != Src || !match(C->getArgOperand(0), m_AllOnes()) ||
        !match(C->getArgOperand(3), m_SpecificInt(31)))
      return false;
    auto *Lane = dyn_cast<ConstantInt>(C->getArgOperand(2));
    if (!Lane || !isPowerOf2_64(Lane->getZExtValue()) || Lane->getZExtValue() > 16)
      return false;
    M = Lane->getZExtValue();
    return true;
  };
  Value *L, *R;
  Intrinsic::ID Kind;
  if (auto *BO = dyn_cast<BinaryOperator>(V)) {
    switch (BO->getOpcode()) {
    case Instruction::Add: Kind = Intrinsic::nvvm_redux_sync_add; break;
    case Instruction::And: Kind = Intrinsic::nvvm_redux_sync_and; break;
    case Instruction::Or: Kind = Intrinsic::nvvm_redux_sync_or; break;
    case Instruction::Xor: Kind = Intrinsic::nvvm_redux_sync_xor; break;
    default: return false;
    }
    L = BO->getOperand(0);
    R = BO->getOperand(1);
  } else if (auto *MM = dyn_cast<MinMaxIntrinsic>(V)) {
    switch (MM->getIntrinsicID()) {
    case Intrinsic::smin: Kind = Intrinsic::nvvm_redux_sync_min; break;
    case Intrinsic::smax: Kind = Intrinsic::nvvm_redux_sync_max; break;
    case Intrinsic::umin: Kind = Intrinsic::nvvm_redux_sync_umin; break;
    case Intrinsic::umax: Kind = Intrinsic::nvvm_redux_sync_umax; break;
    default: return false;
    }
    L = MM->getLHS();
    R = MM->getRHS();
  } else {
    return false;
  }
  if (!isI32(V) || (Op != Intrinsic::not_intrinsic && Op != Kind))
    return false;
  Op = Kind;
  if (Shfl(R, L)) {
    X = L;
    return true;
  }
  if (Shfl(L, R)) {
    X = R;
    return true;
  }
  return false;
}

/// Rewrites a full-warp butterfly reduction ending at \p Root as REDUX.
static bool liftReduction(Instruction &Root) {
  Intrinsic::ID Op = Intrinsic::not_intrinsic;
  Value *X = &Root;
  unsigned Seen = 0, M;
  SmallVector<Instruction *, 5> Steps;
  for (unsigned Step = 0; Step < 5; ++Step) {
    Value *Next;
    if (!matchStep(X, Op, Next, M) || (Seen & M))
      return false;
    Seen |= M;
    Steps.push_back(cast<Instruction>(X));
    X = Next;
  }
  IRBuilder<> B(&Root);
  Value *R = B.CreateIntrinsic(Op, {}, {X, B.getInt32(-1)});
  R->takeName(&Root);
  Root.replaceAllUsesWith(R);
  // Shuffles are not trivially dead (they synchronize the warp, which REDUX
  // does too, and order no memory): erase the unused steps outside in.
  for (Instruction *S : Steps) {
    if (!S->use_empty())
      break;
    SmallVector<Instruction *, 2> Ops;
    for (Value *O : S->operands())
      if (auto *Shfl = dyn_cast<IntrinsicInst>(O);
          Shfl && Shfl->getIntrinsicID() == Intrinsic::nvvm_shfl_sync_bfly_i32)
        Ops.push_back(Shfl);
    S->eraseFromParent();
    for (Instruction *Shfl : Ops)
      if (Shfl->use_empty())
        Shfl->eraseFromParent();
  }
  return true;
}

bool SASSLiftIdioms::runOnFunction(Function &F) {
  bool Changed = false;
  // Roots last-to-first: an outer tree consumes its inner trees' values.
  SmallVector<WeakTrackingVH, 32> Work;
  for (Instruction &I : instructions(F))
    if (isI32(&I) && (isa<BinaryOperator>(I) || isa<MinMaxIntrinsic>(I)))
      Work.push_back(&I);
  for (WeakTrackingVH &H : reverse(Work)) {
    auto *I = dyn_cast_or_null<Instruction>(H);
    if (!I)
      continue;
    if (liftReduction(*I)) {
      Changed = true;
      continue;
    }
    // Only add-tree roots: a single-use add feeding another add is a leaf
    // of its parent's tree.
    auto *Add = dyn_cast<BinaryOperator>(I);
    if (!Add || Add->getOpcode() != Instruction::Add ||
        (Add->hasOneUse() && match(Add->user_back(), m_Add(m_Value(), m_Value()))))
      continue;
    Changed |= liftDot(*Add);
  }
  return Changed;
}

FunctionPass *llvm::createSASSLiftIdiomsPass() { return new SASSLiftIdioms(); }
