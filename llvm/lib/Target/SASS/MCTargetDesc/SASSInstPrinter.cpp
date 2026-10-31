//===-- SASSInstPrinter.cpp - Print SASS in sass-dec dialect --------------===//

#include "SASSInstPrinter.h"
#include "SASSBaseInfo.h"
#include "SASSGMMAFlags.h"
#include "SASSMCTargetDesc.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "SASSGenAsmWriter.inc"

std::string SASSInstPrinter::labelName(StringRef SymName) {
  std::string Out;
  for (char C : SymName.ltrim('.'))
    Out += isAlnum(C) ? C : '_';
  return Out;
}

void SASSInstPrinter::printRegName(raw_ostream &OS, MCRegister Reg) {
  OS << getRegisterName(Reg);
}

void SASSInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                StringRef Annot, const MCSubtargetInfo &STI,
                                raw_ostream &O) {
  int P = MII.get(MI->getOpcode()).findFirstPredOperandIdx();
  if (P >= 0) {
    MCRegister Guard = MI->getOperand(P).getReg();
    bool Neg = MI->getOperand(P + 1).getImm();
    if (Guard != SASS::PT || Neg)
      O << '@' << (Neg ? "!" : "") << getRegisterName(Guard) << ' ';
  }
  std::string Text;
  raw_string_ostream TOS(Text);
  printInstruction(MI, Address, TOS);
  if (MI->getFlags() & SASS::InstFlagFTZ) {
    // .FTZ follows the opcode and its rounding/condition modifiers but
    // precedes the predicate combiner and .SAT: FFMA.FTZ.SAT,
    // FSETP.GEU.FTZ.AND.
    StringRef Op = StringRef(Text).ltrim();
    Op = Op.take_until([](char C) { return C == ' '; });
    size_t At = Op.rfind(".AND");
    if (At == StringRef::npos)
      At = Op.find(".SAT");
    if (At == StringRef::npos)
      At = Op.size();
    Text.insert(Op.data() - Text.data() + At, ".FTZ");
  }
  O << StringRef(Text).ltrim();
  printAnnotation(O, Annot);
}

void SASSInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg())
    O << getRegisterName(Op.getReg());
  else if (Op.isImm())
    printImm(MI, OpNo, O);
  else
    MAI.printExpr(O, *Op.getExpr());
}

void SASSInstPrinter::printGroupReg(MCRegister Reg, raw_ostream &O,
                                    unsigned Count) {
  // Tuples have only 32-bit leaves (sub0, sub1, ...), contiguous from the
  // base register's encoding.
  SmallVector<MCRegister, 64> Subs;
  for (MCRegister Sub : MRI.subregs(Reg))
    if (MRI.subregs(Sub).empty())
      Subs.push_back(Sub);
  llvm::sort(Subs, [&](MCRegister A, MCRegister B) {
    return MRI.getEncodingValue(A) < MRI.getEncodingValue(B);
  });
  if (Count && Count < Subs.size())
    Subs.resize(Count);
  O << '{';
  for (auto [I, Sub] : enumerate(Subs))
    O << (I ? "," : "") << getRegisterName(Sub);
  O << '}';
}

void SASSInstPrinter::printGMMAFlags(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  using namespace SASS::GMMA;
  static const char *const FmtName[] = {"F16", "BF16", "TF32", "E4M3", "E5M2"};
  unsigned F = MI->getOperand(OpNo).getImm();
  O << (isQ(F) ? "QGMMA" : "HGMMA") << ".64x" << n(F) << 'x' << k(F) << '.'
    << (f32(F) ? "F32" : "F16");
  if (isQ(F))
    O << '.' << FmtName[fmtA(F)] << '.' << FmtName[fmtB(F)];
  else if (fmtA(F) != SASS::GMMA::F16)
    O << '.' << FmtName[fmtA(F)];
  // RS forms negate A with a '-' on the register group (printGMMAFrag).
  bool RS = MI->getOpcode() == SASS::GMMA_RS || MI->getOpcode() == SASS::GMMA_RS_Z;
  if ((F & NegA) && !RS)
    O << ".negA";
  if (F & NegB)
    O << ".negB";
  if (F & TnspA)
    O << ".tnspA";
  if (F & TnspB)
    O << ".tnspB";
}

void SASSInstPrinter::printGMMAAcc(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  printGroupReg(MI->getOperand(OpNo).getReg(), O,
                SASS::GMMA::accRegs(MI->getOperand(1).getImm()));
}

void SASSInstPrinter::printGMMAFrag(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  if (MI->getOperand(1).getImm() & SASS::GMMA::NegA)
    O << '-';
  printGroupReg(MI->getOperand(OpNo).getReg(), O);
}

void SASSInstPrinter::printGMMAUP(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  MCRegister R = MI->getOperand(OpNo).getReg();
  if (R != SASS::UPT)
    O << ", " << getRegisterName(R);
}

void SASSInstPrinter::printGMMAGsb(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm())
    O << ", gsb0";
}

void SASSInstPrinter::printGDesc(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  O << getRegisterName(MRI.getSubReg(MI->getOperand(OpNo).getReg(), SASS::sub0));
}

void SASSInstPrinter::printGroup(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  printGroupReg(MI->getOperand(OpNo).getReg(), O);
}

void SASSInstPrinter::printImm(const MCInst *MI, unsigned OpNo,
                               raw_ostream &O) {
  int64_t V = MI->getOperand(OpNo).getImm();
  if (V < 0)
    O << "-0x" << format_hex_no_prefix(-static_cast<uint64_t>(V), 1);
  else
    O << "0x" << format_hex_no_prefix(V, 1);
}

void SASSInstPrinter::printF32Imm(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  // Lowered to the raw IEEE bits by SASSAsmPrinter.
  O << "0f" << format_hex_no_prefix(MI->getOperand(OpNo).getImm() & 0xffffffffu, 8);
}

void SASSInstPrinter::printNegFlag(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm())
    O << '!';
}

void SASSInstPrinter::printICC(const MCInst *MI, unsigned OpNo,
                               raw_ostream &O) {
  unsigned CC = MI->getOperand(OpNo).getImm();
  O << SASS::intCondStr(CC);
  if (SASS::intCondUnsigned(CC))
    O << ".U32";
}

void SASSInstPrinter::printFCC(const MCInst *MI, unsigned OpNo,
                               raw_ostream &O) {
  O << SASS::fpCondStr(MI->getOperand(OpNo).getImm());
}

void SASSInstPrinter::printBrTarget(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  const auto *Ref = cast<MCSymbolRefExpr>(Op.getExpr());
  O << "#label(" << labelName(Ref->getSymbol().getName()) << ')';
}

void SASSInstPrinter::printGlobalAddr(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) {
  O << "desc[{UR4,UR5}][";
  printGroupReg(MI->getOperand(OpNo).getReg(), O);
  printOffset(MI->getOperand(OpNo + 1).getImm(), O);
  O << ']';
}

void SASSInstPrinter::printSharedAddr(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) {
  O << '[' << getRegisterName(MI->getOperand(OpNo).getReg());
  printOffset(MI->getOperand(OpNo + 1).getImm(), O);
  O << ']';
}

void SASSInstPrinter::printRawGlobalAddr(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) {
  O << '[';
  printGroupReg(MI->getOperand(OpNo).getReg(), O);
  printOffset(MI->getOperand(OpNo + 1).getImm(), O);
  O << ']';
}

void SASSInstPrinter::printScope(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  static const char *const Names[] = {"SM", "GPU", "SYS"};
  O << Names[MI->getOperand(OpNo).getImm()];
}

void SASSInstPrinter::printOffset(int64_t Off, raw_ostream &O) {
  if (Off > 0)
    O << "+0x" << format_hex_no_prefix(Off, 1);
  else if (Off < 0)
    O << "-0x" << format_hex_no_prefix(-Off, 1);
}
