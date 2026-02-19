//===- CoreDSLToPy.cpp - Translating CoreDSL to Python --------------------===//
//
// Copyright 2025 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "shortnail/Conversion/CoreDSLToPy.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ToolOutputFile.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::shortnail;
using namespace circt;

//===----------------------------------------------------------------------===//
// Utility function
//===----------------------------------------------------------------------===//

void mlir::shortnail::deleteUnusedSymbols(Operation *startOp) {
  // Remove unused operations that define a symbol
  startOp->walk([&](SymbolOpInterface symbolOp) {
    // Ignore instructions and the top module, they are implicitly assumed to be
    // used!
    if (isa<coredsl::InstructionOp, coredsl::AlwaysOp, ModuleOp>(symbolOp))
      return;

    auto *parent = symbolOp->getParentWithTrait<OpTrait::SymbolTable>();
    auto uses = symbolOp.getSymbolUses(parent);
    if (!uses.has_value() || uses->empty())
      symbolOp->erase();
  });
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

#define PREFIX "VAR_"

static unsigned tmpId = 0;
#define TMP_PREFIX "TMP_"
#define ROM_PREFIX "ROM_"

static WalkResult emitOp(mlir::raw_indented_ostream &os, Operation *op);
static std::string valToPyStr(Value val) {
  std::string s;
  llvm::raw_string_ostream out(s);
  val.printAsOperand(out, OpPrintingFlags());
  // Legalize mlir multi-result names
  std::replace(s.begin(), s.end(), '#', '_');
  // Legalize mlir negative constant names
  std::replace(s.begin(), s.end(), '-', '_');
  return PREFIX + s.substr(1);
}
static void valToPy(mlir::raw_ostream &os, Value val) { os << valToPyStr(val); }
static std::string getNewTmpVar() {
  return (Twine(TMP_PREFIX) + Twine(tmpId++)).str();
}
static std::string intToPy(APSInt intVal) {
  return (Twine("ArbInt.from_int(") + Twine(intVal.getExtValue()) +
          Twine(", ") + Twine(intVal.getBitWidth()) + Twine(", ") +
          Twine(intVal.isSigned() ? "True)" : "False)"))
      .str();
}

static const auto pyZeroVal = intToPy(APSInt(APInt::getZero(1)));

template <class BinOpTy>
static WalkResult emitBinOp(mlir::raw_indented_ostream &os, BinOpTy binOp,
                            bool addUnderscore = false) {
  auto splitName = binOp->getName().getStringRef().split('.');
  valToPy(os, binOp.getResult());
  os << " = ";
  valToPy(os, binOp.getLhs());
  os << "." << splitName.second << (addUnderscore ? "_(" : "(");
  valToPy(os, binOp.getRhs());
  os << ")\n";
  return WalkResult::advance();
}
template <class BinOpTy>
static WalkResult emitHWArithBinOp(mlir::raw_indented_ostream &os,
                                   BinOpTy binOp) {
  auto splitName = binOp->getName().getStringRef().split('.');
  Value lhsValue = binOp.getInputs()[0];
  Value rhsValue = binOp.getInputs()[1];
  valToPy(os, binOp.getResult());
  os << " = ";
  valToPy(os, lhsValue);
  os << "." << splitName.second << "(";
  valToPy(os, rhsValue);
  os << ")\n";
  return WalkResult::advance();
}

template <class Op>
static void
handleWordWiseAccess(Op op,
                     std::function<void(const std::string &)> accessHandler) {
  // Note: callers have to pass the *rewritten* `baseAddr` as an argument.
  // `op.getBase()` returns the *old* operand.

  // determine the access range
  const unsigned startOffset =
      op.hasSingleIdxAccess()
          ? (op.getFrom() ? op.getFrom()->getZExtValue() : 0)
          : std::min(op.getFrom()->getZExtValue(), op.getTo()->getZExtValue());
  const unsigned endOffset =
      op.hasSingleIdxAccess()
          ? startOffset
          : std::max(op.getFrom()->getZExtValue(), op.getTo()->getZExtValue());

  auto baseAddr = op.getBase();
  const std::string baseAddrStr =
      baseAddr ? valToPyStr(baseAddr) + ".as_int()" : "";

  for (unsigned i = startOffset; i <= endOffset; ++i) {
    std::string addr;
    // Calculate the access address by adding an offset, but simply use
    // baseAddr for an offset of zero
    if (baseAddr) {
      addr = baseAddrStr + " + " + std::to_string(i);
    } else {
      addr = std::to_string(i);
    }

    accessHandler(addr);
  }
}

static WalkResult emitCoreDSLOp(mlir::raw_indented_ostream &os, Operation *op) {
  assert(isa<coredsl::CoreDSLDialect>(op->getDialect()));

  return TypeSwitch<Operation *, WalkResult>(op)
      .Case<coredsl::SetOp>([&](auto setOp) {
        auto resolvedSym = setOp.resolveSymbol();

        return TypeSwitch<Operation *, WalkResult>(resolvedSym)
            // .Case<coredsl::AliasOp>([&](coredsl::AliasOp alias) {
            //   // TODO
            //   return WalkResult::interrupt();
            // })
            .template Case<coredsl::RegisterOp>([&](coredsl::RegisterOp reg) {
              if (reg.isShimed()) {
                switch (reg.getAccessMode()) {
                case coredsl::RegisterAccessMode::core_x: {
                  os << "write_reg(";
                  if (auto base = setOp.getBase()) {
                    valToPy(os, base);
                    os << ".as_int()";
                  } else {
                    assert(!setOp.getTo());
                    os << setOp.getFrom()->getZExtValue();
                  }
                  os << ", ";
                  valToPy(os, setOp.getValue());
                  os << ".as_int())\n";
                  break;
                }
                case coredsl::RegisterAccessMode::core_pc: {
                  os << "write_pc(";
                  valToPy(os, setOp.getValue());
                  os << ".as_int())\n";
                  break;
                }
                case coredsl::RegisterAccessMode::core_fp: {
                  // TODO
                  llvm_unreachable("NYI");
                  return WalkResult::interrupt();
                }
                case coredsl::RegisterAccessMode::local: {
                  llvm_unreachable("local registers do not use a shim!");
                  break;
                }
                }

              } else {

                bool reversed = setOp.reversedAccessOrder();
                auto regTy = cast<IntegerType>(reg.getRegType());
                unsigned dataWidth = regTy.getWidth();
                unsigned valOffset =
                    reversed
                        ? setOp.getValue().getType().getIntOrFloatBitWidth() -
                              dataWidth
                        : 0;

                handleWordWiseAccess(setOp, [&](const std::string &addr) {
                  // Extract the relevant bits
                  auto tmpVar = getNewTmpVar();
                  os << tmpVar << " = ";
                  valToPy(os, setOp.getValue());
                  os << ".bitextract(" << pyZeroVal << ", "
                     << valOffset + dataWidth - 1 << ", " << valOffset << ")\n";

                  // Perform the write
                  os << "write_cust_reg(" << reg.getSymNameAttr() << ", "
                     << addr << ", " << tmpVar << ".as_int())\n";

                  valOffset += reversed ? -dataWidth : dataWidth;
                });
              }
              return WalkResult::advance();
            })
            .template Case<coredsl::AddressSpaceOp>([&](coredsl::AddressSpaceOp
                                                            addrspace) {
              // lowerings for the different access modes
              switch (addrspace.getAccessMode()) {
              case coredsl::AddressSpaceAccessMode::core_csr: {
                // TODO
                llvm_unreachable("NYI");
                return WalkResult::interrupt();
                break;
              }
              case coredsl::AddressSpaceAccessMode::core_mem: {

                bool reversed = setOp.reversedAccessOrder();
                unsigned dataWidth =
                    addrspace.getResType().getIntOrFloatBitWidth();
                unsigned valOffset =
                    reversed
                        ? setOp.getValue().getType().getIntOrFloatBitWidth() -
                              dataWidth
                        : 0;

                handleWordWiseAccess(setOp, [&](const std::string &addr) {
                  // Extract the relevant bits
                  auto tmpVar = getNewTmpVar();
                  os << tmpVar << " = ";
                  valToPy(os, setOp.getValue());
                  os << ".bitextract(" << pyZeroVal << ", "
                     << valOffset + dataWidth - 1 << ", " << valOffset << ")\n";

                  // Perform the write
                  os << "write_mem(" << addr << ", " << tmpVar
                     << ".as_int())\n";

                  valOffset += reversed ? -dataWidth : dataWidth;
                });

                return WalkResult::advance();
              }
              case coredsl::AddressSpaceAccessMode::wire:
              case coredsl::AddressSpaceAccessMode::axi4mm: {
                addrspace.emitError(
                    "Lowering of wire/axi4mm to python is not supported!");
                return WalkResult::interrupt();
                break;
              }
              }
              return WalkResult::advance();
            })
            .Default(WalkResult::interrupt());
      })
      .Case<coredsl::GetOp>([&](auto getOp) {
        auto resolvedSym = getOp.resolveSymbol();
        SmallVector<std::string> loads;
        auto res =
            TypeSwitch<Operation *, WalkResult>(resolvedSym)
                .Case<coredsl::AliasOp>([&](coredsl::AliasOp alias) {
                  // TODO
                  llvm_unreachable("NYI");
                  return WalkResult::interrupt();
                })
                .template Case<coredsl::RegisterOp>(
                    [&](coredsl::RegisterOp reg) {
                      auto regTy = cast<IntegerType>(reg.getRegType());
                      if (reg.isShimed()) {
                        switch (reg.getAccessMode()) {
                        case coredsl::RegisterAccessMode::core_x: {
                          valToPy(os, getOp.getResult());
                          os << " = ArbInt.from_int(read_reg(";
                          if (auto base = getOp.getBase()) {
                            valToPy(os, base);
                            os << ".as_int()";
                          } else {
                            assert(!getOp.getTo());
                            os << getOp.getFrom()->getZExtValue();
                          }
                          os << "), " << regTy.getWidth() << ", "
                             << (regTy.isSigned() ? "True" : "False") << ")\n";
                          break;
                        }
                        case coredsl::RegisterAccessMode::core_pc: {
                          valToPy(os, getOp.getResult());
                          os << " = ArbInt.from_int(read_pc(), "
                             << regTy.getWidth() << ", "
                             << (regTy.isSigned() ? "True" : "False") << ")\n";
                          break;
                        }
                        case coredsl::RegisterAccessMode::core_fp: {
                          // TODO
                          llvm_unreachable("NYI");
                          return WalkResult::interrupt();
                        }
                        case coredsl::RegisterAccessMode::local: {
                          llvm_unreachable(
                              "local registers do not use a shim!");
                          break;
                        } break;
                        }
                      } else {
                        bool isROM = reg.getIsConst();
                        handleWordWiseAccess(
                            getOp, [&](const std::string &addr) {
                              auto tmpVar = getNewTmpVar();
                              os << tmpVar << " = ArbInt.from_int(";

                              if (isROM) {
                                os << ROM_PREFIX << reg.getSymName() << '['
                                   << addr << "], ";
                              } else {
                                os << "read_cust_reg(" << reg.getSymNameAttr()
                                   << ", " << addr << "), ";
                              }

                              os << regTy.getWidth() << ", "
                                 << (regTy.isSigned() ? "True" : "False")
                                 << ")\n";
                              loads.push_back(tmpVar);
                            });
                      }
                      return WalkResult::advance();
                    })
                .template Case<coredsl::AddressSpaceOp>(
                    [&](coredsl::AddressSpaceOp addrspace) {
                      // lowerings for the different access modes
                      switch (addrspace.getAccessMode()) {
                      case coredsl::AddressSpaceAccessMode::core_csr: {
                        // TODO
                        llvm_unreachable("NYI");
                        return WalkResult::interrupt();
                      }
                      case coredsl::AddressSpaceAccessMode::core_mem: {

                        auto resTy = cast<IntegerType>(addrspace.getResType());

                        handleWordWiseAccess(
                            getOp, [&](const std::string &addr) {
                              auto tmpVar = getNewTmpVar();
                              os << tmpVar << " = ArbInt.from_int(read_mem("
                                 << addr << "), " << resTy.getWidth() << ", "
                                 << (resTy.isSigned() ? "True" : "False")
                                 << ")\n";

                              loads.push_back(tmpVar);
                            });

                        return WalkResult::advance();
                      }
                      case coredsl::AddressSpaceAccessMode::wire:
                      case coredsl::AddressSpaceAccessMode::axi4mm: {
                        addrspace.emitError("Translating of wire/axi4mm to "
                                            "python is not supported!");
                        return WalkResult::interrupt();
                      }
                      }
                      return WalkResult::interrupt();
                    })
                .Default(WalkResult::interrupt());

        if (!loads.empty()) {
          // adjust the concat order according to reversedAccessOrder()
          if (!getOp.reversedAccessOrder()) {
            std::reverse(loads.begin(), loads.end());
          }

          auto curVal = loads[0];
          for (unsigned i = 1; i < loads.size(); ++i) {
            auto tmpVar = getNewTmpVar();
            os << tmpVar << " = " << curVal << ".concat(" << loads[i] << ")\n";
            curVal = tmpVar;
          }
          valToPy(os, getOp.getResult());
          os << " = " << curVal;
          if (cast<IntegerType>(getOp.getResult().getType()).isSigned()) {
            // concat results are always unsigned -> reinterpret as signed
            os << ".cast("
               << cast<IntegerType>(getOp.getResult().getType()).getWidth()
               << ", True)";
          }
          os << '\n';
        }

        return res;
      })
      .Case<coredsl::InstructionOp, coredsl::AlwaysOp, coredsl::SpawnOp,
            coredsl::EndOp>([](auto _) {
        // Handled elsewhere / Nothing to do here
        return WalkResult::advance();
      })
      .Case<coredsl::BitExtractOp>([&](auto extr) {
        valToPy(os, extr.getResult());
        os << " = ";
        valToPy(os, extr.getValue());
        os << ".bitextract(";
        if (extr.getBase())
          valToPy(os, extr.getBase());
        else
          os << pyZeroVal;
        const unsigned startOffset =
            extr.getFrom() ? extr.getFrom()->getZExtValue() : 0;
        const unsigned endOffset =
            extr.getTo() ? extr.getTo()->getZExtValue() : startOffset;
        os << ", " << startOffset << ", " << endOffset << ")"
           << "\n";
        return WalkResult::advance();
      })
      .Case<coredsl::BitSetOp>([&](auto bitset) {
        valToPy(os, bitset.getResult());
        os << " = ";
        valToPy(os, bitset.getValue());
        os << ".bitset(";
        if (bitset.getBase())
          valToPy(os, bitset.getBase());
        else
          os << pyZeroVal;
        const unsigned startOffset =
            bitset.getFrom() ? bitset.getFrom()->getZExtValue() : 0;
        const unsigned endOffset =
            bitset.getTo() ? bitset.getTo()->getZExtValue() : startOffset;
        os << ", " << startOffset << ", " << endOffset << ", ";
        valToPy(os, bitset.getRhs());
        os << ")\n";
        return WalkResult::advance();
      })
      .Case<coredsl::ConcatOp>(
          [&](auto concat) { return emitBinOp(os, concat); })
      .Case<coredsl::ModOp>([&](auto mod) { return emitBinOp(os, mod); })
      .Case<coredsl::OrOp>([&](auto orOp) { return emitBinOp(os, orOp, true); })
      .Case<coredsl::XorOp>([&](auto xorOp) { return emitBinOp(os, xorOp); })
      .Case<coredsl::AndOp>(
          [&](auto andOp) { return emitBinOp(os, andOp, true); })
      .Case<coredsl::ShiftLeftOp>([&](auto sl) { return emitBinOp(os, sl); })
      .Case<coredsl::ShiftRightOp>([&](auto sr) { return emitBinOp(os, sr); })
      .Case<coredsl::CastOp>([&](auto castOp) {
        const auto targetType = cast<IntegerType>(castOp.getResult().getType());

        valToPy(os, castOp.getResult());
        os << " = ";
        valToPy(os, castOp.getValue());
        os << ".cast(" << targetType.getWidth() << ", "
           << (targetType.isSigned() ? "True" : "False") << ")\n";
        return WalkResult::advance();
      })
      .Default([&](auto _) {
        op->emitError() << "CoreDSLToPy::emitCoreDSLOp lacks emission code for "
                           "this operation!";
        return WalkResult::interrupt();
      });
}

static WalkResult emitHWArithOp(mlir::raw_indented_ostream &os, Operation *op) {
  assert(isa<hwarith::HWArithDialect>(op->getDialect()));

  return TypeSwitch<Operation *, WalkResult>(op)
      .Case<hwarith::ConstantOp>([&](auto constOp) {
        valToPy(os, constOp.getResult());
        os << " = " << intToPy(constOp.getConstantValue()) << "\n";
        return WalkResult::advance();
      })
      .Case<hwarith::AddOp>(
          [&](auto addOp) { return emitHWArithBinOp(os, addOp); })
      .Case<hwarith::SubOp>(
          [&](auto subOp) { return emitHWArithBinOp(os, subOp); })
      .Case<hwarith::MulOp>(
          [&](auto mulOp) { return emitHWArithBinOp(os, mulOp); })
      .Case<hwarith::DivOp>(
          [&](auto divOp) { return emitHWArithBinOp(os, divOp); })
      .Case<hwarith::CastOp>([&](auto castOp) {
        // TODO this cast operation can convert to/from signless types
        const auto targetType = cast<IntegerType>(castOp.getResult().getType());
        valToPy(os, castOp.getResult());
        os << " = ";
        valToPy(os, castOp.getIn());
        os << ".cast(" << targetType.getWidth() << ", "
           << (targetType.isSigned() ? "True" : "False") << ")\n";
        return WalkResult::advance();
      })
      .Case<hwarith::ICmpOp>([&](auto icmpOp) {
        valToPy(os, icmpOp.getResult());
        os << " = ";
        valToPy(os, icmpOp.getLhs());
        os << "." << icmpOp.getPredicate() << "(";
        valToPy(os, icmpOp.getRhs());
        os << ")\n";
        return WalkResult::advance();
      })
      .Default([&](auto _) {
        op->emitError() << "CoreDSLToPy::emitHWArithOp lacks emission code for "
                           "this operation!";
        return WalkResult::interrupt();
      });
}
static WalkResult emitHWOp(mlir::raw_indented_ostream &os, Operation *op) {
  assert(isa<hw::HWDialect>(op->getDialect()));

  return TypeSwitch<Operation *, WalkResult>(op)
      .Case<hw::ConstantOp>([&](auto constOp) {
        auto val = constOp.getValue();
        unsigned width = constOp.getType().getIntOrFloatBitWidth();

        valToPy(os, constOp.getResult());
        os << " = ArbInt.from_int(";
        os << val.getSExtValue() << ", " << width << ", False)\n";
        return WalkResult::advance();
      })
      .Default([&](auto _) {
        op->emitError() << "CoreDSLToPy::emitHWOp lacks emission code for "
                           "this operation!";
        return WalkResult::interrupt();
      });
}
static WalkResult emitArithOp(mlir::raw_indented_ostream &os, Operation *op) {
  assert(isa<arith::ArithDialect>(op->getDialect()));

  return TypeSwitch<Operation *, WalkResult>(op)
      .Case<arith::SelectOp>([&](auto selOp) {
        valToPy(os, selOp.getResult());
        os << " = ";
        valToPy(os, selOp.getTrueValue());
        os << " if ";
        valToPy(os, selOp.getCondition());
        os << ".is_true() else ";
        valToPy(os, selOp.getFalseValue());
        os << '\n';
        return WalkResult::advance();
      })
      .Case<arith::AndIOp>([&](auto andOp) {
        valToPy(os, andOp.getResult());
        os << " = ";
        valToPy(os, andOp.getLhs());
        os << ".and_(";
        valToPy(os, andOp.getRhs());
        os << ")\n";
        return WalkResult::advance();
      })
      .Case<arith::AddIOp>([&](auto addOp) {
        valToPy(os, addOp.getResult());
        os << " = ";
        valToPy(os, addOp.getLhs());
        os << ".add(";
        valToPy(os, addOp.getRhs());
        os << ")\n";
        return WalkResult::advance();
      })
      .Default([&](auto _) {
        op->emitError() << "CoreDSLToPy::emitArithOp lacks emission code for "
                           "this operation!";
        return WalkResult::interrupt();
      });
}
static WalkResult emitSCFOp(mlir::raw_indented_ostream &os, Operation *op) {
  static unsigned uniqueIfNumber = 0;
  static unsigned uniqueForNumber = 0;

  assert(isa<scf::SCFDialect>(op->getDialect()));

  return TypeSwitch<Operation *, WalkResult>(op)
      .Case<scf::IfOp>([&](auto ifOp) {
        // define helper function for this ifOp
        auto ifId = uniqueIfNumber++;
        os << "def helper_function_if_" << ifId << "_then():\n";
        os.indent();
        // Visit all operations inside the then block
        auto res = ifOp.thenBlock()->template walk<WalkOrder::PreOrder>(
            [&os](Operation *op) { return emitOp(os, op); });
        if (res.wasInterrupted())
          return res;
        os.unindent();
        auto *elseBlock = ifOp.elseBlock();
        if (elseBlock) {
          os << "def helper_function_if_" << ifId << "_else():\n";
          os.indent();
          // Visit all operations inside the else block
          res = elseBlock->template walk<WalkOrder::PreOrder>(
              [&os](Operation *op) { return emitOp(os, op); });
          if (res.wasInterrupted())
            return res;
          os.unindent();
        }

        os << "if ";
        valToPy(os, ifOp.getCondition());
        os << ".is_true():\n";
        os.indent();
        if (elseBlock) {
          // Call the created helper function
          if (ifOp.getNumResults() > 0) {
            bool first = true;
            for (auto res : ifOp.getResults()) {
              if (!first)
                os << ", ";
              first = false;

              valToPy(os, res);
            }
            os << " = ";
          }
        }
        os << "helper_function_if_" << ifId << "_then()\n";
        os.unindent();
        if (elseBlock) {
          os << "else:\n";
          os.indent();
          // Call the created helper function
          if (ifOp.getNumResults() > 0) {
            bool first = true;
            for (auto res : ifOp.getResults()) {
              if (!first)
                os << ", ";
              first = false;

              valToPy(os, res);
            }
            os << " = ";
          }
          os << "helper_function_if_" << ifId << "_else()\n";
          os.unindent();
        }

        return WalkResult::skip();
      })
      .Case<scf::YieldOp>([&](auto yieldOp) {
        os << "return ";
        bool first = true;
        for (auto res : yieldOp.getResults()) {
          if (!first)
            os << ", ";

          valToPy(os, res);
          first = false;
        }
        os << "\n";
        return WalkResult::advance();
      })
      // .Case<scf::WhileOp>([&](auto whileOp) {
      //   // TODO PITA
      //   return WalkResult::interrupt();
      // })
      // .Case<scf::ConditionOp>([&](auto condOp) {
      //   // TODO PITA
      //   return WalkResult::interrupt();
      // })
      .Case<scf::ForOp>([&](auto forOp) {
        auto forId = uniqueForNumber++;
        os << "def helper_function_for_" << forId << "():\n";
        os.indent();

        os << "def helper_function_for_" << forId << "_body(i, iter_args):\n";
        os.indent();
        // Convert i to ArbInt
        valToPy(os, forOp.getInductionVar());
        os << " = ArbInt.from_int(i, 32, False) #TODO do not hardcode type... "
              "what about negative values?\n";
        // Convert iter_args to block arguments
        if (forOp.getRegionIterArgs().size()) {
          bool first = true;
          for (auto blockArg : forOp.getRegionIterArgs()) {
            if (!first)
              os << ", ";
            first = false;
            valToPy(os, blockArg);
          }
          os << " = iter_args\n";
        }
        // Emit the loop body
        auto res = forOp.getBody()->template walk<WalkOrder::PreOrder>(
            [&os](Operation *op) { return emitOp(os, op); });
        if (res.wasInterrupted())
          return res;
        os.unindent();

        // Initialize iter_args
        os << "iter_args = (";
        bool first = true;
        for (auto initVal : forOp.getInitArgs()) {
          if (!first)
            os << ", ";
          first = false;
          valToPy(os, initVal);
        }
        os << ")\n";

        os << "for i in range(";
        valToPy(os, forOp.getLowerBound());
        os << ".as_int(), ";
        valToPy(os, forOp.getUpperBound());
        os << ".as_int(), ";
        valToPy(os, forOp.getStep());
        os << ".as_int()):\n";
        os.indent();
        os << "iter_args = helper_function_for_" << forId
           << "_body(i, iter_args)\n";
        os.unindent();
        os << "return iter_args\n";
        os.unindent();

        if (forOp.getNumResults() > 0) {
          bool first = true;
          for (auto res : forOp.getResults()) {
            if (!first)
              os << ", ";
            first = false;

            valToPy(os, res);
          }
          os << " = ";
        }
        os << "helper_function_for_" << forId << "()\n";

        return WalkResult::skip();
      })
      .Default([&](auto _) {
        op->emitError()
            << "CoreDSLToPy::emitSCFOp lacks emission code for this operation!";
        return WalkResult::interrupt();
      });
}

static WalkResult emitOp(mlir::raw_indented_ostream &os, Operation *op) {
  return TypeSwitch<Dialect *, WalkResult>(op->getDialect())
      .Case<hwarith::HWArithDialect>(
          [&](auto _) { return emitHWArithOp(os, op); })
      .Case<hw::HWDialect>([&](auto _) { return emitHWOp(os, op); })
      .Case<arith::ArithDialect>([&](auto _) { return emitArithOp(os, op); })
      .Case<coredsl::CoreDSLDialect>(
          [&](auto _) { return emitCoreDSLOp(os, op); })
      .Case<scf::SCFDialect>([&](auto _) { return emitSCFOp(os, op); })
      .Default([&](auto dialect) {
        op->emitError() << "CoreDSLToPy: missing handling for entire dialect: "
                        << dialect->getNamespace() << "\n";
        return WalkResult::interrupt();
      });
}

static bool emitInstr(mlir::raw_indented_ostream &os,
                      coredsl::InstructionOp instr) {
  // Define the python function
  os << "def " << instr.getName()
     << "(opcode, read_reg=read_reg, write_reg=write_reg, "
        "read_cust_reg=read_cust_reg, write_cust_reg=write_cust_reg, "
        "read_pc=read_pc, write_pc=write_pc, "
        "read_mem=read_mem, write_mem=write_mem):\n";
  os.indent();

  os << "bitvector_opcode = ArbInt.from_int(opcode, 32, False)\n";

  // Make encoding fields available
  Block *insBlock = &instr.getFunctionBody().front();
  const auto numArgs = insBlock->getNumArguments();
  SmallVector<Value> argReplacements;
  if (numArgs != 0) {
    SmallVector<coredsl::InstructionOp::EncodingField> encFields;
    instr.getEncodingFields(encFields);

    for (const auto &[encField, arg] :
         llvm::zip(encFields, insBlock->getArguments())) {
      valToPy(os, arg);
      os << " = bitvector_opcode.bitextract(" << pyZeroVal << ", "
         << encField.msb << ", " << encField.lsb << ")\n";
    }
  }

  // Lower all operations inside the instruction
  auto res = instr->walk<WalkOrder::PreOrder>(
      [&](Operation *op) { return emitOp(os, op); });

  os.unindent();

  return !res.wasInterrupted();
}

static bool emitAlways(mlir::raw_indented_ostream &os,
                       coredsl::AlwaysOp always) {
  // Define the python function
  os << "def " << always.getName()
     << "(read_reg=read_reg, write_reg=write_reg, "
        "read_cust_reg=read_cust_reg, write_cust_reg=write_cust_reg, "
        "read_pc=read_pc, write_pc=write_pc, "
        "read_mem=read_mem, write_mem=write_mem):\n";
  os.indent();

  // Lower all operations inside the instruction
  auto res = always->walk<WalkOrder::PreOrder>(
      [&](Operation *op) { return emitOp(os, op); });

  os.unindent();

  return !res.wasInterrupted();
}

namespace {
struct CoreDSLToPy : public CoreDSLToPyBase<CoreDSLToPy> {
  void runOnOperation() override {
    auto isaxOp = getOperation();

    // Inline all functions into the InstructionOps
    OpPassManager inlinePM(isaxOp.getOperationName());
    inlinePM.addPass(createInlinerPass());
    if (failed(runPipeline(inlinePM, isaxOp)))
      return signalPassFailure();

    deleteUnusedSymbols(isaxOp);

    auto isaxName = isaxOp.getNameAttr();
    if (!isaxName || isaxName.empty()) {
      isaxName = StringAttr::get(isaxOp.getContext(), "UNNAMED_ISAX");
    }
    std::string fileName = isaxName.str() + ".py";

    std::string errorMessage;
    auto output = mlir::openOutputFile(fileName, &errorMessage);
    if (!output) {
      mlir::emitError(OpBuilder{isaxOp->getContext()}.getUnknownLoc())
          << errorMessage;
      return signalPassFailure();
    }
    mlir::raw_indented_ostream os(output->os());

    os << R"(
try:
  from Antmicro.Renode.Peripherals.CPU import RegisterValue
except:
  pass

def read_mem(addr):
  data = machine.SystemBus.ReadByte(addr)
  print("READ from 0x{:08X} = 0x{:02X}".format(addr, int(data)))
  return data
def write_mem(addr, data):
  machine.SystemBus.WriteByte(addr, data)
  print("WRITE to 0x{:08X} = 0x{:02X}".format(addr, data))

def read_pc():
  return cpu.PC.RawValue
def write_pc(new_pc):
  cpu.PC = RegisterValue.Create(new_pc, 32) #TODO do not hardcode 32 bit

def read_reg(addr):
  return cpu.GetRegister(addr).RawValue
def write_reg(addr, data):
  cpu.SetRegister(addr, RegisterValue.Create(data, 32)) #TODO do not hardcode 32 bit

def init_cust_regs():
  state["cust_regs"] = dict()
)";

    os.indent();
    SmallVector<coredsl::RegisterOp> roms;
    isaxOp->walk([&](coredsl::RegisterOp reg) {
      if (!reg.isShimed() &&
          reg.getAccessMode() == coredsl::RegisterAccessMode::local) {
        if (reg.getIsConst()) {
          roms.push_back(reg);
          return;
        }
        os << "state[\"cust_regs\"][" << reg.getSymNameAttr() << "] = [0] * "
           << reg.getSize() << "\n";
        // Handle init values
        if (ArrayAttr initializer = reg.getInitializerAttr()) {
          bool regIsSigned = cast<IntegerType>(reg.getRegType()).isSigned();

          unsigned addr = 0;
          for (auto iv : initializer.getAsValueRange<IntegerAttr>()) {
            os << "state[\"cust_regs\"][" << reg.getSymNameAttr() << "]["
               << addr << "] = int(";
            if (regIsSigned)
              os << iv.getSExtValue();
            else
              os << iv.getZExtValue();
            os << ")\n";
            ++addr;
          }
        }
      }
    });
    os.unindent();

    for (auto rom : roms) {
      os << ROM_PREFIX << rom.getSymName() << " : list = [";

      ArrayAttr initializer = rom.getInitializerAttr();
      bool regIsSigned = cast<IntegerType>(rom.getRegType()).isSigned();
      for (auto iv : initializer.getAsValueRange<IntegerAttr>()) {
        os << "int(";
        if (regIsSigned)
          os << iv.getSExtValue();
        else
          os << iv.getZExtValue();
        os << "), ";
      }
      os << "]\n";
    }

    os << R"(
def read_cust_reg(name, addr):
  return state["cust_regs"][name][addr]
def write_cust_reg(name, addr, data):
  state["cust_regs"][name][addr] = int(data)

# Init ISAX state
if 'state' in locals() or 'state' in globals():
  if "cust_regs" not in state:
    init_cust_regs()

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ArbInt import *

)";

    SmallVector<std::pair<StringAttr, StringRef>> maskInstrNameList;

    auto res =
        isaxOp->walk<WalkOrder::PreOrder>([&](coredsl::InstructionOp instr) {
          if (!emitInstr(os, instr))
            return WalkResult::interrupt();

          maskInstrNameList.emplace_back(instr.getEncodingMask(),
                                         instr.getName());

          return WalkResult::skip();
        });

    if (res.wasInterrupted())
      return signalPassFailure();

    res = isaxOp->walk<WalkOrder::PreOrder>([&](coredsl::AlwaysOp always) {
      if (!emitAlways(os, always))
        return WalkResult::interrupt();

      return WalkResult::skip();
    });

    if (res.wasInterrupted())
      return signalPassFailure();

    // Emit the decoder
    os << "\n\ndecoder_table = [\n";
    os.indent();
    for (const auto &[mask, instrName] : maskInstrNameList) {
      os << "(0b";
      for (char c : mask.getValue())
        os << (c == '-' ? '0' : '1');
      os << ", 0b";
      for (char c : mask.getValue())
        os << (c == '-' ? '0' : c);
      os << ", " << instrName << "),\n";
    }
    os.unindent();
    os << "]\n";

    os << R"(
def decode(instruction):
  for mask, match_value, handler in decoder_table:
    if (instruction & mask) == match_value:
      handler(instruction)
      return
  1/0 # Unknown instruction

# Init ISAX state
if 'instruction' in locals() or 'instruction' in globals():
  decode(instruction)
)";

    output->keep();
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::shortnail::createCoreDSLToPyPass() {
  return std::make_unique<CoreDSLToPy>();
}
