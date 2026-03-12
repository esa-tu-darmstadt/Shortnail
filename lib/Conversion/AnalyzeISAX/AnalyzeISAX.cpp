//===- AnalyzeISAX.cpp - Analyze CoreDSL ISAX for LLVM patching -----------===//
//
// This pass analyzes CoreDSL MLIR (ShortNail dialect) and outputs a structured
// YAML description of each ISAX instruction's encoding, operand semantics,
// and side effects. This YAML is consumed by the Python patch generator.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "circt/Dialect/HWArith/HWArithDialect.h"

#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Get the bit width of an integer type (standard MLIR IntegerType covers
/// both signless and HWArith signed/unsigned integer types in this CIRCT).
static unsigned getTypeWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  return 0;
}

/// Check if a type is signed (IntegerType with Signed signedness).
static bool isSignedType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.isSigned();
  return false;
}

/// Clean instruction/ISAX name by removing '_' and '.' characters
/// and lowercasing (builtins/intrinsics must be lowercase).
static std::string cleanName(StringRef name) {
  std::string result;
  result.reserve(name.size());
  for (char c : name) {
    if (c != '_' && c != '.')
      result.push_back(std::tolower(c));
  }
  return result;
}

/// Per-field metadata extracted from the lil.enc_immediates attribute.
struct EncImmInfo {
  std::string cleanName;
  /// CoreDSL field bit range: start >= end (Treenail normalizes reversed
  /// fields so that start is always the MSB). For imm12[9:4], start=9 end=4.
  unsigned start;
  unsigned end;
  /// True if the CoreDSL declaration had the bit range reversed (e.g.
  /// imm12[4:9] instead of imm12[9:4]). Treenail resolves the reversal
  /// in the MLIR via coredsl.bitextract, so the SSA value already has
  /// bits in their natural order. For tablegen field subscripts the
  /// reversed flag inverts the subscript range so the encoding correctly
  /// maps MSB→end, LSB→start instead of the normal MSB→start, LSB→end.
  bool reversed;
};

/// Parse the lil.enc_immediates attribute on an InstructionOp.
///
/// Treenail sets this attribute with the format (see EncodingFieldSwitch.java):
///   [[["<ssa_name>", start, end, reversed, "<clean_name>"], ...], ...]
///
/// Each outer element is one logical encoding field (possibly non-contiguous
/// with multiple parts). Returns a map from SSA name → EncImmInfo.
static llvm::StringMap<EncImmInfo>
parseEncImmediates(coredsl::InstructionOp instOp) {
  llvm::StringMap<EncImmInfo> result;
  auto encImm = instOp->getAttrOfType<ArrayAttr>("lil.enc_immediates");
  assert(encImm && "lil.enc_immediates attribute missing on InstructionOp; "
                   "ensure MLIR is produced by a recent Treenail version");

  for (auto fieldAttr : encImm) {
    auto partsArray = dyn_cast<ArrayAttr>(fieldAttr);
    if (!partsArray || partsArray.empty())
      continue;

    // Each part is [ssa_name, start, end, reversed, clean_name]
    for (auto partAttr : partsArray) {
      auto partArray = dyn_cast<ArrayAttr>(partAttr);
      assert(partArray && partArray.size() >= 5 &&
             "lil.enc_immediates: each part must have 5 elements "
             "[ssa_name, start, end, reversed, clean_name]");

      auto ssaNameAttr = dyn_cast<StringAttr>(partArray[0]);
      auto startAttr = dyn_cast<IntegerAttr>(partArray[1]);
      auto endAttr = dyn_cast<IntegerAttr>(partArray[2]);
      auto reversedAttr = dyn_cast<IntegerAttr>(partArray[3]);
      auto cleanNameAttr = dyn_cast<StringAttr>(partArray[4]);
      assert(ssaNameAttr && startAttr && endAttr && reversedAttr &&
             cleanNameAttr && "lil.enc_immediates: malformed part entry");

      StringRef ssaName = ssaNameAttr.getValue();
      if (ssaName.starts_with("%"))
        ssaName = ssaName.drop_front(1);

      EncImmInfo info;
      info.cleanName = cleanNameAttr.getValue().str();
      info.start = startAttr.getInt();
      info.end = endAttr.getInt();
      info.reversed = reversedAttr.getInt() != 0;
      result[ssaName] = std::move(info);
    }
  }
  return result;
}

/// Look up a symbol by name in the given ISAXOp.
static Operation *lookupSymbol(coredsl::ISAXOp isaxOp, StringRef name) {
  return SymbolTable::lookupSymbolIn(
      isaxOp, StringAttr::get(isaxOp.getContext(), name));
}

/// Resolve a symbol name to the underlying RegisterOp or AddressSpaceOp,
/// following through AliasOp chains.
static Operation *resolveToTarget(coredsl::ISAXOp isaxOp, StringRef symName) {
  auto *sym = lookupSymbol(isaxOp, symName);
  if (auto aliasOp = dyn_cast_or_null<coredsl::AliasOp>(sym))
    return aliasOp.resolveSymbol();
  return sym;
}

/// Collect all values that represent the same logical value as `root`,
/// following through coredsl.cast operations.
static void collectTransitiveValues(Value root,
                                    SmallVectorImpl<Value> &values) {
  SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  worklist.push_back(root);
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (!visited.insert(val).second)
      continue;
    values.push_back(val);
    for (Operation *user : val.getUsers()) {
      if (auto castOp = dyn_cast<coredsl::CastOp>(user))
        worklist.push_back(castOp.getResult());
    }
  }
}

/// Check if a value (result of reading a register) flows into a memory
/// access operation (GetOp/SetOp on a core_mem address space).
static bool isUsedInMemAccess(Value regVal, coredsl::ISAXOp isaxOp) {
  SmallVector<Value> transitiveValues;
  collectTransitiveValues(regVal, transitiveValues);

  for (Value val : transitiveValues) {
    for (Operation *user : val.getUsers()) {
      if (auto getOp = dyn_cast<coredsl::GetOp>(user)) {
        if (getOp.getBase() == val) {
          auto *sym = resolveToTarget(isaxOp, getOp.getSym());
          if (auto addrOp = dyn_cast_or_null<coredsl::AddressSpaceOp>(sym)) {
            if (addrOp.getAccessMode() ==
                coredsl::AddressSpaceAccessMode::core_mem)
              return true;
          }
        }
      }
      if (auto setOp = dyn_cast<coredsl::SetOp>(user)) {
        if (setOp.getBase() == val) {
          auto *sym = resolveToTarget(isaxOp, setOp.getSym());
          if (auto addrOp = dyn_cast_or_null<coredsl::AddressSpaceOp>(sym)) {
            if (addrOp.getAccessMode() ==
                coredsl::AddressSpaceAccessMode::core_mem)
              return true;
          }
        }
      }
    }
  }
  return false;
}

/// Trace a value backwards through intermediate operations to see if it
/// originated from a memory load (GetOp on core_mem).
static bool valueComesFromMem(Value val, coredsl::ISAXOp isaxOp) {
  SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  worklist.push_back(val);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;

    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      continue;

    if (auto getOp = dyn_cast<coredsl::GetOp>(defOp)) {
      auto *sym = resolveToTarget(isaxOp, getOp.getSym());
      if (auto addrOp = dyn_cast_or_null<coredsl::AddressSpaceOp>(sym)) {
        if (addrOp.getAccessMode() == coredsl::AddressSpaceAccessMode::core_mem)
          return true;
      }
    }

    // Trace through CoreDSL data-processing ops
    if (isa<coredsl::CastOp, coredsl::BitExtractOp, coredsl::BitSetOp,
            coredsl::ConcatOp, coredsl::OrOp, coredsl::XorOp, coredsl::AndOp,
            coredsl::ShiftLeftOp, coredsl::ShiftRightOp, coredsl::ModOp>(
            defOp)) {
      for (Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }
    // Trace through hwarith ops (add, sub, mul, div, etc.)
    if (defOp->getDialect() &&
        isa<circt::hwarith::HWArithDialect>(defOp->getDialect())) {
      for (Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }
  }
  return false;
}

/// Check if an immediate field is signed by tracing through CastOps.
static bool isSignedImmediate(BlockArgument arg) {
  SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  worklist.push_back(arg);

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (!visited.insert(val).second)
      continue;

    for (Operation *user : val.getUsers()) {
      if (auto castOp = dyn_cast<coredsl::CastOp>(user)) {
        if (isSignedType(castOp.getResult().getType()))
          return true;
        worklist.push_back(castOp.getResult());
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Data structures for YAML output
//===----------------------------------------------------------------------===//

struct FieldInfo {
  std::string name;
  /// Encoding bit ranges (MSB-first order). For contiguous fields, contains
  /// one entry. For non-contiguous fields (e.g. B-type imm12), contains
  /// multiple entries whose bit widths sum to `bits`.
  SmallVector<std::pair<unsigned, unsigned>> ranges; // (msb, lsb) pairs
  /// Field bit ranges — which bits of the field each encoding range maps to.
  /// Parallel to `ranges`. For contiguous fields this is trivially 0..bits-1.
  /// For non-contiguous fields (e.g. B-type imm12), the CoreDSL encoding
  /// specifies exactly which field bits go where (e.g. imm12[10:10] at bit 7).
  SmallVector<std::pair<unsigned, unsigned>>
      fieldRanges;   // (field_msb, field_lsb)
  unsigned bits = 0; // total bit width across all ranges
  enum FieldType { Register, Immediate, Mask } type = Mask;

  // Register-specific fields
  std::string regFile;
  unsigned regWidth = 0;
  bool isInput = false;
  bool isOutput = false;
  bool memInput = false;  // register value used as memory load address
  bool memOutput = false; // register written from memory load result
  bool constrained = false;

  // Immediate-specific fields
  bool isSigned = false;
};

struct InstructionInfo {
  std::string name;
  std::string mask;
  bool isJump = false;
  bool clobbersAllRegs = false; // true when GPR write index is unresolvable
  llvm::DenseSet<unsigned>
      specificClobbers; // register indices for known-constant writes
  enum SideEffects {
    NoSideEffect,
    ReadOnly,
    WriteOnly,
    ReadWrite
  } sideEffects = NoSideEffect;
  SmallVector<FieldInfo> fields;
};

struct ISAXInfo {
  std::string name;
  SmallVector<InstructionInfo, 4> instructions;
};

//===----------------------------------------------------------------------===//
// Analysis logic
//===----------------------------------------------------------------------===//

/// Analyze a single encoding field (block argument) to determine its type.
static void analyzeField(BlockArgument blockArg, coredsl::ISAXOp isaxOp,
                         FieldInfo &fi) {
  // Collect all values that this argument flows through (via casts)
  SmallVector<Value> argValues;
  collectTransitiveValues(blockArg, argValues);

  bool isUsed = false;

  for (Value val : argValues) {
    for (Operation *user : val.getUsers()) {
      // Skip CastOps (already handled in collectTransitiveValues)
      if (isa<coredsl::CastOp>(user))
        continue;

      isUsed = true;

      // Check if used as index in coredsl.get @X[field] (register read)
      if (auto getOp = dyn_cast<coredsl::GetOp>(user)) {
        if (getOp.getBase() == val) {
          auto *sym = resolveToTarget(isaxOp, getOp.getSym());
          if (auto regOp = dyn_cast_or_null<coredsl::RegisterOp>(sym)) {
            auto mode = regOp.getAccessMode();
            if (mode == coredsl::RegisterAccessMode::core_x ||
                mode == coredsl::RegisterAccessMode::core_fp) {
              fi.type = FieldInfo::Register;
              fi.isInput = true;
              fi.regFile = regOp.getSymName().str();
              fi.regWidth = getTypeWidth(getOp.getResult().getType());
              // Check if register value flows to memory access
              if (isUsedInMemAccess(getOp.getResult(), isaxOp))
                fi.memInput = true;
            }
          }
        }
      }

      // Check if used as index in coredsl.set @X[field] (register write)
      if (auto setOp = dyn_cast<coredsl::SetOp>(user)) {
        if (setOp.getBase() == val) {
          auto *sym = resolveToTarget(isaxOp, setOp.getSym());
          if (auto regOp = dyn_cast_or_null<coredsl::RegisterOp>(sym)) {
            auto mode = regOp.getAccessMode();
            if (mode == coredsl::RegisterAccessMode::core_x ||
                mode == coredsl::RegisterAccessMode::core_fp) {
              fi.type = FieldInfo::Register;
              fi.isOutput = true;
              fi.regFile = regOp.getSymName().str();
              fi.regWidth = getTypeWidth(setOp.getValue().getType());
              // Check if written value came from memory
              if (valueComesFromMem(setOp.getValue(), isaxOp))
                fi.memOutput = true;
            }
          }
        }
      }
    }
  }

  fi.constrained = fi.isInput && fi.isOutput;

  // If not a register but used somewhere, it's an immediate
  if (fi.type == FieldInfo::Mask && isUsed)
    fi.type = FieldInfo::Immediate;

  // Check signedness for immediates
  if (fi.type == FieldInfo::Immediate)
    fi.isSigned = isSignedImmediate(blockArg);
}

/// Returns true only if v traces to a single BlockArgument from the
/// instruction's entry block (an encoding field) through single-operand ops
/// (casts, truncations, etc.).  Any multi-operand op (e.g. concat), constant
/// leaf, or non-entry block argument (loop IV) returns false.
static bool traceToSingleInstructionArg(Value v,
                                        coredsl::InstructionOp instOp) {
  Block &entryBlock = instOp.getRegion().front();
  while (true) {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return arg.getParentBlock() == &entryBlock;
    Operation *defOp = v.getDefiningOp();
    if (!defOp || defOp->getNumOperands() != 1)
      return false; // constant, multi-operand op, or no def
    v = defOp->getOperand(0);
  }
}

/// Try to evaluate v as a compile-time constant integer.
/// Matches any ConstantLike op (hw.constant, arith.constant, hwarith.constant)
/// via matchPattern/m_ConstantInt, and threads through single-operand cast-like
/// ops (casts, truncations).  Multi-operand ops return false.
static bool tryFoldToConstant(Value v, uint64_t &result) {
  while (true) {
    llvm::APInt constVal;
    if (mlir::matchPattern(v, mlir::m_ConstantInt(&constVal))) {
      result = constVal.getZExtValue();
      return true;
    }
    Operation *defOp = v.getDefiningOp();
    if (!defOp || defOp->getNumOperands() != 1)
      return false;
    v = defOp->getOperand(0);
  }
  return false;
}

/// Analyze side effects and jump status of an instruction.
static void analyzeSideEffects(coredsl::InstructionOp instOp,
                               coredsl::ISAXOp isaxOp, InstructionInfo &info) {
  bool hasRead = false;
  bool hasWrite = false;
  info.isJump = false;

  // Walk all operations in the instruction body (including spawn blocks).
  // Function calls are already inlined by the inliner pass that runs before
  // this analysis, so all side effects are visible as direct ops.
  instOp.walk([&](Operation *op) {
    if (auto getOp = dyn_cast<coredsl::GetOp>(op)) {
      auto *sym = resolveToTarget(isaxOp, getOp.getSym());

      // Memory read
      if (auto addrOp = dyn_cast_or_null<coredsl::AddressSpaceOp>(sym)) {
        if (addrOp.getAccessMode() == coredsl::AddressSpaceAccessMode::core_mem)
          hasRead = true;
      }
      // Custom (local) register read — skip const registers (e.g. LUTs)
      if (auto regOp = dyn_cast_or_null<coredsl::RegisterOp>(sym)) {
        if (regOp.getAccessMode() == coredsl::RegisterAccessMode::local &&
            !regOp.getIsConst())
          hasRead = true;
      }
    }

    if (auto setOp = dyn_cast<coredsl::SetOp>(op)) {
      auto *sym = resolveToTarget(isaxOp, setOp.getSym());

      // Memory write
      if (auto addrOp = dyn_cast_or_null<coredsl::AddressSpaceOp>(sym)) {
        if (addrOp.getAccessMode() == coredsl::AddressSpaceAccessMode::core_mem)
          hasWrite = true;
      }
      // Custom (local) register write
      if (auto regOp = dyn_cast_or_null<coredsl::RegisterOp>(sym)) {
        if (regOp.getAccessMode() == coredsl::RegisterAccessMode::local)
          hasWrite = true;
      }
      // PC write → jump
      if (auto regOp = dyn_cast_or_null<coredsl::RegisterOp>(sym)) {
        if (regOp.getAccessMode() == coredsl::RegisterAccessMode::core_pc)
          info.isJump = true;
      }
      // GPR/FP write whose index is not fully from encoding fields
      if (auto regOp = dyn_cast_or_null<coredsl::RegisterOp>(sym)) {
        auto mode = regOp.getAccessMode();
        if (mode == coredsl::RegisterAccessMode::core_x ||
            mode == coredsl::RegisterAccessMode::core_fp) {
          // Determine offset range from from/to attributes.
          unsigned lo = 0, hi = 0;
          if (auto from = setOp.getFrom()) {
            lo = hi = from->getZExtValue();
            if (auto to = setOp.getTo()) {
              unsigned toVal = to->getZExtValue();
              lo = std::min(lo, toVal);
              hi = std::max(hi, toVal);
            }
          }

          // Resolve the base index (if present).
          unsigned baseIdx = 0;
          if (setOp.getBase()) {
            if (!setOp.getFrom() &&
                traceToSingleInstructionArg(setOp.getBase(), instOp))
              return; // encoding field — handled by field analysis

            uint64_t val;
            if (tryFoldToConstant(setOp.getBase(), val))
              baseIdx = static_cast<unsigned>(val);
            else {
              info.clobbersAllRegs = true;
              return;
            }
          }

          for (unsigned r = lo; r <= hi; ++r)
            info.specificClobbers.insert(baseIdx + r);
        }
      }
    }
  });

  if (hasRead && hasWrite)
    info.sideEffects = InstructionInfo::ReadWrite;
  else if (hasRead)
    info.sideEffects = InstructionInfo::ReadOnly;
  else if (hasWrite)
    info.sideEffects = InstructionInfo::WriteOnly;
  else
    info.sideEffects = InstructionInfo::NoSideEffect;
}

/// Look up the EncImmInfo for an encoding argument.
static const EncImmInfo &
lookupEncImm(coredsl::InstructionOp instOp, unsigned argIdx,
             const llvm::StringMap<EncImmInfo> &encImmMap) {
  StringRef rawName = instOp.getInstructionArgumentName(argIdx);
  StringRef lookupName = rawName;
  if (lookupName.starts_with("%"))
    lookupName = lookupName.drop_front(1);
  auto it = encImmMap.find(lookupName);
  assert(it != encImmMap.end() &&
         "encoding argument not found in lil.enc_immediates map");
  return it->second;
}

/// Analyze a single InstructionOp.
static void analyzeInstruction(coredsl::InstructionOp instOp,
                               coredsl::ISAXOp isaxOp, InstructionInfo &info) {
  // 1. Name and encoding mask
  info.name = cleanName(instOp.getName());
  info.mask = instOp.getEncodingMask().getValue().str();

  // 2. Parse lil.enc_immediates attribute (field names, ranges, reversed flag)
  auto encImmMap = parseEncImmediates(instOp);

  // 3. Iterate encoding fields (MSB to LSB)
  // Non-contiguous fields (same name, split across multiple bit ranges)
  // are merged into a single FieldInfo with multiple ranges.
  ArrayRef<Attribute> enc = instOp.getEncoding().getValue();
  unsigned argIdx = 0;
  unsigned bitPos = 32;

  // Map field name → index in info.fields for merging non-contiguous fields
  llvm::StringMap<unsigned> fieldIndex;

  for (auto attr : enc) {
    StringRef field = cast<StringAttr>(attr).getValue();
    if (field.empty())
      continue;

    if (coredsl::isEncodingFieldLiteral(field)) {
      bitPos -= field.size();
      continue;
    }

    unsigned argSize = getTypeWidth(instOp.getArgument(argIdx).getType());
    unsigned msb = bitPos - 1;
    unsigned lsb = bitPos - argSize;
    bitPos -= argSize;

    const auto &encImm = lookupEncImm(instOp, argIdx, encImmMap);
    std::string fieldName = encImm.cleanName;

    // Field bit range from the attribute (CoreDSL field bit indices).
    // For non-reversed: MSB→start, LSB→end (e.g. imm12[9:4] → (9, 4)).
    // For reversed: the encoding MSB maps to end and LSB maps to start
    // (Treenail normalized start>end but the mapping is inverted).
    std::pair<unsigned, unsigned> fieldBits;
    if (encImm.reversed)
      fieldBits = {encImm.end, encImm.start};
    else
      fieldBits = {encImm.start, encImm.end};

    auto it = fieldIndex.find(fieldName);
    if (it != fieldIndex.end()) {
      // Non-contiguous: merge into existing field entry
      FieldInfo &existing = info.fields[it->second];
      existing.ranges.push_back({msb, lsb});
      existing.fieldRanges.push_back(fieldBits);
      existing.bits += argSize;
    } else {
      FieldInfo fi;
      fi.name = fieldName;
      fi.ranges.push_back({msb, lsb});
      fi.fieldRanges.push_back(fieldBits);
      fi.bits = argSize;

      // Analyze field semantics
      auto blockArg = instOp->getRegion(0).front().getArgument(argIdx);
      analyzeField(blockArg, isaxOp, fi);

      fieldIndex[fieldName] = info.fields.size();
      info.fields.push_back(fi);
    }
    argIdx++;
  }

  // 4. Side effects and jump detection
  analyzeSideEffects(instOp, isaxOp, info);
}

//===----------------------------------------------------------------------===//
// YAML output
//===----------------------------------------------------------------------===//

static void writeYAML(const ISAXInfo &isax, mlir::raw_indented_ostream &os) {
  os << "extension: \"" << isax.name << "\"\n";
  os << "instructions:\n";

  for (const auto &inst : isax.instructions) {
    os.indent();
    os << "- name: \"" << inst.name << "\"\n";
    os << "  mask: \"" << inst.mask << "\"\n";
    os << "  is_jump: " << (inst.isJump ? "true" : "false") << "\n";

    const char *seStr;
    switch (inst.sideEffects) {
    case InstructionInfo::NoSideEffect:
      seStr = "none";
      break;
    case InstructionInfo::ReadOnly:
      seStr = "read";
      break;
    case InstructionInfo::WriteOnly:
      seStr = "write";
      break;
    case InstructionInfo::ReadWrite:
      seStr = "readwrite";
      break;
    }
    os << "  side_effects: \"" << seStr << "\"\n";
    if (inst.clobbersAllRegs)
      os << "  clobbers_all_regs: true\n";
    else if (!inst.specificClobbers.empty()) {
      SmallVector<unsigned> regs(inst.specificClobbers.begin(),
                                 inst.specificClobbers.end());
      llvm::sort(regs);
      os << "  clobbers_regs: [";
      for (unsigned i = 0; i < regs.size(); ++i) {
        if (i > 0)
          os << ", ";
        os << "\"x" << regs[i] << "\"";
      }
      os << "]\n";
    }

    os << "  fields:\n";
    for (const auto &f : inst.fields) {
      os.indent();
      os << "  - name: \"" << f.name << "\"\n";
      if (f.ranges.size() == 1) {
        os << "    range: \"" << f.ranges[0].first << "-" << f.ranges[0].second
           << "\"\n";
      } else {
        os << "    ranges:\n";
        for (const auto &r : f.ranges)
          os << "      - \"" << r.first << "-" << r.second << "\"\n";
        // Output the original CoreDSL field bit indices for non-contiguous
        // fields. These tell gen_patches.py which bits of the field variable
        // map to each encoding range (e.g. imm12{9-4} not imm12{10-5}).
        os << "    field_ranges:\n";
        for (const auto &fr : f.fieldRanges)
          os << "      - \"" << fr.first << "-" << fr.second << "\"\n";
      }
      os << "    bits: " << f.bits << "\n";

      const char *typeStr;
      switch (f.type) {
      case FieldInfo::Register:
        typeStr = "register";
        break;
      case FieldInfo::Immediate:
        typeStr = "immediate";
        break;
      case FieldInfo::Mask:
        typeStr = "mask";
        break;
      }
      os << "    type: \"" << typeStr << "\"\n";

      if (f.type == FieldInfo::Register) {
        os << llvm::formatv(
            R"(    reg_file: "{0}"
    reg_width: {1}
    is_input: {2}
    is_output: {3}
    mem_input: {4}
    mem_output: {5}
    constrained: {6}
)",
            f.regFile, f.regWidth, f.isInput ? "true" : "false",
            f.isOutput ? "true" : "false", f.memInput ? "true" : "false",
            f.memOutput ? "true" : "false", f.constrained ? "true" : "false");
      }

      if (f.type == FieldInfo::Immediate) {
        os << "    signed: " << (f.isSigned ? "true" : "false") << "\n";
      }
      os.unindent();
    }
    os.unindent();
  }
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct AnalyzeISAXPass : public shortnail::AnalyzeISAXBase<AnalyzeISAXPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Expect exactly one ISAXOp (merged ISAX module)
    auto isaxOps = llvm::to_vector(moduleOp.getOps<coredsl::ISAXOp>());
    if (isaxOps.size() != 1) {
      moduleOp.emitError() << "expected exactly 1 coredsl.ISAXOp, found "
                           << isaxOps.size();
      return signalPassFailure();
    }

    if (outputPath.empty()) {
      moduleOp.emitError("'output' option is required for analyze-isax");
      return signalPassFailure();
    }

    auto isaxOp = isaxOps[0];

    // Unroll all scf::ForOp loops with constant bounds once, globally, so that
    // per-iteration GPR writes become individually visible as constant-index
    // writes.  loopUnrollFull silently fails for non-constant-bounded loops.
    {
      SmallVector<scf::ForOp> loops;
      isaxOp.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
      for (auto forOp : loops)
        (void)mlir::loopUnrollFull(forOp);
    }

    // CSE + Canonicalize to simplify unrolled IV arithmetic (e.g. fold
    // arith.addi(0, N) → arith.constant N) so tryFoldToConstant can identify
    // concrete register indices.
    {
      PassManager pm(isaxOp->getContext());
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());
      if (failed(pm.run(isaxOp)))
        return signalPassFailure();
    }

    ISAXInfo isax;
    isax.name = cleanName(isaxOp.getName());

    for (auto instOp : isaxOp.getOps<coredsl::InstructionOp>()) {
      InstructionInfo instInfo;
      analyzeInstruction(instOp, isaxOp, instInfo);
      isax.instructions.push_back(std::move(instInfo));
    }

    // Write YAML output
    std::error_code EC;
    llvm::raw_fd_ostream outFile(outputPath, EC);
    if (EC) {
      isaxOp.emitError("Failed to open output file: ") << outputPath;
      return signalPassFailure();
    }
    mlir::raw_indented_ostream indentedOS(outFile);
    writeYAML(isax, indentedOS);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//

namespace mlir {
namespace shortnail {
std::unique_ptr<Pass> createAnalyzeISAXPass() {
  return std::make_unique<AnalyzeISAXPass>();
}
} // namespace shortnail
} // namespace mlir
