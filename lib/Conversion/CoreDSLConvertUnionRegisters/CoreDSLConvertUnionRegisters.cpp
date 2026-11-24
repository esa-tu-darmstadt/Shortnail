#include "mlir/Transforms/DialectConversion.h"
#include "shortnail/Conversion/Passes.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HWArith/HWArithOps.h"

namespace mlir {
namespace shortnail {
#define GEN_PASS_DEF_COREDSLCONVERTUNIONREGISTERS
#include "shortnail/Conversion/Passes.h.inc"
} // namespace shortnail
} // namespace mlir

using namespace mlir;
using namespace mlir::shortnail;
using namespace circt;

namespace {

// Examples:
// struct { i32 a, b } and struct {i16 x, y; i32 z}
// => { i16 x_a_upper, y_a_lower; i32 b_z }
// struct { i32 a, b } and i32 x
// => struct { i32 a_x, b }
// struct { i16 a, b, c } and i64 x
// => struct { i16 a_x, b_x, c_x, x_lower }

// IDEA: descend recursively into the smaller item
// struct { struct {i16 a, b}, i32 c } and i64 x
// => first descend into struct 1 then 2 to split up

// TODO: maybe avoid std::string allocation (maybe via Twine)
std::string appendBitRangeToName(hw::StructType::FieldInfo field, unsigned allocatedBits, unsigned bitsToAllocate) {
  // If we allocate the entire member, just use the name
  if (allocatedBits == 0 && bitsToAllocate == field.type.getIntOrFloatBitWidth()) {
    return field.name.str();
  }
  return field.name.str() + "_" + std::to_string(allocatedBits) + "_" + std::to_string(allocatedBits + bitsToAllocate);
}

std::string combineMemberNames(unsigned bitsToAllocate, hw::StructType::FieldInfo field1, unsigned allocatedBits1, hw::StructType::FieldInfo field2, unsigned allocatedBits2) {
  return appendBitRangeToName(field1, allocatedBits1, bitsToAllocate) + "__" + appendBitRangeToName(field2, allocatedBits2, bitsToAllocate);
}

// TODO:
// - generalize to more than 2 lists
// - need to catalouge which members are now split up into which new ones
SmallVector<hw::StructType::FieldInfo> mergeSubsequentIntegerTypes(ArrayRef<hw::StructType::FieldInfo> types1, ArrayRef<hw::StructType::FieldInfo> types2) {
  SmallVector<hw::StructType::FieldInfo> result;
  result.reserve(std::max(types1.size(), types2.size()));
  auto it1 = types1.begin();
  unsigned member1AlreadyAllocatedBits = 0;
  auto it2 = types2.begin();
  unsigned member2AlreadyAllocatedBits = 0;
  auto *ctx = it1->type.getContext();
  while (it1 != types1.end() && it2 != types2.end()) {
    assert(it1->type.getIntOrFloatBitWidth() > member1AlreadyAllocatedBits);
    assert(it2->type.getIntOrFloatBitWidth() > member2AlreadyAllocatedBits);
    unsigned remainingWidth1 = it1->type.getIntOrFloatBitWidth() - member1AlreadyAllocatedBits;
    unsigned remainingWidth2 = it2->type.getIntOrFloatBitWidth() - member2AlreadyAllocatedBits;
    if (remainingWidth1 == remainingWidth2) {
      auto name = combineMemberNames(remainingWidth1, *it1, member1AlreadyAllocatedBits, *it2, member2AlreadyAllocatedBits);
      result.push_back({StringAttr::get(ctx, name), IntegerType::get(ctx, remainingWidth1, IntegerType::Unsigned)});
      member1AlreadyAllocatedBits = 0;
      member2AlreadyAllocatedBits = 0;
      ++it1;
      ++it2;
    } else {
      unsigned *smallerAlreadyAllocated;
      unsigned *largerAlreadyAllocated;
      ArrayRef<hw::StructType::FieldInfo>::iterator *smallerIt;
      if (member1AlreadyAllocatedBits < member2AlreadyAllocatedBits) {
        smallerAlreadyAllocated = &member1AlreadyAllocatedBits;
        largerAlreadyAllocated = &member2AlreadyAllocatedBits;
        smallerIt = &it1;
      } else {
        smallerAlreadyAllocated = &member2AlreadyAllocatedBits;
        largerAlreadyAllocated = &member1AlreadyAllocatedBits;
        smallerIt = &it2;
      }

      auto name = combineMemberNames(*smallerAlreadyAllocated, *it1, member1AlreadyAllocatedBits, *it2, member2AlreadyAllocatedBits);
      result.push_back({StringAttr::get(ctx, name), IntegerType::get(ctx, *smallerAlreadyAllocated, IntegerType::Unsigned)});
      *largerAlreadyAllocated += *smallerAlreadyAllocated;
      ++*smallerIt;
      *smallerAlreadyAllocated = 0;
    }
  }
  // Handle nonempty results
  if (it1 != types1.end() || it2 != types2.end()) {
    ArrayRef<hw::StructType::FieldInfo>::iterator it, end;
    unsigned alreadyAllocated;
    if (it1 != types1.end()) {
      it = it1;
      end = types1.end();
      alreadyAllocated = member1AlreadyAllocatedBits;
    } else {
      it = it2;
      end = types2.end();
      alreadyAllocated = member2AlreadyAllocatedBits;
    }
    const unsigned toAllocate = it->type.getIntOrFloatBitWidth() - alreadyAllocated;
    auto name = appendBitRangeToName(*it, alreadyAllocated, toAllocate);
    result.push_back({StringAttr::get(ctx, name), IntegerType::get(ctx, toAllocate, IntegerType::Unsigned)});
    ++it;
    while (it != end) {
      result.push_back(*it);
      ++it;
    }
  }
  return result;
}

Type convertUnionType(hw::UnionType type) {
  assert(type.getElements().size() == 2);
  auto field1 = type.getElements()[0];
  assert(field1.offset == 0);
  auto field2 = type.getElements()[1];
  assert(field2.offset == 0);

  auto *ctx = type.getContext();

  auto structType1 = dyn_cast<hw::StructType>(field1.type);
  auto structType2 = dyn_cast<hw::StructType>(field2.type);
  if (structType1 && structType2) {
    auto res = mergeSubsequentIntegerTypes(structType1.getElements(), structType2.getElements());
    return hw::StructType::get(ctx, res);
  } else if (structType1 || structType2) {
    hw::StructType structType;
    hw::UnionType::FieldInfo otherField;
    if (structType1) {
      structType = structType1;
      otherField = field2;
    } else {
      structType = structType2;
      otherField = field1;
    }
    auto integerType = cast<IntegerType>(otherField.type);
    hw::StructType::FieldInfo singleInfo{otherField.name, integerType};
    auto res = mergeSubsequentIntegerTypes(structType.getElements(), singleInfo);
    return hw::StructType::get(ctx, res);
  } else {
    // TODO:
  }
}

// NOTE: this is only top-level, so struct types nested within unions need to be
// handled separately
hw::StructType convertStructType(hw::StructType type) {
  SmallVector<hw::StructType::FieldInfo> fields;
  for (hw::StructType::FieldInfo field : type.getElements()) {
    Type type;
    if (auto structType = dyn_cast<hw::StructType>(field.type)) {
      type = convertStructType(structType);
    } else if (auto unionType = dyn_cast<hw::UnionType>(field.type)) { 
      type = convertUnionType(unionType);
    } else {
      type = field.type;
      // must be hwarith integer
      assert(isa<IntegerType>(type) && cast<IntegerType>(type).getSignedness() != IntegerType::Signless);
    }
    fields.push_back({field.name, type});
  }
  return hw::StructType::get(type.getContext(), fields);
}

struct UnionConversionPattern : public OpConversionPattern<coredsl::RegisterOp> {
  LogicalResult matchAndRewrite(coredsl::RegisterOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    if (auto unionType = dyn_cast<hw::UnionType>(op.getRegType())) {
      auto newType = convertUnionType(unionType);
      // TODO:
    } else if (auto structType = dyn_cast<hw::StructType>(op.getRegType())) {
      auto newType = convertStructType(structType);
      if (newType == structType) {
        // TODO: is returning failure right here?
        return LogicalResult::failure();
      }
      // TODO: search for unions
    }
    return LogicalResult::failure();
  }
};

struct CoreDSLConvertUnionRegisters
    : public mlir::shortnail::impl::CoreDSLConvertUnionRegistersBase<
          CoreDSLConvertUnionRegisters> {
  using CoreDSLConvertUnionRegistersBase::CoreDSLConvertUnionRegistersBase;

  void runOnOperation() override {
    // IDEA:
    // 1. Convert all union registers types
    // 2. Convert all reads or writes that now write the wrong type to first read the converted struct / union, then convert to the one we want
  }
};

} // namespace
