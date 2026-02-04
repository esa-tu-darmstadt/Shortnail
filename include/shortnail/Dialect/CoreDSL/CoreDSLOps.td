//===- CoreDSLOps.td - CoreDSL dialect ops -----------------*- tablegen -*-===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef SHORTNAIL_DIALECT_COREDSL_COREDSLOPS_TD
#define SHORTNAIL_DIALECT_COREDSL_COREDSLOPS_TD

include "CoreDSLInterfaces.td"
include "circt/Dialect/HWArith/HWArithTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def CoreDSL_ISAXOp : CoreDSL_Op<"isax", [SingleBlock, NoTerminator, NoRegionArguments, IsolatedFromAbove, SymbolTable]> {
  let summary = "Container for coredsl instructions.";
  let description = [{
    This operation represents an ISAX as a collection of `coredsl.instruction`s, `coredsl` state elements and functions.
  }];

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{ $name $body attr-dict }];
}

def CoreDSL_InstructionOp : CoreDSL_Op<"instruction",
    [HasParent<"coredsl::ISAXOp">, IsolatedFromAbove, FunctionOpInterface, CallableOpInterface, Symbol, OpAsmOpInterface]> {
  let summary = "Defines an instruction.";
  let description = [{
    This operation defines a CoreDSL [instruction](https://github.com/Minres/CoreDSL/wiki/Structure-and-concepts#instructions)
    with the given symbol name, binary encoding, and behavior.

    The operation's argument list naturally captures the encoding from MSB to
    LSB as a sequence of fields and patterns. The single-block region describes
    the instruction's behavior, which is terminated by either a `coredsl.end` or
    a `coredsl.spawn` operation.

    Example:
    ```
    coredsl.instruction @ADDI(%imm : ui12, %rs1 : ui5, "000", %rd : ui5, "0010011") {
      %arg1 = coredsl.get @X[%rs1 : ui5] : ui32
      %simm = coredsl.cast %imm : ui12 to si12
      %add = hwarith.add %arg1, %simm : (ui32, si12) -> si34
      %trunc = coredsl.bitextract %add[31:0] : (si34) -> ui32
      coredsl.set @X[%rd : ui5] = %trunc : ui32
      coredsl.end
    }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    StrArrayAttr:$encoding
  );

  let regions = (region AnyRegion);

  let extraClassDeclaration = [{
    // FunctionOpInterface
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    LogicalResult verifyType();

    // CallableOpInterface
    Region* getCallableRegion()         { return &getRegion(); }
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }
    mlir::ArrayAttr getCallableArgAttrs() { return nullptr; }
    mlir::ArrayAttr getCallableResAttrs() { return nullptr; }

    // OpAsmOpInterface
    void getAsmBlockArgumentNames(::mlir::Region &region,
                                  ::mlir::OpAsmSetValueNameFn setNameFn);

    // Utility
    struct EncodingField { unsigned lsb; unsigned msb; };
    void getEncodingFields(SmallVectorImpl<EncodingField> &fields);
    StringAttr getEncodingMask();

    StringRef getInstructionArgumentName(size_t argNo);
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def CoreDSL_SpawnOp : CoreDSL_Op<"spawn",
    [HasParent<"InstructionOp">, SingleBlock, NoRegionArguments, Terminator]> {
  let summary = "Initiates decoupled execution.";
  let description = [{
    This operation defines a [spawn-block](https://github.com/Minres/CoreDSL/wiki/Statements#spawn-blocks).
    Its single-block region contains the decoupled behavior, which must be
    terminated by a `coredsl.end` op (i.e. spawn-blocks cannot be nested).

    Example:
    ```
    coredsl.instruction @SQRT("000000000000", %rs1 : ui5, "000", %rd : ui5, "0001011") {
      %arg1 = coredsl.get @X[%rs1 : ui5] : ui32
      coredsl.spawn {
        %res = func.call @sqrt_cordic_fixp(%arg1) : (ui32) -> ui32
        coredsl.set @X[%rd : ui5] = %res : ui32
        coredsl.end
      }
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $body attr-dict
  }];
}

def CoreDSL_AlwaysOp : CoreDSL_Op<"always",
    [HasParent<"coredsl::ISAXOp">, Symbol, SingleBlock, IsolatedFromAbove, NoRegionArguments]> {
  let summary = "Defines an always-block.";
  let description = [{
    This operation defines an [always-block](https://github.com/Minres/CoreDSL/wiki/Structure-and-concepts#always-blocks)
    with the given name. The behavior, described in the single-block region,
    will be executed repeatedly at the same rate as instructions are fetched,
    and in parallel with regular instructions and other always-blocks.

    Example:
    ```
    coredsl.always @implicit_pc_increment {
      %oldpc = coredsl.get @PC : ui32
      %c1 = hwarith.constant 4 : ui3
      %incr = hwarith.add %oldpc, %c1 : (ui32, ui3) -> ui33
      %newpc = coredsl.cast %incr : ui33 to ui32
      coredsl.set @PC = %newpc : ui32
      coredsl.end
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name $body attr-dict
  }];
}

def CoreDSL_EndOp : CoreDSL_Op<"end",
    [ParentOneOf<["InstructionOp", "AlwaysOp", "SpawnOp"]>, Terminator]> {
  let summary = "No-op terminator.";
  let assemblyFormat = [{
    attr-dict
  }];
}

// Base class for memory operations accessible via coredsl.get and coredsl.set
class GetSettableOp<string mnemonic, string prefixAssemblyFormat = "", string suffixAssemblyFormat = "attr-dict", dag moreArgs = (ins), list<Trait> traits = []> :
      CoreDSL_Op<mnemonic, traits # [Symbol, DeclareOpInterfaceMethods<GetSetOpInterface>]> {

  let arguments = !con((ins SymbolNameAttr:$sym_name, UnitAttr:$isConst, UnitAttr:$isVolatile), moreArgs);

  let assemblyFormat = prefixAssemblyFormat # " oilist(`const` $isConst | `volatile` $isVolatile) $sym_name " # suffixAssemblyFormat;
}


// Register access modes
def DefaultRegisterFile  : I64EnumAttrCase<"core_x", 0b0001>;
def PCRegister  : I64EnumAttrCase<"core_pc", 0b0010>;
def FPRegister  : I64EnumAttrCase<"core_fp", 0b0100>;
def LocalRegister  : I64EnumAttrCase<"local", 0b1000>;

let cppNamespace = "::mlir::coredsl" in
def RegisterAccessMode : I64EnumAttr<
    "RegisterAccessMode",
    "coredsl.register access mode",
    [DefaultRegisterFile, PCRegister, FPRegister, LocalRegister]>;

def IndexArrayAttr
  : TypedArrayAttrBase<IndexAttr, "Index array attribute">;

def CoreDSL_RegisterOp : 
    GetSettableOp<"register", /*prefixAssemblyFormat=*/"$accessMode",
                  /*suffixAssemblyFormat=*/"(`[` $numElements^ `]`)? `` custom<Initializer>($initializer) `:` $regType attr-dict",
                  /*moreArgs=*/(ins OptionalAttr<IndexAttr>:$numElements,
                   OptionalAttr<IndexArrayAttr>:$initializer, TypeAttr:$regType,
                   RegisterAccessMode:$accessMode)> {
  let summary = "Declares a register (file).";
  let description = [{
    This operation declares a [CoreDSL register](https://github.com/Minres/CoreDSL/wiki/Structure-and-concepts#registers)
    with the given name, type, flags, and *access mode*. Optionally, a number of
    elements and and an initializer can be specified. 
    
    A register can be marked as read-only by the `const` keyword. The `volatile`
    keyword is accepted but ignored by the lowering passes.

    The following access modes are supported:

    - `core_x`: Access the core's general-purpose register file via SCAIE-V.
    - `core_pc`: Access the core's program counter via SCAIE-V.
    - `local`: Access a custom register or register file instantiated by SCAIE-V.
    - (`core_fp`: Reserved for future F/D-extension use).

    Examples:
    ```
    coredsl.register core_x @X[32] : ui32
    coredsl.register core_pc @PC : ui32
    coredsl.register local @MY_REG = 44 : ui14
    coredsl.register const local @MY_REG_FIELD[3] = [282, 999, 151] : ui37
    ```
  }];

  let extraClassDeclaration = [{
    bool isShimed();
    bool isRegField();
    uint64_t getSize();
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

// Address space access protocols
def MemAddrSpace  : I64EnumAttrCase<"core_mem", 0b0001>;
def CSR_AddrSpace : I64EnumAttrCase<"core_csr", 0b0010>;
def AXI_AddrSpace : I64EnumAttrCase<"axi4mm", 0b0100>;
def IRQ_Signal    : I64EnumAttrCase<"wire", 0b1000>;

let cppNamespace = "::mlir::coredsl" in
def AddressSpaceAccessMode : I64EnumAttr<
    "AddressSpaceAccessMode",
    "coredsl.addrspace access protocol",
    [MemAddrSpace, CSR_AddrSpace, AXI_AddrSpace, IRQ_Signal]>;

def CoreDSL_AddressSpaceOp : 
    GetSettableOp<"addrspace", /*prefixAssemblyFormat=*/"$accessMode",
                  /*suffixAssemblyFormat=*/"`:` (`(`$addrType^`)` `->`)? $resType attr-dict",
                  /*moreArgs=*/(ins OptionalAttr<TypeAttr>:$addrType, TypeAttr:$resType,
                   AddressSpaceAccessMode:$accessMode)> {
  let summary = "Declares an address space.";
  let description = [{
    This operation declares a [CoreDSL address space](https://github.com/Minres/CoreDSL/wiki/Structure-and-concepts#address-spaces)
    with the given name, element type, flags, and *access mode*. The optional
    unsigned address type determines the number of addressable elements.
    
    An address space can be marked as read-only by the `const` keyword. The
    `volatile` keyword is accepted but ignored by the lowering passes.

    The following access modes are supported:

    - `core_mem`: Access to the core's main memory via SCAIE-V.
    - `core_csr`: Access to the core's control-status registers via SCAIE-V.
    - (`axi4mm` and `wire` are reserved for future use.)

    Examples:
    ```
    coredsl.addrspace core_mem @MEM : (ui32) -> ui8
    coredsl.addrspace core_csr @CSR : (ui12) -> ui32
    ```
  }];

  let hasVerifier = 1;
}

class CoreDSL_AccessOp<string mnemonic, string lhsAssemblyFormat, string rhsAssemblyFormat,
                       dag additionalArgs = (ins), dag res = (outs), string extraClassDecls = "",
                       string isRangeAccessOptional = "false", list<Trait> traits = [],
                       string folderPrototype = "OpFoldResult $cppClass::fold(FoldAdaptor adaptor)",
                       string folderPosRes = "getResult()",
                       string folderNegRes = "{}">
    : CoreDSL_Op<mnemonic, traits> {

  let arguments = !con((ins Optional<AllOfType<[HWArithIntegerType, AnyUnsignedInteger]>>:$base, 
                        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$from,
                        OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$to), additionalArgs);
  let results = res;

  let assemblyFormat = lhsAssemblyFormat # "`` custom<RangedAccess>(\"" # isRangeAccessOptional # "\", $base, type($base), $from, $to) " # rhsAssemblyFormat;

  let extraClassDeclaration = [{
    unsigned getAccessWidth();
    bool hasSingleIdxAccess();
    bool reversedAccessOrder();
  }] # extraClassDecls;

  let extraClassDefinition = [{
    unsigned $cppClass::getAccessWidth() {
      if (hasSingleIdxAccess()) {
        return 1;
      }
      int64_t width = static_cast<int64_t>(getFrom()->getZExtValue()) -
                      static_cast<int64_t>(getTo()->getZExtValue());
      return std::abs(width) + 1;
    }

    bool $cppClass::hasSingleIdxAccess() {
      return !getTo() || !getFrom() || getFrom()->getZExtValue() == getTo()->getZExtValue();
    }

    bool $cppClass::reversedAccessOrder() {
      return getTo() && getFrom() && getFrom()->getZExtValue() < getTo()->getZExtValue();
    }

    }] # folderPrototype # [{ {
      if (hasSingleIdxAccess() && getFrom()) {
        bool changed = false;
        // canonicalize single bit accesses such as val[%base, 0:0] to val[%base]
        if (getBase() && getFrom()->getZExtValue() == 0) {
          removeFromAttr();
          changed = true;
        }

        // canonicalize single bit accesses such as val[2:2] to val[2]
        if (getTo()) {
          removeToAttr();
          changed = true;
        }
        if (changed)
          return }] # folderPosRes # [{;
      }

      return }] # folderNegRes # [{;
    }
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def CoreDSL_GetOp : CoreDSL_AccessOp<"get", "$sym", "`:` type($result) attr-dict",
                                    (ins FlatSymbolRefAttr:$sym), 
                                    (outs HWArithIntegerType:$result),
                                    /*extraClassDecls=*/"Operation* resolveSymbol(); std::optional<MemInfo> getMemInfo();",
                                    /*isRangeAccessOptional=*/"true"> {
  let summary = "Reads from architectural state.";
  let description = [{
    This operation accesses the register, address space or alias denoted by the
    given symbol. If the architectural state item has multiple elements (e.g. 
    a register file), then either a single element or a range of elements can be
    retrieved. In the latter case, the elements are optionally reversed and
    concatened according the rules of the [CoreDSL subscript operator](https://github.com/Minres/CoreDSL/wiki/Expressions#range-operator),
    though here we use a slightly different notation with a base index and
    constant from/to offsets.

    Examples:
    ```
    %pc  = coredsl.get @X : ui32
    %addr = coredsl.get @X[%rs1 : ui5] : ui32
    %half_le = coredsl.get @MEM[%addr : ui32, 1:0] : ui16 // little-endian
    %word_be = coredsl.get @MEM[%addr : ui32, 0:3] : ui32 // big-endian
    ```
  }];

  let hasCanonicalizeMethod = 1;
}

def CoreDSL_SetOp : CoreDSL_AccessOp<"set", "$sym", "`=` $value `:` type($value) attr-dict",
                                    (ins FlatSymbolRefAttr:$sym, HWArithIntegerType:$value), 
                                    (outs), /*extraClassDecls=*/"Operation* resolveSymbol(); std::optional<MemInfo> getMemInfo();",
                                    /*isRangeAccessOptional=*/"true", [],
                                    /*folderPrototype=*/"LogicalResult $cppClass::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult> &results)",
                                    /*folderPosRes=*/"success()",
                                    /*folderNegRes=*/"failure()"> {
  let summary = "Writes to architectural state.";
  let description = [{
    This operation accesses the register, address space or alias denoted by the
    given symbol. If the architectural state item has multiple elements (e.g. 
    a register file), then either a single element or a range of elements can be
    stored. In the latter case, the elements are optionally reversed and
    concatened according the rules of the [CoreDSL subscript operator](https://github.com/Minres/CoreDSL/wiki/Expressions#range-operator),
    though here we use a slightly different notation with a base index and
    constant from/to offsets.

    Examples:
    ```
    coredsl.set @PC = %0 : ui32
    coredsl.set @X[%rd : ui5] = %1 : ui32
    coredsl.set @MEM[%addr : ui32, 1:0] = %2 : ui16 // little-endian
    coredsl.set @MEM[%addr : ui32, 0:3] = %3 : ui32 // big-endian
    ```
  }];

  let hasVerifier = 1;
}

class CoreDSL_BitAccessOp<string mnemonic, string rhsAssemblyFormat,
                          dag additionalArgs = (ins), list<Trait> traits = []>
      :  CoreDSL_AccessOp<mnemonic, "$value", rhsAssemblyFormat,
                          !con(additionalArgs, (ins HWArithIntegerType:$value)),
                          (outs HWArithIntegerType:$result), /*extraClassDecls=*/"",
                          /*isRangeAccessOptional=*/"false", traits # [Pure]> {
}


def CoreDSL_BitExtractOp
    : CoreDSL_BitAccessOp<"bitextract",
                          "`:` functional-type($value, $result) attr-dict"> {
  let summary = "Extracts bits from a scalar value.";
  let description = [{
    This operation extracts a single bit or a range of bits from the given
    operand. In the latter case, the bits are optionally reversed and concatened
    according the rules of the [CoreDSL subscript operator](https://github.com/Minres/CoreDSL/wiki/Expressions#range-operator),
    though here we use a slightly different notation with a base index and
    constant from/to offsets.

    Examples:
    ```
    coredsl.bitextract %0[63] : (si64) -> ui1 // sign-bit
    coredsl.bitextract %1[%2 : ui5] : (ui32) -> ui1
    coredsl.bitextract %3[23:16] : (ui32) -> ui8
    coredsl.bitextract %4[%5 : ui4, 0:3] : (si16) -> ui4 // reversed nibble
    ```
  }];

  let hasCanonicalizeMethod = 1;
}

def CoreDSL_BitSetOp
    : CoreDSL_BitAccessOp<
          "bitset",
          "`=` $rhs `:` custom<FancyFunctionalType>(type($result), "
          "type($value), type($rhs)) attr-dict",
          (ins HWArithIntegerType: $rhs)> {
  let summary = "Sets bits in a scalar value.";
  let description = [{
    This operation updates a single bit or a range of bits in first operands
    with the second operand. In the latter case, the bits are optionally
    reversed and concatened according the rules of the [CoreDSL subscript operator](https://github.com/Minres/CoreDSL/wiki/Expressions#range-operator),
    though here we use a slightly different notation with a base index and
    constant from/to offsets.

    Examples:
    ```
    coredsl.bitset %0[42] = %1 : (si64, ui1) -> si64
    coredsl.bitset %2[%3 : ui5] = %4 : (ui32, ui1) -> ui32
    coredsl.bitset %5[23:16] = %6 : (ui32, ui8) -> ui32
    coredsl.bitset %7[%8 : ui4, 0:3] = %9 : (si16, ui4) -> si16 // reversed nibble
    ```
  }];
}

def CoreDSL_AliasOp : CoreDSL_Op<"alias", [Symbol]> {
  let summary = "Declares an alias.";
  let description = [{
    This operation declares a [CoreDSL alias](https://github.com/Minres/CoreDSL/wiki/Structure-and-concepts#aliases)
    with the given name to another architectural state item, referencing either
    a single element or a range of elements. The optional `const` keyword marks
    the alias as read-only. Aliases to read-only architectural state must be
    `const`. The `volatile` keyword is accepted, but ignored in the lowering
    passes.

    Examples:
    ```
    coredsl.alias @A1 = @REG
    coredsl.alias @A2 = @REGFIELD[6]
    coredsl.alias @A3 = @MEM[7:0]
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$ref,
                   UnitAttr:$isConst, UnitAttr:$isVolatile, 
                   OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$from,
                   OptionalAttr<ConfinedAttr<IndexAttr, [IntNonNegative]>>:$to);

  let extraClassDeclaration = [{
    GetSetOpInterface resolveSymbol();
    std::optional<MemInfo> getMemInfo();
    unsigned getAliasRange();
    uint64_t getRangeStart();
  }];

  let extraClassDefinition = [{
    unsigned $cppClass::getAliasRange() {
      if (!getTo() && !getFrom()) {
        return 0;
      } else if (!getTo()) {
        return 1;
      } else {
        return getFrom()->getZExtValue() - getTo()->getZExtValue() + 1;
      }
    }
    uint64_t $cppClass::getRangeStart() {
      if (auto from = getFrom()) {
        if (auto to = getTo())
          return to->getZExtValue();
        else
          return from->getZExtValue();
      }
      return 0;
    }
  }];

  let assemblyFormat = [{
    oilist(`const` $isConst | `volatile` $isVolatile) $sym_name `=` $ref `` custom<IdxRange>($from, $to)
    attr-dict
  }];

  let hasVerifier = 1;
}

// Base class for binary operators.
class CoreDSL_BinOp<string mnemonic, list<Trait> traits = []> :
      CoreDSL_Op<mnemonic, traits # [Pure]> {
  let arguments = (ins HWArithIntegerType:$lhs, HWArithIntegerType:$rhs);
  let results = (outs HWArithIntegerType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

// Binary operator with result type inference.
class TIBinOp<string mnemonic, list<Trait> traits = []> :
      CoreDSL_BinOp<mnemonic, traits # [InferTypeOpInterface]> {

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];
}

def CoreDSL_ConcatOp : TIBinOp<"concat"> {
  let summary = "Concatenates two values.";
  let description = [{
    This operation represents the [CoreDSL `::`-operator](https://github.com/Minres/CoreDSL/wiki/Expressions#concatenation)
    and concatenates its two operands. The result is always unsigned.

    Example:
    ```
    %2 = coredsl.concat %0, %1 : ui8, si3 // yields ui11
    ```
  }];
}

// Arithmetic Operations.
def ModOp : TIBinOp<"mod"> {
  let summary = "Modulus/remainder operator.";
  let description = [{
    This operation computes the modulus of the operands. The result type is
    determined by the type rules of the [CoreDSL `%`-operator](https://github.com/Minres/CoreDSL/wiki/Expressions#modulusremainder).

    Examples:
    ```
    %2 = coredsl.mod %0, %1 : ui2, ui8 // yields ui2
    %5 = coredsl.mod %3, %4 : si5, ui2 // yields si3
    ```
  }];
}

// Bitwise Operations.
class BitOp<string mnemonic> : TIBinOp<mnemonic, [Commutative]> {
  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(MLIRContext *context,
                                              std::optional<Location> loc,
                                              ValueRange operands,
                                              DictionaryAttr attrs,
                                              mlir::OpaqueProperties properties,
                                              mlir::RegionRange regions,
                                              SmallVectorImpl<Type> &results) {
      return ::mlir::coredsl::inferBitOpTypes(context, loc, operands, attrs,
                                              properties, regions, results);
    }
  }];
}
def OrOp : BitOp<"or"> {
  let summary = "Bitwise OR operator.";
  let description = [{
    This operation computes the bitwise OR of the operands. Differently signed
    and sized operands are allowed; the smaller of the operands will be zero- or
    sign-extended to the result type with is determined by the corresponding
    [CoreDSL type rules](https://github.com/Minres/CoreDSL/wiki/Expressions#bitwise-and-or-xor).

    Example:
    ```
    %2 = coredsl.or %0, %1 : ui3, si2 // yields si3
    ```
  }];
}
def XorOp : BitOp<"xor"> {
  let summary = "Bitwise XOR operator.";
  let description = [{
    This operation computes the bitwise XOR of the operands. Differently signed
    and sized operands are allowed; the smaller of the operands will be zero- or
    sign-extended to the result type with is determined by the corresponding
    [CoreDSL type rules](https://github.com/Minres/CoreDSL/wiki/Expressions#bitwise-and-or-xor).

    Example:
    ```
    %2 = coredsl.xor %0, %1 : ui3, si2 // yields si3
    ```
  }];
}
def AndOp : BitOp<"and"> {
  let summary = "Bitwise AND operator.";
  let description = [{
    This operation computes the bitwise AND of the operands. Differently signed
    and sized operands are allowed; the smaller of the operands will be zero- or
    sign-extended to the result type with is determined by the corresponding
    [CoreDSL type rules](https://github.com/Minres/CoreDSL/wiki/Expressions#bitwise-and-or-xor).

    Example:
    ```
    %2 = coredsl.and %0, %1 : ui3, si2 // yields si3
    ```
  }];
}

// Shift Operations.
def ShiftLeftOp : CoreDSL_BinOp<"shift_left", [TypesMatchWith<"lhs type matches result type",
                                                      "lhs", "result", "$_self">]> {
  let summary = "Left shift.";
  let description = [{
    This operation performs a left-shift on the first operand by the amount
    given in the second operand. The result type is equal to the type of the
    first operand. In accordance to the [CoreDSL `<<`-operator](https://github.com/Minres/CoreDSL/wiki/Expressions#shift-operations),
    negative shift amounts are legal and result in a right shift by the absolute
    value of the amount.

    Example:
    ```
    %shl = coredsl.shift_left %arg1, %shamt : ui32, ui5
    ```
  }];
}
                                                    
def ShiftRightOp : CoreDSL_BinOp<"shift_right", [TypesMatchWith<"lhs type matches result type",
                                                        "lhs", "result", "$_self">]> {
  let summary = "Right shift.";
  let description = [{
    This operation performs a right-shift on the first operand by the amount
    given in the second operand. The result type is equal to the type of the
    first operand. Depending on the signedness of the first operand, either a
    logical or arithmetic right shift is performed.  In accordance to the
    specification of the [CoreDSL `<<`-operator](https://github.com/Minres/CoreDSL/wiki/Expressions#shift-operations),
    negative shift amounts are legal and result in a left shift by the absolute
    value of the amount.

    Examples:
    ```
    %lshr = coredsl.shift_right %0, %shamt : ui32, ui5
    %ashr = coredsl.shift_right %1, %shamt : si17, ui5
    ```
  }];

  let extraClassDeclaration = [{
    bool isArithmeticShift();
  }];
  let extraClassDefinition = [{
    bool $cppClass::isArithmeticShift() {
      return cast<IntegerType>(getLhs().getType()).isSigned();
    }
  }];
}

def CastOp : CoreDSL_Op<"cast", [Pure]> {
  let summary = "Generic cast operation.";
  let description = [{
    This operations allows casts between arbitrary signed and unsigned types,
    and as a special case, casts single-bit values (`si1`/`ui1`) to the signless
    `i1` to provide a bridge to standard MLIR dialects such as `scf`.

    Casts are implemented as truncations, or zero-/sign-extensions depending on
    the type of the source operand.

    Examples:
    ```
    %1 = coredsl.cast %0 : ui8 to si8
    %3 = coredsl.cast %2 : ui1 to i1
    %5 = coredsl.cast %4 : si65 to ui32
    ```
  }];

  let arguments = (ins HWArithIntegerType:$value);
  let results = (outs AnyInteger:$result);

  let assemblyFormat = "$value `:` type($value) `to` type($result) attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : CoreDSL_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["SwitchOp"]>]> {
  let summary = "loop yield and termination operation";
  // TODO: update description
  let description = [{
    The `scf.yield` operation yields an SSA value from the SCF dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If `scf.yield` has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the `scf.yield` may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

def SwitchOp : CoreDSL_Op<"switch", [RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"coredsl::YieldOp">,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getRegionInvocationBounds",
                               "getEntrySuccessorRegions"]>]> {
  let summary = "switch-case operation on an index argument";
  let description = [{
    `coredsl.switch` is a modified version of `scf.index_switch` that branches
    to one of the given regions based on the values of the argument and the
    cases. As opposed to `scf.index_switch`, this operation accepts any integer
    as argument.

    The operation always has a "default" region and any number of case regions
    denoted by integer constants. Control-flow transfers to the case region
    whose constant value equals the value of the argument. If the argument does
    not equal any of the case values, control-flow transfer to the "default"
    region.

    Example:

    // TODO: fix when syntax finalized
    ```mlir
    %0 = coredsl.switch %arg0 : ui32 -> ui32
    case 2 {
      %1 = hwarith.constant 10 : ui32
      coredsl.yield %1 : ui32
    }
    case 5 {
      %2 = hwarith.constant 20 : ui32
      coredsl.yield %2 : ui32
    }
    default {
      %3 = hwarith.constant 30 : ui32
      coredsl.yield %3 : ui32
    }
    ```
  }];

  let arguments = (ins AnyInteger:$arg, ArrayAttr:$cases);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$defaultRegion,
                        VariadicRegion<SizedRegion<1>>:$caseRegions);

  let assemblyFormat = [{
    $arg attr-dict `:` type($arg) (`->` type($results)^)?
    custom<SwitchCases>($cases, $caseRegions) `\n`
    `` `default` $defaultRegion
  }];

  let extraClassDeclaration = [{
    /// Get the number of cases.
    unsigned getNumCases();

    /// Get the default region body.
    Block &getDefaultBlock();

    /// Get the body of a case region.
    Block &getCaseBlock(unsigned idx);
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

#endif // SHORTNAIL_DIALECT_COREDSL_COREDSLOPS_TD
