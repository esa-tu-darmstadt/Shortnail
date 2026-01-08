# Shortnail's MLIR-based input format

## Introduction
Shortnail's input is an *elaborated* description of an ISA extension (ISAX) comprised of an arbitrary number of instructions, always blocks, functions, and definitions of architectural state elements. The frontend is responsible to resolve imports, flatten the inheritance hierarchy, and elaborate the parameters and dependent types in a CoreDSL description. It must only include instructions and architectural state elements that shall be included or referenced in the generated hardware module.

### Example

#### CoreDSL
```verilog
import "https://raw.githubusercontent.com/Minres/RISCV_ISA_CoreDSL/master/RV32I.core_desc"

InstructionSet RBNN extends RV32I {
  instructions {
    MAC {
      encoding: 7'd0 :: rs2[4:0] :: rs1[4:0] :: 3'b010 :: rd[4:0] :: 7'b0101011;
      behavior: {
        signed<65> res = (signed) X[rs1] * (signed) X[rs2] + (signed) X[rd];
        if (rd != 0)
          X[rd] = (unsigned<32>) res;
      }
    }
  }
}
```

#### Corresponding input
```mlir
coredsl.isax "RBNN" {
  coredsl.register core_x @X[32] : ui32

  coredsl.instruction @MAC("0000000", %rs2 : ui5, %rs1 : ui5,
                           "010", %rd : ui5, "0101011") {
    %op1_read = coredsl.get @X[%rs1 : ui5] : ui32
    %op1 = coredsl.cast %op1_read : ui32 to si32
    %op2_read = coredsl.get @X[%rs2 : ui5] : ui32
    %op2 = coredsl.cast %op2_read : ui32 to si32
    %mul = hwarith.mul %op1, %op2 : (si32, si32) -> si64
    %op3_read = coredsl.get @X[%rd : ui5] : ui32
    %op3 = coredsl.cast %op3_read : ui32 to si32
    %add = hwarith.add %mul, %op3 : (si64, si32) -> si65
    
    %c0 = hwarith.constant 0 : ui1
    %cond = hwarith.icmp ne %rd, %c0 : ui5, ui1
    scf.if %cond {
      %result = coredsl.bitextract %add[31:0] : (si65) -> ui32
      coredsl.set @X[%rd : ui5] = %result : ui32
    }
    coredsl.end
  }
}
```

### Syntax
Shortnail expects to parse a valid, textual MLIR file. We refer the reader to the [MLIR language reference](https://mlir.llvm.org/docs/LangRef/) for the lexical and syntactical considerations. For the remainder of this document, we focus on the dialects and operations that constitute Shortnail's input format.

### Dialects
| Dialect | Ops | Origin | Use in Shortnail |
| ------- | --- | ------ | --------------- |
| [`builtin`](https://mlir.llvm.org/docs/Dialects/Builtin/) | `module` | MLIR | Generic container for ISAX |
| [`func`](https://mlir.llvm.org/docs/Dialects/Func/) | `func`, `call` | MLIR | Function definitions and calls |
| [`scf`](https://mlir.llvm.org/docs/Dialects/SCFDialect/) | `if`, `while`, `for` | MLIR | Structured control-flow constructs | 
| [`hwarith`](https://circt.llvm.org/docs/Dialects/HWArith/) | all | CIRCT | Basic bit-width-aware arithmetic operations |
| `coredsl` | all | Shortnail | Instructions, registers, address spaces, access to architectural state, additional bit-width-aware arithmetic operations |

## Input format specification

### Types
Shortnail relies on the [MLIR standard](https://mlir.llvm.org/docs/Dialects/Builtin/#integertype) unsigned and signed types, e.g. `ui32` or `si17`. Unfortunately, these types cannot be used in conjunction with the upstream `scf` dialect, hence a conversion to a signless type (e.g. `i1`) is necessary in some situations.

### ISAX structure

#### Root element
The top-level container for the input ISAX is a `coredsl.isax`. Its symbol name is used throughout the compilation flow to prefix identifiers.

```mlir
// Top-level container
coredsl.isax "ISAX_name" {
  // May contain an arbitrary number of the following operations:
  // coredsl.register, coredsl.addrspace
  // coredsl.instruction
  // func.func
}
```

#### Register
A `coredsl.register` represents a named register or register field of a given type. Single-element registers optionally carry an initializer. The *protocol specifier* determines how the register will be implemented and accessed in the generated hardware module.

```mlir
coredsl.register core_x  @X[32] : ui32   // RISC-V standard registers via SCAIE-V
coredsl.register core_pc @PC : ui32      // Program counter via SCAIE-V
coredsl.register local   @RGB[3] : ui10  // ISAX-internal memory module
coredsl.register local   @ACC = 0 : si48 // Zero-initialized ISAX-internal memory module
```

#### Address space
A `coredsl.addrspace` represents a named address space with given element and address types. Again, a protocol specifier selects from a range of implementations.

```mlir
coredsl.addrspace core_mem @MEM : (ui32) -> ui8  // RISC-V main memory via SCAIE-V
coredsl.addrspace core_csr @CSR : (ui12) -> ui32 // RISC-V CSRs via SCAIE-V
```

#### Instruction
The `coredsl.instruction` operation defines an instruction. The operation has a function-like argument list that represents the encoding (first bit is the MSB), and carries a single-block region to represent the behavior. The argument list elements are either string literals (comprised of '0' or '1'), or declare typed operands for the behavior region. The region contains a sequence of other MLIR operations, terminated by a `coredsl.end` operation.

```mlir
// A typical R-type instruction
coredsl.instruction @MAC("0000000", %rs2 : ui5, %rs1 : ui5, "010", %rd : ui5, "0101011") {
  // behavior (other MLIR operations)
  coredsl.end
}
```

#### Always block
NYI.

#### Function
Shortnail supports function definitions with the upstream [`func.func` operation](https://mlir.llvm.org/docs/Dialects/Func/#funcfunc-mlirfuncfuncop). Currently, all function calls are subject to inlining, thus function declarations (i.e. without a body region) are not yet supported.

```mlir
func.func @flipSign(%flip : i1, %val : si32) -> (ui32) {
  // behavior (other MLIR operations)  
  return %res : ui32
}
```

### Behavior regions
The IR described here is valid for instruction behavior, always blocks, and function bodies. In the following code snippets, the definitions of SSA values used as operands may have been elided for brevity.

#### Constant
The `hwarith.constant` operation produces a constant unsigned or signed value.

```mlir
%c42 = hwarith.constant 42 : ui6
%cm7 = hwarith.constant -7 : si9
```

#### Addition, subtraction, multiplication, division
The elementary arithmetic operations are provided by the `hwarith` dialect. The operations accept operands of different widths and signedness, and produce a result according to the CoreDSL type rules.

```mlir
%0 = hwarith.add %10, %11 : (ui1, ui1) -> ui2
%1 = hwarith.sub %12, %13 : (ui8, ui2) -> si9
%2 = hwarith.mul %14, %15 : (ui2, ui5) -> ui7
%3 = hwarith.div %16, %17 : (si9, ui1) -> si9
```

*Note:* A modulo/remainder operation is not yet implemented.

#### And, or, xor
The `coredsl` dialect defines the usual bitwise operations. These are compatible with (un)signed operands and follow the CoreDSL type rules.

```mlir
%0 = coredsl.and %20, %21 : ui8, ui5 // result: ui8
%1 = coredsl.or %22, %23 : si1, si2  // result: si2
%2 = coredsl.xor %24, %25 : ui8, si1 // result: si8
```

#### Shifts
The `coredsl` dialect also defines shift operations that are compatible with the CoreDSL specification. There is only one `shift_right` operation because the distinction between a logical and an arithmetic right shift is made based on the signedness of the first operand.

```mlir
// Left shift, result: ui32
%0 = coredsl.shift_left %30, %31 : ui32, ui5
// Logical right shift (unsigned first operand), result: ui32
%1 = coredsl.shift_right %32, %33 : ui32, ui5
// Arithmetic right shift (signed first operand), result: si32
%2 = coredsl.shift_right %34, %35 : si32, ui5
```

#### Bit-level access
The `coredsl.bitextract` operation retrieves either a single bit or a bit range from its operand. The `coredsl.bitset` operation replaces either a single bit or a range of bits in the first operand with the second operand, and returns the result.

```mlir
%0 = coredsl.bitextract %val1[15] : (si16) -> ui1     // single bit
%1 = coredsl.bitextract %val2[63:32] : (ui64) -> ui32 // bit range
%2 = coredsl.bitset %val3[5:1] = %val4 : (ui10, ui10) -> ui10
```

*Note:* The new range operator with a variable base index will be supported with [!77](https://gitlab.esa.informatik.tu-darmstadt.de/scale4edge/shortnail/-/merge_requests/77).

#### Concatenation
The `coredsl.concat` operation concatenates two values together. The first operand contributes the MSB. The result is always unsigned.

```mlir
%0 = hwarith.constant -1 : si4
%1 = hwarith.constant 0 : ui2
%2 = coredsl.concat %0, %1 : si4, ui2 // result: 6'b111100 (ui6)
```

#### Cast
The `coredsl.cast` operation handles the zero/sign-extension and truncation of values, according to the CoreDSL type rules. In addition, it converts `ui1` values to the signless `i1` type for use with `scf` dialect's operations.

```mlir
%0 = coredsl.cast %50 : ui64 to ui48 // truncation
%1 = coredsl.cast %51 : si2 to si10  // sign-extend
%2 = coredsl.cast %52 : ui3 to ui5   // zero-extend
%3 = coredsl.cast %53 : ui1 to i1    // transition to signless world
```

#### Comparison
The `hwarith.icmp` operation implements the usual integer comparisons (`eq`, `ne`, `lt`, `ge`, `le`, `gt`). The result type is always `i1`.

```mlir
%0 = hwarith.icmp ne %60, %61 : ui5, si12
```

#### Access to architectural state
The `coredsl.get` and `coredsl.set` operations provide uniform access to all architectural state elements.

```mlir
%arg1 = coredsl.get @X[rs1 : ui5] : ui32
%accu = coredsl.get @ACC : si48
%load = coredsl.get @MEM[%addr : ui32] : ui8
coredsl.set @X[%rd : ui5] = %arg1 : ui32
coredsl.set @ACC = %zero : si48
coredsl.set @CSR[%reg : ui12] = %val : ui32
```

*Note:* Support for the new address range operator is currently being investigated.

#### If
Shortnail adopts on the [MLIR modeling](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfif-mlirscfifop) of branches.

```mlir
// cast to signless world required
%cc = coredsl.cast %0 : ui1 to i1

// return a value from the `if` (cf. the ?:-operator in C)
// -> `else` and `yield`terminator mandatory
%7 = scf.if %6 -> (ui32) {
  // behavior that computes `%tv`
  scf.yield %tv : ui32
} else {
  // behavior that computes `%fv`
  %fv = ...
  scf.yield %fv : ui32
}

// no value returned
// -> `else` region and `yield` terminators optional
scf.if %cc {
  // conditional behavior
}
```

#### While and for
Shortnail again adopts the MLIR model of [`while`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfwhile-mlirscfwhileop) and [`for`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-mlirscfforop) loops. This section will be completed once the support matures.

#### Spawn
NYI.
