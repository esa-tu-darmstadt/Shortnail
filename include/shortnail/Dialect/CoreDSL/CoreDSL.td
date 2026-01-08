//===- CoreDSL.td - CoreDSL dialect ------------------------*- tablegen -*-===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef SHORTNAIL_DIALECT_COREDSL_COREDSL_TD
#define SHORTNAIL_DIALECT_COREDSL_COREDSL_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CoreDSL dialect definition.
//===----------------------------------------------------------------------===//

def CoreDSL_Dialect : Dialect {
    let name = "coredsl";
    let summary = "CoreDSL-specific constructs for ISAX descriptions.";
    let description = [{
      The `coredsl` dialect provides the necessary operations to model a CoreDSL
      description of an ISA extensions (ISAX), contained in a named
      `builtin.module`.

      The dialect is designed to be mixed with MLIR's [`scf`](https://mlir.llvm.org/docs/Dialects/SCFDialect/)
      and [`func`](https://mlir.llvm.org/docs/Dialects/Func/) dialects, and
      augments CIRCT's [`hwarith`](https://circt.llvm.org/docs/Dialects/HWArith/)
      dialect.

      The available operations can be classified as follows:

      - `coredsl.instruction`, `coredsl.spawn` and `coredsl.always` are
        containers for high-level, untimed behavior descriptions.
      - `coredsl.register`, `coredsl.addrspace` and `coredsl.alias` represent
        architectural state elements, which are accessed by the `coredsl.get`
        and `.set` ops.
      - The remaining ops implement the bitwidth-aware arithmetic/logic
        operators on MLIR's `uiN`/`siN` types that are missing in the `hwarith`
        dialect.

      Example:
      ```
      coredsl.isax "sparkle" {
        coredsl.register core_x @X[32] : ui32
        coredsl.register local const @ROT_0[4] = [31, 17, 0, 24] : ui8
        coredsl.register local const @ROT_1[4] = [24, 17, 31, 16] : ui8
        coredsl.register local const @RCON[8] =
          [3084996962, 3211876480, 951376470, 844003128,
          3138487787, 1333558103, 3485442504, 3266521405] : ui32
        
        func.func @ROTR(%arg0: ui32, %arg1: ui8) -> ui32 {
          %0 = hwarith.constant 32 : ui6
          %1 = coredsl.shift_right %arg0, %arg1 : ui32, ui8
          %2 = hwarith.sub %0, %arg1 : (ui6, ui8) -> si9
          %3 = coredsl.shift_left %arg0, %2 : ui32, si9
          %4 = coredsl.or %1, %3 : ui32, ui32
          return %4 : ui32
        }
        
        func.func @ELL(%arg0: ui32) -> ui32 {
          %0 = hwarith.constant 16 : ui5
          %1 = coredsl.shift_left %arg0, %0 : ui32, ui5
          %2 = coredsl.xor %arg0, %1 : ui32, ui32
          %3 = coredsl.cast %0 : ui5 to ui8
          %4 = call @ROTR(%2, %3) : (ui32, ui8) -> ui32
          return %4 : ui32
        }
        
        coredsl.instruction @sparkle_ell("0000010", %rs2 : ui5, %rs1 : ui5, "111", %rd : ui5, "1111011"){
          %0 = coredsl.get @X[%rs1 : ui5] : ui32
          %1 = coredsl.get @X[%rs2 : ui5] : ui32
          %2 = coredsl.xor %0, %1 : ui32, ui32
          %3 = func.call @ELL(%2) : (ui32) -> ui32
          coredsl.set @X[%rd : ui5] = %3 : ui32
          coredsl.end
        }
        
        coredsl.instruction @sparkle_rcon("0000", %imm : ui3, %rs2 : ui5, %rs1 : ui5, "110", %rd : ui5, "1111011"){
          %0 = coredsl.get @X[%rs1 : ui5] : ui32
          %1 = coredsl.get @RCON[%imm : ui3] : ui32
          %2 = coredsl.xor %0, %1 : ui32, ui32
          coredsl.set @X[%rd : ui5] = %2 : ui32
          coredsl.end
        }
      }
      ```
    }];
    let cppNamespace = "::mlir::coredsl";
}

//===----------------------------------------------------------------------===//
// Base CoreDSL operation definition.
//===----------------------------------------------------------------------===//

class CoreDSL_Op<string mnemonic, list<Trait> traits = []> :
        Op<CoreDSL_Dialect, mnemonic, traits>;

include "CoreDSLOps.td"

#endif // SHORTNAIL_DIALECT_COREDSL_COREDSL_TD
