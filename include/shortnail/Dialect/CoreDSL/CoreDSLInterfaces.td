//===- CoreDSLInterfaces.td - CoreDSL dialect interfaces ---*- tablegen -*-===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef COREDSL_DIALECT_COREDSL_COREDSLINTERFACES_TD
#define COREDSL_DIALECT_COREDSL_COREDSLINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CoreDSL interface definitions.
//===----------------------------------------------------------------------===//

def GetSetOpInterface : OpInterface<"GetSetOpInterface"> {
  let description =
      [{"Common interface for operations that can be accessed with `coredsl.get`
         and `coredsl.set`."}];

  let methods = [
    InterfaceMethod<
        "Returns whether the memory may only be read.", "bool",
        "isConst",
        (ins), /*methodBody=*/ [{
            return $_op.getIsConst();
        }]>,
    InterfaceMethod<
        "Returns whether the memory is considered volatile.", "bool",
        "isVolatile",
        (ins), /*methodBody=*/ [{
            return $_op.getIsVolatile();
        }]>,

    InterfaceMethod<
        "Get the memory size. A value of 0 indicates that the size is unknown.",
        "uint64_t", "getMemSize">,
    InterfaceMethod<
        "Get the maximal allowed index width. A value of 0 indicates that no"
        "index may be used!", "unsigned", "getMaxIndexWidth">,
    InterfaceMethod<
        "Get the minimal required index width.", "unsigned", "getMinIndexWidth">,
    InterfaceMethod<
        "Get the underlying element type of the memory.", "::mlir::IntegerType",
        "getElementType">
  ];
}

#endif // COREDSL_DIALECT_COREDSL_COREDSLINTERFACES_TD
