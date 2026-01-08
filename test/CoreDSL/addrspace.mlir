// RUN: shortnail-opt %s -canonicalize -split-input-file -verify-diagnostics | shortnail-opt | FileCheck %s

// CHECK: module {
// CHECK:   coredsl.addrspace core_mem  @MEM : (ui32) -> ui8
// CHECK:   coredsl.instruction @readWriteMem("00", "000000000000000", "000", "00000", "0000000"){
// CHECK:     %[[_VAR_0:.*]] = hwarith.constant 2 : ui3
// CHECK:     %[[_VAR_1:.*]] = hwarith.constant 1 : ui2
// CHECK:     %[[_VAR_2:.*]] = hwarith.constant 0 : ui32
// CHECK:     %[[_VAR_3:.*]] = coredsl.get @MEM[%[[_VAR_2]] : ui32] : ui8
// CHECK:     coredsl.set @MEM[%[[_VAR_1]] : ui2] = %[[_VAR_3]] : ui8
// CHECK:     %[[_VAR_4:.*]] = coredsl.get @MEM[%[[_VAR_0]] : ui3] : ui8
// CHECK:     coredsl.set @MEM[%[[_VAR_2]] : ui32] = %[[_VAR_4]] : ui8
// CHECK:     coredsl.end
// CHECK:   }
// CHECK: }

coredsl.isax "" {
    coredsl.addrspace core_mem @MEM : (ui32) -> ui8

    coredsl.instruction @readWriteMem("00", "000000000000000", "000", "00000", "0000000") {
      %idx_3 = hwarith.constant 2 : ui3
      %idx_2 = hwarith.constant 1 : ui2
      %zero = hwarith.constant 0 : ui32

      %read1 = coredsl.get @MEM[%zero : ui32] : ui8
      coredsl.set @MEM[%idx_2 : ui2] = %read1 : ui8

      %read2 = coredsl.get @MEM[%idx_3 : ui3] : ui8
      coredsl.set @MEM[%zero : ui32] = %read2 : ui8

      coredsl.end
    }
}

// -----

// CHECK: module {
// CHECK:   coredsl.addrspace core_mem   @MEM : (ui32) -> ui8
// CHECK:   coredsl.instruction @memRangeAccess("00", "000000000000000", "000", "00000", "0000000"){
// CHECK:     %[[VAR_0:.*]] = hwarith.constant 2 : ui3
// CHECK:     %[[VAR_1:.*]] = hwarith.constant 1 : ui2
// CHECK:     %[[VAR_2:.*]] = hwarith.constant 0 : ui32
// CHECK:     %[[VAR_3:.*]] = coredsl.get @MEM[%[[VAR_2]] : ui32, 0:2] : ui24
// CHECK:     coredsl.set @MEM[%[[VAR_1]] : ui2, 2:0] = %[[VAR_3]] : ui24
// CHECK:     %[[VAR_4:.*]] = coredsl.get @MEM[%[[VAR_0]] : ui3, 1:2] : ui16
// CHECK:     coredsl.set @MEM[%[[VAR_2]] : ui32, 2:1] = %[[VAR_4]] : ui16
// CHECK:     coredsl.end
// CHECK:   }
// CHECK: }

coredsl.isax "" {
    coredsl.addrspace core_mem @MEM : (ui32) -> ui8

    coredsl.instruction @memRangeAccess("00", "000000000000000", "000", "00000", "0000000") {
      %idx_3 = hwarith.constant 2 : ui3
      %idx_2 = hwarith.constant 1 : ui2
      %zero = hwarith.constant 0 : ui32

      %read1 = coredsl.get @MEM[%zero : ui32, 0:2] : ui24
      coredsl.set @MEM[%idx_2 : ui2, 2:0] = %read1 : ui24

      %read2 = coredsl.get @MEM[%idx_3 : ui3, 1:2] : ui16
      coredsl.set @MEM[%zero : ui32, 2:1] = %read2 : ui16

      coredsl.end
    }
}

// -----

// CHECK: module {
// CHECK:   coredsl.register local   @REG : ui42
// CHECK:   coredsl.register local   @REGFIELD[42] : ui32
// CHECK:   coredsl.register core_x   @X[32] : ui32
// CHECK:   coredsl.addrspace core_mem   @MEM : (ui32) -> ui8
// CHECK:   coredsl.addrspace axi4mm   @AXI : (ui8) -> ui42
// CHECK:   coredsl.addrspace wire   @WIRE : ui1
// CHECK:   coredsl.alias   @A1 = @REG
// CHECK:   coredsl.alias   @A2 = @REGFIELD[5:0]
// CHECK:   coredsl.alias   @A3 = @REGFIELD[6]
// CHECK:   coredsl.alias   @A4 = @MEM[20000:42]
// CHECK:   coredsl.alias   @A5 = @AXI[200:19]
// CHECK:   coredsl.alias   @A6 = @WIRE
// CHECK:   coredsl.instruction @aliasAccess("00", "000000000000000", "000", "00000", "0000000"){
// CHECK:     %[[VAR_0:.*]] = coredsl.get @A1 : ui42
// CHECK:     %[[VAR_1:.*]] = coredsl.get @A2[5] : ui32
// CHECK:     %[[VAR_2:.*]] = coredsl.get @A3 : ui32
// CHECK:     %[[VAR_3:.*]] = coredsl.get @A4[0] : ui8
// CHECK:     %[[VAR_4:.*]] = coredsl.get @A5[0] : ui42
// CHECK:     %[[VAR_5:.*]] = coredsl.get @A6 : ui1
// CHECK:     coredsl.end

coredsl.isax "" {
  coredsl.register local @REG : ui42
  coredsl.register local @REGFIELD[42] : ui32
  coredsl.register core_x @X[32] : ui32
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8
  coredsl.addrspace axi4mm @AXI : (ui8) -> ui42
  coredsl.addrspace wire @WIRE : ui1

  coredsl.alias @A1 = @REG
  coredsl.alias @A2 = @REGFIELD[5:0]
  coredsl.alias @A3 = @REGFIELD[6]
  coredsl.alias @A4 = @MEM[20000:42]
  coredsl.alias @A5 = @AXI[200:19]
  coredsl.alias @A6 = @WIRE

  coredsl.instruction @aliasAccess("00", "000000000000000", "000", "00000", "0000000") {
    %0 = coredsl.get @A1 : ui42
    %1 = coredsl.get @A2[5] : ui32
    %2 = coredsl.get @A3 : ui32
    %3 = coredsl.get @A4[0] : ui8
    %4 = coredsl.get @A5[0] : ui42
    %5 = coredsl.get @A6 : ui1
    coredsl.end
  }
}
