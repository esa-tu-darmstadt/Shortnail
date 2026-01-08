// RUN: shortnail-opt %s -split-input-file -verify-diagnostics | shortnail-opt -verify-diagnostics | FileCheck %s

// CHECK: module {
// CHECK:   coredsl.register core_x   @X[32] : ui32
// CHECK:   coredsl.instruction @spawn("0000000", %[[VAR_0:.*]] : ui5, %[[VAR_1:.*]] : ui5, "000", %[[VAR_2:.*]] : ui5, "0001011"){
// CHECK:     %[[VAR_3:.*]] = coredsl.get @X[%[[VAR_1]] : ui5] : ui32
// CHECK:     %[[VAR_4:.*]] = coredsl.get @X[%[[VAR_0]] : ui5] : ui32
// CHECK:     coredsl.spawn {
// CHECK:       %[[VAR_5:.*]] = hwarith.mul %[[VAR_3]], %[[VAR_4]] : (ui32, ui32) -> ui64
// CHECK:       %[[VAR_6:.*]] = coredsl.cast %[[VAR_5]] : ui64 to ui32
// CHECK:       coredsl.set @X[%[[VAR_2]] : ui5] = %[[VAR_6]] : ui32
// CHECK:       coredsl.end
// CHECK:     }
// CHECK:   }
coredsl.isax "" {
  coredsl.register core_x @X[32] : ui32

  coredsl.instruction @spawn("0000000", %rs2 : ui5, %rs1 : ui5, "000", %rd : ui5, "0001011") {
    %val_rs1 = coredsl.get @X[%rs1 : ui5] : ui32
    %val_rs2 = coredsl.get @X[%rs2 : ui5] : ui32

    coredsl.spawn {
      // Do some time consuming calculations...
      %res = hwarith.mul %val_rs1, %val_rs2 : (ui32, ui32) -> ui64
      %res_trunc = coredsl.cast %res : ui64 to ui32
      coredsl.set @X[%rd : ui5] = %res_trunc : ui32
      coredsl.end
    }
  }
}

// -----

coredsl.isax "" {
  coredsl.instruction @nested_spawn("00000000000000000", "001", "00000", "0010111") {
    coredsl.spawn {
      // expected-error @+1 {{'coredsl.spawn' op expects parent op 'coredsl.instruction'}}
      coredsl.spawn {
        coredsl.end
      }
    }
  }
}

// -----

// CHECK: module {
// CHECK:   coredsl.register core_pc   @PC : ui32
// CHECK:   coredsl.register local   @TOGGLE = 0 : ui1
// CHECK:   coredsl.always @HARDCORE {
// CHECK:     %[[VAR_0:.*]] = coredsl.get @TOGGLE : ui1
// CHECK:     %[[VAR_1:.*]] = hwarith.constant 1 : ui1
// CHECK:     %[[VAR_2:.*]] = coredsl.xor %[[VAR_0]], %[[VAR_1]] : ui1, ui1
// CHECK:     coredsl.set @TOGGLE = %[[VAR_2]] : ui1
// CHECK:     %[[VAR_3:.*]] = coredsl.cast %[[VAR_0]] : ui1 to i1
// CHECK:     scf.if %[[VAR_3]] {
// CHECK:       %[[VAR_4:.*]] = coredsl.get @PC : ui32
// CHECK:       %[[VAR_5:.*]] = hwarith.constant 4 : ui3
// CHECK:       %[[VAR_6:.*]] = hwarith.add %[[VAR_4]], %[[VAR_5]] : (ui32, ui3) -> ui33
// CHECK:       %[[VAR_7:.*]] = coredsl.cast %[[VAR_6]] : ui33 to ui32
// CHECK:       coredsl.set @PC = %[[VAR_7]] : ui32
// CHECK:     }
// CHECK:     coredsl.end
// CHECK:   }
coredsl.isax "" {
  coredsl.register core_pc @PC : ui32
  coredsl.register local @TOGGLE = 0 : ui1

  coredsl.always @HARDCORE {
    %cond = coredsl.get @TOGGLE : ui1
    %1 = hwarith.constant 1 : ui1
    %newToggle = coredsl.xor %cond, %1 : ui1, ui1
    coredsl.set @TOGGLE = %newToggle : ui1
    %cc = coredsl.cast %cond : ui1 to i1
    scf.if %cc {
      %pc = coredsl.get @PC : ui32
      %update = hwarith.constant 4 : ui3
      %tmp = hwarith.add %pc, %update : (ui32, ui3) -> ui33
      %newPc = coredsl.cast %tmp : ui33 to ui32
      coredsl.set @PC = %newPc : ui32
    }
    coredsl.end
  }
}

// -----

coredsl.isax "" {

  coredsl.instruction @alwaysInvalid("00000000000000000", "001", "00000", "0010111") {
    // expected-error @+1 {{'coredsl.always' op expects parent op 'coredsl.isax'}}
    coredsl.always @HARDCORE {
      coredsl.end
    }
    coredsl.end
  }

}
