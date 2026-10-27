// RUN: shortnail-opt %s -canonicalize -split-input-file -verify-diagnostics | shortnail-opt | FileCheck %s

// CHECK: module {
// CHECK:   coredsl.isax "Example" {
// CHECK:     coredsl.register local @TEST_REG  = 10 : ui32
// CHECK:     coredsl.register local @TEST_REGBANK[64]  : ui32
// CHECK:     coredsl.alias @REGBANK_REF = @TEST_REGBANK[63:32]
// CHECK:     coredsl.alias @REG_REF = @TEST_REG
// CHECK:     coredsl.alias @REG_REF_REF = @REG_REF
// CHECK:     coredsl.addrspace core_mem @MEM : (ui32) -> ui8
// CHECK:     coredsl.alias @MEM_REF = @MEM[1023:512]
// CHECK:     coredsl.alias @MEM_REF_REF = @MEM_REF[512:256]
// CHECK:     coredsl.instruction @test("00", "000000000000000", "000", "00000", "0000000"){
// CHECK:       %0 = coredsl.get @MEM_REF_REF[27] : ui8
// CHECK:       %1 = coredsl.cast %0 : ui8 to ui32
// CHECK:       coredsl.set @REGBANK_REF[7] = %1 : ui32
// CHECK:       coredsl.end
// CHECK:     }
// CHECK:   }
// CHECK: }

coredsl.isax "Example" {
  coredsl.register local @TEST_REG = 10 : ui32
  coredsl.register local @TEST_REGBANK[64] : ui32
  coredsl.alias @REGBANK_REF = @TEST_REGBANK[63:32]
  coredsl.alias @REG_REF = @TEST_REG
  coredsl.alias @REG_REF_REF = @REG_REF

  coredsl.addrspace core_mem @MEM : (ui32) -> ui8
  coredsl.alias @MEM_REF = @MEM[1023:512]
  coredsl.alias @MEM_REF_REF = @MEM_REF[512:256]
  coredsl.instruction @test("00", "000000000000000", "000", "00000", "0000000") {
    %0 = coredsl.get @MEM_REF_REF[27] : ui8
    %1 = coredsl.cast %0 : ui8 to ui32
    coredsl.set @REGBANK_REF[7] = %1 : ui32
    coredsl.end
  }
}

// -----

coredsl.isax "Example" {
  coredsl.register local @TEST_REG = 10 : ui32
  coredsl.register local @TEST_REGBANK[64] : ui32
  coredsl.alias @REGBANK_REF = @TEST_REGBANK[63:32]
  coredsl.alias @REG_REF = @TEST_REG
  coredsl.alias @REG_REF_REF = @REG_REF
  // expected-error @+1 {{the specified alias range exceeds the underlying size of the 'coredsl.register' or 'coredsl.addrspace'}}
  coredsl.alias @REG_BIT_REF = @REGBANK_REF[32:0]
}

// -----

coredsl.isax "Example" {
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8
  coredsl.alias @MEM_REF = @MEM[4294967294:0]
  // expected-error @+1 {{the end index of the specified alias range exceeds the 'coredsl.register' or 'coredsl.addrspace' index width.}}
  coredsl.alias @MEM_REF_REF = @MEM_REF[5000000000:0]
}
