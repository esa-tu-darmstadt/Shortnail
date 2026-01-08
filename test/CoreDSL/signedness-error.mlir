// RUN: shortnail-opt %s -split-input-file -verify-diagnostics

coredsl.isax "" {
  %0 = hwarith.constant 1 : ui32
  // expected-error @+1 {{bit index 'to' exceeds input value width}}
  %1 = coredsl.bitextract %0[31:32] : (ui32) -> ui2
}

// -----

coredsl.isax "" {
  %0 = hwarith.constant 1 : si32
  // expected-error @+1 {{the result type for a read access must be unsigned and of the same size as the index range}}
  %1 = coredsl.bitextract %0[1:0] : (si32) -> si2
}

// -----

coredsl.isax "" {
  %0 = hwarith.constant 1 : ui32
  // expected-error @+1 {{the result type for a read access must be unsigned and of the same size as the index range}}
  %1 = coredsl.bitextract %0[1:0] : (ui32) -> ui3
}

// -----

coredsl.isax "" {
  %0 = hwarith.constant 1 : ui32
  %1 = hwarith.constant 1 : si2
  // expected-error @+1 {{the result type for a write access must be the same as the value type}}
  %2 = coredsl.bitset %0[2:1] = %1 : (ui32, si2) -> ui2
}

// -----

coredsl.isax "" {
  %0 = hwarith.constant 1 : ui32
  // expected-error @+1 {{casting to a signless integer is only allowed from SI1/UI1 to I1}}
  %1 = coredsl.cast %0 : ui32 to i32
}

// -----

coredsl.isax "" {
  %0 = hwarith.constant 3298983243 : ui32
  %idx = hwarith.constant 1 : ui5
  // expected-error @+1 {{base address width exceeds the max required index width of 4}}
  %ext = coredsl.bitextract %0[%idx : ui5, 23:0] : (ui32) -> ui24
}
