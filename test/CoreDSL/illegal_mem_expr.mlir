// RUN: shortnail-opt %s -split-input-file -verify-diagnostics

coredsl.isax "" {
    coredsl.register local @ACC : ui48

    coredsl.instruction @reset_accum(%acc : ui2, "000000000000000", "001", "00000", "0010111") {
      %idx_1 = hwarith.constant 0 : ui1

      // Illegal register accesses: caught by index width checks
      // expected-error @+1 {{index and ranged accesses are not allowed for this register/address space}}
      %illegal_1 = coredsl.get @ACC[%idx_1 : ui1] : ui48

      coredsl.end
    }
}

// -----

coredsl.isax "" {
    coredsl.register local @ACC : ui48

    coredsl.instruction @reset_accum(%acc : ui2, "000000000000000", "001", "00000", "0010111") {
      %zero = hwarith.constant 0 : ui48
      %idx_1 = hwarith.constant 0 : ui1

      // Illegal register accesses:
      // expected-error @+1 {{index and ranged accesses are not allowed for this register/address space}}
      coredsl.set @ACC[%idx_1 : ui1] = %zero : ui48

      coredsl.end
    }
}

// -----

coredsl.isax "" {
    coredsl.register local @ACCUM[4] : ui48

    coredsl.instruction @reset_accum(%acc : ui2, "000000000000000", "001", "00000", "0010111") {
      // Illegal register field accesses:
      // expected-error @+1 {{missing index variable or range to access this register/address space}}
      %illegal_2 = coredsl.get @ACCUM : ui48

      coredsl.end
    }
}

// -----

coredsl.isax "" {
    coredsl.register core_x @ACCUM[4] : ui48

    coredsl.instruction @reset_accum(%acc : ui2, "000000000000000", "001", "00000", "0010111") {
      %zero = hwarith.constant 0 : ui48
      // Illegal register field accesses:
      // expected-error @+1 {{missing index variable or range to access this register/address space}}
      coredsl.set @ACCUM = %zero : ui48

      coredsl.end
    }
}

// -----

coredsl.isax "" {
    // Too few values to initialize register field:
    // expected-error @+1 {{number of elements in initializer does not match register size}}
    coredsl.register local @ACCUM[4] = 0 : ui48
}

// -----

coredsl.isax "" {
    // To many values to initialize single register:
    // expected-error @+1 {{number of elements in initializer does not match register size}}
    coredsl.register local @ACCUM = [0, 1, 2] : ui48
}

// -----

coredsl.isax "" {
    // Legal init value expressions:
    // expected-error @+1 {{initial value width exceeds register width}}
    coredsl.register core_x @ACC6 = 256 : ui8
}

coredsl.isax "" {
    // Legal init value expressions:
    // expected-error @+1 {{initial value width exceeds register width}}
    coredsl.register core_x @ACC6 = -129 : si8
}

// -----

coredsl.isax "" {
    // Legal init value expressions:
    // expected-error @+1 {{initial value width exceeds register width: -129 (9 bits)}}
    coredsl.register core_x @ACC6[3] = [0, -129, 4] : si8
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{register fields of size 0 are invalid}}
    coredsl.register local @INVALID_REG_FIELD[0] : ui32
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{the default register file must be declared as register field}}
    coredsl.register core_x @T1 : ui32
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{the default register file must be declared as register field}}
    coredsl.register core_x @T2[1] : ui32
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{register type must be an arbitrary precision integer with signedness semantics}}
    coredsl.register local @T1[42] : i32
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{register type must be an arbitrary precision integer with signedness semantics}}
    coredsl.register local @T1 : i32
}

// -----

coredsl.isax "" {
    coredsl.register local const @R = 0 : ui32

    coredsl.instruction @reset_r("00", "000000000000000", "001", "00000", "0010111") {
      %zero = hwarith.constant 0 : ui32

      // trying to write to a const register:
      // expected-error @+1 {{writing to a const 'coredsl.register', 'coredsl.addrspace' or 'coredsl.alias' is prohibited}}
      coredsl.set @R = %zero : ui32

      coredsl.end
    }
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{this protocol expects an address type}}
    coredsl.addrspace axi4mm @AXI : ui32
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{the wire protocol has no address type}}
    coredsl.addrspace wire @IRQ : (ui32) -> ui1
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{the address type must be an arbitrary precision integer with signedness semantics}}
    coredsl.addrspace axi4mm @AXI : (i1) -> ui32
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{the result type must be an arbitrary precision integer with signedness semantics}}
    coredsl.addrspace axi4mm @AXI : (ui32) -> i32
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{the result type is restricted to 8 bits}}
    coredsl.addrspace core_mem @MEM : (ui32) -> ui32
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{the address type is restricted to 12 bits}}
    coredsl.addrspace core_csr @MEM : (ui10) -> ui32
}

// -----

coredsl.isax "" {
    coredsl.addrspace core_mem @MEM : (ui32) -> ui8
    // expected-error @+1 {{'coredsl.get' op attribute 'from' failed to satisfy constraint: index attribute whose value is non-negative}}
    %1 = coredsl.get @MEM[-1:0] : ui8
}

// -----

coredsl.isax "" {
    coredsl.addrspace core_mem @MEM : (ui32) -> ui8
    %0 = hwarith.constant 0 : si1
    // expected-error @+1 {{'coredsl.get' op operand #0 must be an arbitrary precision integer with signedness semantics and unsigned integer, but got 'si1'}}
    %1 = coredsl.get @MEM[%0 : si1] : ui8
}

// -----

coredsl.isax "" {
    coredsl.register local @REGFIELD[9] : ui8
    %0 = coredsl.get @REGFIELD[8] : ui8
    // expected-error @+1 {{the access range exceeds the bounds of the register/address space}}
    %1 = coredsl.get @REGFIELD[9] : ui8
}

// -----

coredsl.isax "" {
    coredsl.register local @REGFIELD[9] : ui8
    coredsl.alias @A1 = @REGFIELD[6:2]
    %0 = coredsl.get @A1[4] : ui8
    // expected-error @+1 {{the access range exceeds the bounds of the register/address space}}
    %1 = coredsl.get @A1[5] : ui8
}

// -----

coredsl.isax "" {
    coredsl.register local @REGFIELD[9] : ui8
    coredsl.alias @A1 = @REGFIELD[8]
    coredsl.alias @A2 = @REGFIELD[8:7]
    // expected-error @+1 {{the specified alias range exceeds the underlying size of the 'coredsl.register' or 'coredsl.addrspace'}}
    coredsl.alias @A3 = @REGFIELD[9]
}

// -----

coredsl.isax "" {
  coredsl.addrspace axi4mm @AXI : (ui8) -> ui42
  // expected-error @+1 {{the end index of the specified alias range exceeds the 'coredsl.register' or 'coredsl.addrspace' index width.}}
  coredsl.alias @A1 = @AXI[256:255]
}

// -----

coredsl.isax "" {
    coredsl.register local const @R = 0 : ui32
    coredsl.alias @A1 = @R
    coredsl.alias const @A2 = @R

    coredsl.instruction @set_const_alias("00", "000000000000000", "001", "00000", "0010111") {
      %zero = hwarith.constant 0 : ui32

      // trying to write to a const register via non const alias:
      coredsl.set @A1 = %zero : ui32
      // trying to write to a const register via const alias:
      // expected-error @+1 {{writing to a const 'coredsl.register', 'coredsl.addrspace' or 'coredsl.alias' is prohibited}}
      coredsl.set @A2 = %zero : ui32

      coredsl.end
    }
}
