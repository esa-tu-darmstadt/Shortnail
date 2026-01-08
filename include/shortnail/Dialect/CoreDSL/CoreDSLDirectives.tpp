//===- CoreDSLDirectives.tpp - CoreDSL custom directive implementations ---===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#include "shortnail/Dialect/CoreDSL/CoreDSLDirectives.h"

namespace mlir::coredsl {

//===----------------------------------------------------------------------===//
// custom<FancyFunctionalType>
//===----------------------------------------------------------------------===//

template <std::size_t... N, class... Types>
static bool parseInputs(OpAsmParser &parser, std::index_sequence<N...>,
                        Types &...inputs) {
  constexpr unsigned maxInputIdx = sizeof...(inputs) - 1;

  // use a fold expression to parse a comma separated type list and return the
  // parsing result. The ternary operator is used to ensure that no comma will
  // be parsed for the last input argument
  return ((parser.parseType(inputs) ||
           (N < maxInputIdx ? parser.parseComma() : ParseResult(success()))) ||
          ...);
}

template <class Result, class... Types>
ParseResult parseFancyFunctionalType(OpAsmParser &parser, Result &result,
                                     Types &...inputs) {

  // parse the argument list of fixed size: (arg1, ..., argN) -> result
  if (parser.parseLParen() ||
      parseInputs(parser, std::index_sequence_for<Types &...>{}, inputs...) ||
      parser.parseRParen() || parser.parseArrow() || parser.parseType(result)) {
    return failure();
  }

  return success();
}

template <bool isNotLast, class InTy>
static void printSingleInput(OpAsmPrinter &p, InTy input) {
  p.printType(input);

  // Only print a comma if this input is not the last one of the argument list
  if constexpr (isNotLast) {
    p << ", ";
  }
}

template <std::size_t... N, class... Types>
static void printInputs(OpAsmPrinter &p, std::index_sequence<N...>,
                        Types... inputs) {
  constexpr unsigned maxInputIdx = sizeof...(inputs) - 1;

  // use a fold expression to print the input argument types as a comma
  // separated list
  (printSingleInput<(N < maxInputIdx)>(p, inputs), ...);
}

template <class OP, class Result, class... Types>
void printFancyFunctionalType(OpAsmPrinter &p, OP &op, Result result,
                              Types... inputs) {
  p << '(';

  printInputs(p, std::index_sequence_for<Types...>{}, inputs...);

  p << ") -> ";
  p.printType(result);
}

} // namespace mlir::coredsl
