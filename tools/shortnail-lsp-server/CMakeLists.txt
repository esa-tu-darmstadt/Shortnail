get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(shortnail_dialect_libs GLOBAL PROPERTY SHORTNAIL_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  ${mlir_dialect_libs}
  ${circt_dialect_libs}
  ${shortnail_dialect_libs}
  ${conversion_libs}
  MLIRAnalysis
  MLIRDialect
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  )

add_shortnail_tool(shortnail-lsp-server
  shortnail-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(shortnail-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(shortnail-lsp-server)

mlir_check_all_link_libraries(shortnail-lsp-server)
