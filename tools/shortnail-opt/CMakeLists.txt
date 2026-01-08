add_shortnail_executable(shortnail-opt shortnail-opt.cpp)

llvm_update_compile_flags(shortnail-opt)
target_link_libraries(shortnail-opt
    PRIVATE

    MLIRArithDialect
    MLIRIR
    MLIRMemRefDialect
    MLIROptLib
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRTransforms

    CIRCTComb
    CIRCTExportVerilog
    CIRCTHW
    CIRCTScheduling
    CIRCTSeq
    CIRCTSeqToSV
    CIRCTSeqTransforms
    CIRCTSupport
    CIRCTSV
    CIRCTSVTransforms

    ShortnailCoreDSL
    ShortnailCoreDSLToPy
    ShortnailMergeISAX
)

mlir_check_all_link_libraries(shortnail-opt)
install(TARGETS shortnail-opt RUNTIME DESTINATION bin)
