add_shortnail_library(ShortnailAnalyzeISAX
    AnalyzeISAX.cpp

    DEPENDS
    MLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRSCFUtils
    MLIRTransforms
    LLVMSupport

    ShortnailCoreDSL
    CIRCTHWArith
)
