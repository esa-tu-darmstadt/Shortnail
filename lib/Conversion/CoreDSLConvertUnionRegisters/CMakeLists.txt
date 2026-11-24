add_shortnail_library(ShortnailCoreDSLConvertUnionRegisters
    CoreDSLConvertUnionRegisters.cpp

    DEPENDS
    MLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass

    CIRCTHW

    ShortnailCoreDSL
)
