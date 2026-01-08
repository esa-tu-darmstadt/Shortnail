add_shortnail_dialect_library(ShortnailCoreDSL
    CoreDSLDialect.cpp
    CoreDSLDirectives.cpp
    CoreDSLOps.cpp

    DEPENDS
    MLIRCoreDSLIncGen
    MLIRCoreDSLEnumsIncGen
    MLIRCoreDSLInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)
