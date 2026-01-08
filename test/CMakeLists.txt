set(SHORTNAIL_TEST_TIMEOUT 1500) # Set a 25min timeout on individual tests.
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SHORTNAIL_TEST_DEPENDS
        FileCheck count not
        shortnail-opt
        )

add_lit_testsuite(check-shortnail "Running the Shortnail regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SHORTNAIL_TEST_DEPENDS}
        )
set_target_properties(check-shortnail PROPERTIES FOLDER "Tests")

add_lit_testsuites(Shortnail ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SHORTNAIL_TEST_DEPENDS})
