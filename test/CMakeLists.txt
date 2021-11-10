configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  MAIN_CONFIG
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
  )

set(PGEX_TEST_DEPENDS
  FileCheck count not
  runner 
  )

add_lit_testsuite(check-pgex "Running the PGEX regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${PGEX_TEST_DEPENDS}
  )
set_target_properties(check-pgex PROPERTIES FOLDER "Tests")

add_lit_testsuites(PGEX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${PGEX_TEST_DEPS}
)
