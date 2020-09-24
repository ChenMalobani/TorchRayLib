
# Build zlib (only on Windows)
set(ZLIB_BUILD_STATIC_LIBS OFF CACHE BOOL " " FORCE)
set(ZLIB_BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)
add_subdirectory(zlib)

set(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/zlib;${CMAKE_CURRENT_BINARY_DIR}/zlib" CACHE PATH " " FORCE)
set(ZLIB_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/zlib/$<CONFIGURATION>/zlib.lib" CACHE FILEPATH " " FORCE)

set_property(TARGET zlib PROPERTY FOLDER "dependencies")
include_directories(${ZLIB_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/zlib")

# Build libpng 1.6 (on Windows & MacOS)
set(PNG_SHARED ON CACHE BOOL " " FORCE)
set(PNG_STATIC OFF CACHE BOOL " " FORCE)
set(PNG_TESTS OFF CACHE BOOL " " FORCE)
set(PNG_SKIP_INSTALL_ALL TRUE)
add_subdirectory(libpng)
set_property(TARGET png16 PROPERTY FOLDER "dependencies")
add_dependencies(png16 zlib)
set(PNG_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/libpng;${CMAKE_CURRENT_BINARY_DIR}/libpng" PARENT_SCOPE)
set(PNG_LIBRARIES    "$<TARGET_LINKER_FILE:png16>" PARENT_SCOPE)
set(PNG_DEFINES      -DMTS_HAS_LIBPNG PARENT_SCOPE)
# Give libpng & libjpeg a name that's guaranteeed not to match other
# libraries that may already be loaded (e.g. into a Python interpreter)
set_property(TARGET png16 PROPERTY OUTPUT_NAME "png16-torch")


if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64|AMD64")

endif()

# Disable annoying MSVC warnings in rgb2spec build
if (MSVC)
  add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
endif()

