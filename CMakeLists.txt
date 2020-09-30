cmake_minimum_required(VERSION 3.11) # FetchContent is available in 3.11+
#project(music_tiles)

project(TorchRayLib
        DESCRIPTION
        "TorchRayLib"
        LANGUAGES
        CXX C
        )

include(FetchContent)

# Uncomment to run on a CPU, make sure you remove _deps if you uncomment
#set(CUDA_V "cpu")
#set(LIBTORCH_DEVICE "cpu")

# Uncomment to run on a GPU make sure you remove _deps if you uncomment
set(CUDA_V "10.2")
set(LIBTORCH_DEVICE "cu102")

# shlomo
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin64)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo
MinSizeRel."
            FORCE )
endif()

set(CMAKE_BUILD_TYPE Release)
set(CXX_RELEASE_FLAGS = -O3 -march=native )
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "*** Have SIZEOF void * = 8, so 64-bit")
    set( IS_64_BIT 1 )
else ()
    message(STATUS "*** SIZEOF void * != 8, so not 64-bit")
    message(FATAL_ERROR "Only 64-bit i supported")
endif ()
if (MSVC)
    # Disable annoying MSVC warnings (all targets)
    add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
    # Parallel build on MSVC (all targets)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Set this to the minimal version you want to support
find_package(raylib 3.0 QUIET) # Let CMake search for a raylib-config.cmake

# You could change the QUIET above to REQUIRED and remove this if() clause
# This part downloads raylib and builds it if it's not installed on your system
if (NOT raylib_FOUND) # If there's none, fetch and build raylib
    include(FetchContent)
    FetchContent_Declare(
        raylib
        URL https://github.com/raysan5/raylib/archive/3.0.0.tar.gz
    )
    FetchContent_GetProperties(raylib)
    if (NOT raylib_POPULATED) # Have we downloaded raylib yet?
        set(FETCHCONTENT_QUIET NO)
        FetchContent_Populate(raylib)
        set(BUILD_GAMES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
        # build raylib
        add_subdirectory(${raylib_SOURCE_DIR} ${raylib_BINARY_DIR})
    endif()
endif()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/ext")
set(PYTORCH_VERSION "1.6.0")
find_package(Torch ${PYTORCH_VERSION} EXACT PATHS "${CMAKE_SOURCE_DIR}/_deps/libtorch")
if(NOT Torch_FOUND)
    unset(Torch_FOUND)
    include(download_libtorch)
endif()
if(NOT Torch_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmake_find")
    find_package(Torch REQUIRED PATHS "${CMAKE_SOURCE_DIR}/../../../libtorch")
endif()

if(${LIBTORCH_DEVICE} STREQUAL "cu102")
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        # There is some bug in CMake (FindCUDA.cmake) that causes it to mess up the -std=c++11 option
        # if you let it propagate it to nvcc in some cases.
        if(NOT MSVC)
            set(CUDA_PROPAGATE_HOST_FLAGS OFF)
            list(APPEND CUDA_NVCC_FLAGS -std=c++11)
        endif()
    endif()
endif()

include(copy_torch_dlls)
################# EXAMPLE 001 ########################
# TARGET
set(EXAMPLE_001_EXE torch_ray_sanity)
#add_executable(${EXAMPLE_001_EXE} src/${EXAMPLE_001_EXE}.cpp include/utils/vision_utils.hpp)
add_executable(${EXAMPLE_001_EXE} src/${EXAMPLE_001_EXE}.cpp)

set(raylib_VERBOSE 1)
target_link_libraries(${EXAMPLE_001_EXE} raylib  ${TORCH_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} m)
target_include_directories(${EXAMPLE_001_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_dpes/libtorch/include/")
target_include_directories(${EXAMPLE_001_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_deps/libtorch/include/torch/csrc/api/")
set_target_properties(${EXAMPLE_001_EXE} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
copy_torch_dlls(${EXAMPLE_001_EXE})


################# EXAMPLE 000 ########################
# TARGET
set(EXAMPLE_000_EXE torch_empty_template)
#add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)
add_executable(${EXAMPLE_000_EXE} src/${EXAMPLE_000_EXE}.cpp include/utils/vision_utils.hpp)
set(raylib_VERBOSE 1)
target_link_libraries(${EXAMPLE_000_EXE} raylib  ${TORCH_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} m)
target_include_directories(${EXAMPLE_000_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_dpes/libtorch/include/")
target_include_directories(${EXAMPLE_000_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_deps/libtorch/include/torch/csrc/api/")
set_target_properties(${EXAMPLE_000_EXE} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
copy_torch_dlls(${EXAMPLE_000_EXE})
################# EXAMPLE 004 ########################


################# EXAMPLE 001 ########################
#set(EXAMPLE_002_EXE torch_music_app)
## This is the main part:
#add_executable(${EXAMPLE_002_EXE} main.cpp list.c sound.c)
#set(raylib_VERBOSE 1)
#target_link_libraries(${EXAMPLE_002_EXE} raylib  ${TORCH_LIBRARIES})
##target_link_libraries(${PROJECT_NAME} m)
#target_include_directories(${EXAMPLE_002_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_dpes/libtorch/include/")
#target_include_directories(${EXAMPLE_002_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_deps/libtorch/include/torch/csrc/api/")
#set_target_properties(${EXAMPLE_002_EXE} PROPERTIES
#        CXX_STANDARD 17
#        CXX_STANDARD_REQUIRED YES)
#
## Copy audio directory to build directory
#file(COPY ${CMAKE_SOURCE_DIR}/audio DESTINATION ${CMAKE_BINARY_DIR})
#copy_torch_dlls(${EXAMPLE_002_EXE})

################# EXAMPLE 003 ########################
# TARGET
set(EXAMPLE_003_EXE torch_core_basic_window)
add_executable(${EXAMPLE_003_EXE} src/${EXAMPLE_003_EXE}.cpp)
set(raylib_VERBOSE 1)
target_link_libraries(${EXAMPLE_003_EXE} raylib  ${TORCH_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} m)
target_include_directories(${EXAMPLE_003_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_dpes/libtorch/include/")
target_include_directories(${EXAMPLE_003_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_deps/libtorch/include/torch/csrc/api/")
set_target_properties(${EXAMPLE_003_EXE} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
copy_torch_dlls(${EXAMPLE_003_EXE})
################# EXAMPLE 003 ########################

################# EXAMPLE 004 ########################
# TARGET
set(EXAMPLE_004_EXE torch_core_random_values)
add_executable(${EXAMPLE_004_EXE} src/${EXAMPLE_004_EXE}.cpp)
set(raylib_VERBOSE 1)
target_link_libraries(${EXAMPLE_004_EXE} raylib  ${TORCH_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} m)
target_include_directories(${EXAMPLE_004_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_dpes/libtorch/include/")
target_include_directories(${EXAMPLE_004_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_deps/libtorch/include/torch/csrc/api/")
set_target_properties(${EXAMPLE_004_EXE} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
copy_torch_dlls(${EXAMPLE_004_EXE})
################# EXAMPLE 004 ########################


################# EXAMPLE 005 ########################
# TARGET
set(EXAMPLE_005_EXE torch_textures_image_processing)
#add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)
add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp include/utils/vision_utils.hpp)
set(raylib_VERBOSE 1)
target_link_libraries(${EXAMPLE_005_EXE} raylib  ${TORCH_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} m)
target_include_directories(${EXAMPLE_005_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_dpes/libtorch/include/")
target_include_directories(${EXAMPLE_005_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_deps/libtorch/include/torch/csrc/api/")
set_target_properties(${EXAMPLE_005_EXE} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
copy_torch_dlls(${EXAMPLE_005_EXE})
################# EXAMPLE 005 ########################


################# EXAMPLE 006 ########################
# TARGET
set(EXAMPLE_006_EXE torch_sigmoid)
#add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)
add_executable(${EXAMPLE_006_EXE} src/${EXAMPLE_006_EXE}.cpp include/utils/vision_utils.hpp)
set(raylib_VERBOSE 1)
target_link_libraries(${EXAMPLE_006_EXE} raylib  ${TORCH_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} m)
target_include_directories(${EXAMPLE_006_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_dpes/libtorch/include/")
target_include_directories(${EXAMPLE_006_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_deps/libtorch/include/torch/csrc/api/")
set_target_properties(${EXAMPLE_006_EXE} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
copy_torch_dlls(${EXAMPLE_006_EXE})
################# EXAMPLE 006 ########################


################# EXAMPLE 006 ########################
# TARGET
set(EXAMPLE_007_EXE torch_super_resolution)
#add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)
add_executable(${EXAMPLE_007_EXE} src/${EXAMPLE_007_EXE}.cpp include/utils/vision_utils.hpp)
set(raylib_VERBOSE 1)
target_link_libraries(${EXAMPLE_007_EXE} raylib  ${TORCH_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} m)
target_include_directories(${EXAMPLE_007_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_dpes/libtorch/include/")
target_include_directories(${EXAMPLE_007_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_deps/libtorch/include/torch/csrc/api/")
set_target_properties(${EXAMPLE_007_EXE} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
copy_torch_dlls(${EXAMPLE_007_EXE})
################# EXAMPLE 006 ########################


################# EXAMPLE 006 ########################
# TARGET
set(EXAMPLE_008_EXE torch_super_resolution_gui)
#add_executable(${EXAMPLE_005_EXE} src/${EXAMPLE_005_EXE}.cpp)
add_executable(${EXAMPLE_008_EXE} src/${EXAMPLE_008_EXE}.cpp include/utils/vision_utils.hpp include/raygui/raygui.h
        include/raygui/ricons.h)
set(raylib_VERBOSE 1)
target_link_libraries(${EXAMPLE_008_EXE} raylib  ${TORCH_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} m)
target_include_directories(${EXAMPLE_008_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_dpes/libtorch/include/")
target_include_directories(${EXAMPLE_008_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_deps/libtorch/include/torch/csrc/api/")
set_target_properties(${EXAMPLE_008_EXE} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
copy_torch_dlls(${EXAMPLE_008_EXE})
################# EXAMPLE 006 ########################


# # Copy trained PTH files
file(COPY "resources/mosaic_cpp.pt" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/candy_cpp.pt" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/udnie_cpp.pt" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/erfnet_fs.pt" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/RRDB_ESRGAN_x4_000.pt" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)


file(COPY "resources/amber.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/windmill.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/parrots.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/cat.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/comic.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/baboon.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/img_004_SRF_2_LR.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/img_003_SRF_2_LR.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/img_005_SRF_2_LR.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/img_002_SRF_2_LR.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/logo.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)

file(COPY "resources/GameCube.ttf" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/GoudyThirty-Light.ttf" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/GLC.otf" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
