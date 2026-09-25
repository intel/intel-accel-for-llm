cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED IAXL_PROJECT_VERSION)
    message(FATAL_ERROR "IAXL_PROJECT_VERSION must be provided by setup.py")
endif()

project(
    intel-accel-for-llm
    VERSION ${IAXL_PROJECT_VERSION}
    LANGUAGES C CXX)

set(DEVICE
    "cuda"
    CACHE STRING "Inference device: cuda, xpu, or cpu")
set_property(CACHE DEVICE PROPERTY STRINGS cuda xpu cpu)
if(NOT DEVICE MATCHES "^(cuda|xpu|cpu)$")
    message(FATAL_ERROR "Unsupported DEVICE=${DEVICE}; expected cuda, xpu, or cpu")
endif()

option(BUILD_CPU_BIND_SUPPORT "Enable CPU binding support" OFF)
option(BUILD_AVX512_SUPPORT "Enable AVX512 support" OFF)
option(ENABLE_TORCH_PROFILER "Enable Torch profiler integration" ON)
option(IAXL_USE_SYSTEM_QAT "Link installed QATlib and USDM instead of building the QAT SDK" OFF)
if(DEVICE STREQUAL "cuda")
    option(ENABLE_NVTX "Enable NVTX profiling integration" ON)
else()
    option(ENABLE_NVTX "Enable NVTX profiling integration" OFF)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CXX11_ABI)
    message(FATAL_ERROR "CXX11_ABI must be provided by setup.py")
endif()

find_package(
    Python
    COMPONENTS Interpreter Development
    REQUIRED)
find_package(pybind11 REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)"
    OUTPUT_VARIABLE PYTORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTORCH_RESULT)
if(NOT PYTORCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to locate the active PyTorch installation")
endif()

set(IAXL_CSRC_DIR "${CMAKE_SOURCE_DIR}/iaxl/csrc")
set(IAXL_LIB_DIR
    "${CMAKE_SOURCE_DIR}/_lib"
    CACHE PATH "QAT library directory")

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE IAXL_COMMIT_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

file(GLOB TORCH_EXT_SRCS CONFIGURE_DEPENDS "${IAXL_CSRC_DIR}/torch_ext/*.cpp")
file(GLOB KV_POOL_SRCS CONFIGURE_DEPENDS "${IAXL_CSRC_DIR}/kv_pool/*.cpp")
file(GLOB KV_ZIP_SRCS CONFIGURE_DEPENDS "${IAXL_CSRC_DIR}/kv_zip/*.cpp")
file(GLOB QAT_ZIP_SRCS CONFIGURE_DEPENDS "${IAXL_CSRC_DIR}/qat_zip/*.c")
file(GLOB CPU_ZIP_SRCS CONFIGURE_DEPENDS "${IAXL_CSRC_DIR}/cpu_zip/*.c")
file(GLOB IAA_ZIP_SRCS CONFIGURE_DEPENDS "${IAXL_CSRC_DIR}/iaa_zip/*.c")
file(GLOB COMMON_SRCS CONFIGURE_DEPENDS "${IAXL_CSRC_DIR}/common/*.c")
set(IAXL_SRCS ${TORCH_EXT_SRCS} ${KV_POOL_SRCS} ${KV_ZIP_SRCS} ${QAT_ZIP_SRCS} ${CPU_ZIP_SRCS}
              ${IAA_ZIP_SRCS} ${COMMON_SRCS})

if(DEVICE STREQUAL "cpu")
    list(APPEND IAXL_SRCS "${IAXL_CSRC_DIR}/kv_xfer/rdma_stub.cpp")
else()
    list(APPEND IAXL_SRCS "${IAXL_CSRC_DIR}/kv_xfer/rdma.cpp")
endif()

if(DEVICE STREQUAL "cuda")
    list(APPEND IAXL_SRCS "${IAXL_CSRC_DIR}/kv_xfer/cuda.cpp" "${IAXL_CSRC_DIR}/kv_xfer/dsa.cpp")
    file(GLOB DSA_GD_SRCS CONFIGURE_DEPENDS "${IAXL_CSRC_DIR}/dsa_gd/*.c")
    list(APPEND IAXL_SRCS ${DSA_GD_SRCS})
    set_source_files_properties(${DSA_GD_SRCS} PROPERTIES COMPILE_OPTIONS
                                                          "-std=gnu11;-D_GNU_SOURCE")
elseif(DEVICE STREQUAL "xpu")
    list(APPEND IAXL_SRCS "${IAXL_CSRC_DIR}/kv_xfer/xpu.cpp")
else()
    list(APPEND IAXL_SRCS "${IAXL_CSRC_DIR}/kv_xfer/cpu.cpp")
    # Host-only DSA memcpy (no GDRCopy) for the pure-copy segments of the CPU path.
    set(DSA_MEMCPY_SRC "${IAXL_CSRC_DIR}/dsa_gd/dsa_memcpy.c")
    list(APPEND IAXL_SRCS ${DSA_MEMCPY_SRC})
    set_source_files_properties(${DSA_MEMCPY_SRC} PROPERTIES COMPILE_OPTIONS
                                                             "-std=gnu11;-D_GNU_SOURCE")
endif()

if(IAXL_USE_SYSTEM_QAT)
    find_library(IAXL_QAT_LIBRARY NAMES qat REQUIRED)
    find_library(IAXL_USDM_LIBRARY NAMES usdm REQUIRED)
    set(QAT_USERSPACE_LIBS "${IAXL_QAT_LIBRARY}" "${IAXL_USDM_LIBRARY}")
else()
    set(QAT_USERSPACE_LIBS "${IAXL_LIB_DIR}/libqat_s.so" "${IAXL_LIB_DIR}/libusdm_drv_s.so")
    add_custom_command(
        OUTPUT ${QAT_USERSPACE_LIBS}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${IAXL_LIB_DIR}"
        COMMAND make -C "${IAXL_CSRC_DIR}" DEVICE=${DEVICE} qat-userspace
        COMMENT "Building Intel QAT user-space libraries"
        VERBATIM)
endif()
add_custom_target(qat_userspace_libs DEPENDS ${QAT_USERSPACE_LIBS})

set(QPL_PREFIX "${CMAKE_SOURCE_DIR}/_build/qpl/install")
set(QPL_USERSPACE_LIBS "${IAXL_LIB_DIR}/libqpl.so")
add_custom_command(
    OUTPUT ${QPL_USERSPACE_LIBS}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${IAXL_LIB_DIR}"
    COMMAND make -C "${IAXL_CSRC_DIR}" DEVICE=${DEVICE} qpl-userspace
    COMMENT "Building Intel QPL (IAA) user-space library"
    VERBATIM)
add_custom_target(qpl_userspace_libs DEPENDS ${QPL_USERSPACE_LIBS})

pybind11_add_module(torch_ext ${IAXL_SRCS})
add_dependencies(torch_ext qat_userspace_libs qpl_userspace_libs)

target_compile_definitions(
    torch_ext
    PRIVATE TORCH_EXTENSION_NAME=torch_ext _GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}
            IAXL_PROJECT_NAME="${PROJECT_NAME}" IAXL_PROJECT_VERSION="${PROJECT_VERSION}"
            IAXL_COMMIT_ID="${IAXL_COMMIT_ID}" IAXL_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
            IAXL_DEVICE="${DEVICE}")
target_compile_options(
    torch_ext PRIVATE -O3 -fopenmp $<$<BOOL:${BUILD_AVX512_SUPPORT}>:-mavx2;-mavx512f;-mavx512bw>)
target_link_options(torch_ext PRIVATE -fopenmp)
target_include_directories(
    torch_ext
    PRIVATE "${PYTORCH_PATH}/include" "${PYTORCH_PATH}/include/torch/csrc/api/include"
            "${IAXL_CSRC_DIR}/include" "${IAXL_CSRC_DIR}/torch_ext"
            "${IAXL_CSRC_DIR}/qat_zip/include" "${QPL_PREFIX}/include")
target_link_directories(torch_ext PRIVATE "${PYTORCH_PATH}/lib" "${IAXL_LIB_DIR}")
find_package(ZLIB REQUIRED)
target_link_libraries(torch_ext PRIVATE torch torch_python ${QAT_USERSPACE_LIBS} qpl ZLIB::ZLIB dl)

if(NOT DEVICE STREQUAL "cpu")
    find_path(NIXL_INCLUDE_DIR nixl.h HINTS $ENV{NIXL_INCLUDE_DIR} PATHS /usr/local/include
                                                                   NO_DEFAULT_PATH)
    find_library(NIXL_LIBRARY nixl HINTS $ENV{NIXL_LIB_DIR} PATHS /usr/local/lib/x86_64-linux-gnu
                                                                  NO_DEFAULT_PATH)
    if(NOT NIXL_INCLUDE_DIR OR NOT NIXL_LIBRARY)
        message(FATAL_ERROR "NIXL not found; set NIXL_INCLUDE_DIR / NIXL_LIB_DIR")
    endif()
    get_filename_component(NIXL_LIB_DIR "${NIXL_LIBRARY}" DIRECTORY)
    message(STATUS "NIXL: ${NIXL_INCLUDE_DIR} ${NIXL_LIBRARY}")
    target_include_directories(torch_ext PRIVATE "${NIXL_INCLUDE_DIR}")
    target_link_libraries(torch_ext PRIVATE "${NIXL_LIBRARY}")
endif()

if(DEVICE STREQUAL "cuda")
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(torch_ext PRIVATE CUDA_SUPPORT=1 DSA_SUPPORT=1)
    target_include_directories(torch_ext PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(torch_ext PRIVATE ${CUDAToolkit_LIBRARIES} gdrapi dl)

    if(ENABLE_NVTX)
        file(GLOB _nvtx_pip_hints "/usr/local/lib/python*/dist-packages/nvidia/*/include"
             "/usr/lib/python*/dist-packages/nvidia/*/include")
        find_path(NVTX_INCLUDE_DIR "nvtx3/nvToolsExt.h"
                  HINTS ${CUDAToolkit_INCLUDE_DIRS} $ENV{NVTX_INCLUDE_DIR} ${_nvtx_pip_hints})
        if(NVTX_INCLUDE_DIR)
            target_compile_definitions(torch_ext PRIVATE ENABLE_NVTX=1)
            target_include_directories(torch_ext PRIVATE "${NVTX_INCLUDE_DIR}")
            message(STATUS "NVTX enabled from ${NVTX_INCLUDE_DIR}")
        else()
            message(WARNING "nvtx3/nvToolsExt.h not found; NVTX disabled")
        endif()
    endif()
elseif(DEVICE STREQUAL "xpu")
    target_compile_definitions(torch_ext PRIVATE XPU_SUPPORT=1)
else()
    target_compile_definitions(torch_ext PRIVATE CPU_SUPPORT=1 DSA_MEMCPY_SUPPORT=1)
endif()

if(ENABLE_TORCH_PROFILER)
    target_compile_definitions(torch_ext PRIVATE ENABLE_TORCH_PROFILER=1)
endif()

if(BUILD_CPU_BIND_SUPPORT)
    target_compile_definitions(torch_ext PRIVATE CPU_BIND_SUPPORT=1)
    target_link_libraries(torch_ext PRIVATE numa)
endif()

find_package(SQLite3 QUIET)
if(TARGET SQLite3::SQLite3)
    target_link_libraries(torch_ext PRIVATE SQLite3::SQLite3)
elseif(TARGET SQLite::SQLite3)
    target_link_libraries(torch_ext PRIVATE SQLite::SQLite3)
else()
    target_link_libraries(torch_ext PRIVATE sqlite3)
endif()

set(IAXL_BUILD_RPATH "${IAXL_LIB_DIR}")
set(IAXL_INSTALL_RPATH "${IAXL_LIB_DIR}:$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib")
if(NIXL_LIB_DIR)
    string(APPEND IAXL_BUILD_RPATH ":${NIXL_LIB_DIR}")
    string(APPEND IAXL_INSTALL_RPATH ":${NIXL_LIB_DIR}")
endif()
set_target_properties(
    torch_ext
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/iaxl"
               BUILD_RPATH "${IAXL_BUILD_RPATH}"
               INSTALL_RPATH "${IAXL_INSTALL_RPATH}")
