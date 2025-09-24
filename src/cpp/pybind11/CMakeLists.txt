# cmake -S . -B build/pybind11
# cmake -S . -B build/pybind11 -DPython3_EXECUTABLE=$(which python)
# cmake --build ./build/pybind11 --config Release --target pywebifc -j 32

# docker run --rm -e CMAKE_BUILD_PARALLEL_LEVEL=4 -v "$PWD/..":/repo quay.io/pypa/manylinux_2_28_x86_64 bash -lc 'set -eux; PYBIN=/opt/python/cp312-cp312/bin; "$PYBIN/python" -m pip install -U pip setuptools wheel cmake ninja pybind11 auditwheel; "$PYBIN/python" -m pip wheel /repo/pybind11 -w /repo/pybind11/wheels; auditwheel repair /repo/pybind11/wheels/*pywebifc*.whl -w /repo/pybind11/dist'

cmake_minimum_required(VERSION 3.18)
project(pywebifc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# On Linux, ensure static libs are built with -fPIC so they can link
# into the Python shared module without relocation errors.
if(UNIX AND NOT APPLE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# 调试取消注释下面两行
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")

# Bring in the core web-ifc library from the parent directory
add_subdirectory(.. core_build)

# Make sure the core static library is position independent when linking
# into the Python extension (shared module).
if(TARGET web-ifc-library)
  set_property(TARGET web-ifc-library PROPERTY POSITION_INDEPENDENT_CODE ON)
  if(UNIX AND NOT APPLE)
    target_compile_options(web-ifc-library PRIVATE -fPIC)
  endif()
endif()

# Prefer a locally installed pybind11 (e.g., via pip in your mamba/conda env),
# and fall back to FetchContent if not found.

include(CMakeDependentOption)

# Ensure CMake knows about Python and has Development.Module helpers like
# python3_add_library that pybind11's new tools may use.
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import sysconfig;print(sysconfig.get_paths().get('purelib',''))"
      OUTPUT_VARIABLE PY_PURELIB
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PY_PURELIB)
      set(PYBIND11_CMAKE_DIR "${PY_PURELIB}/pybind11/share/cmake/pybind11")
      if(EXISTS "${PYBIND11_CMAKE_DIR}/pybind11Config.cmake")
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
      endif()
    endif()
  endif()
  # Try again after possibly extending CMAKE_PREFIX_PATH
  find_package(pybind11 CONFIG QUIET)
endif()

if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.13.1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(pywebifc MODULE
  web-ifc-pybind11.cpp
)

target_link_libraries(pywebifc PRIVATE web-ifc-library)
target_compile_features(pywebifc PRIVATE cxx_std_20)

# Avoid libstdc++ version mismatch on Linux by statically linking
# libstdc++ and libgcc into the extension module. This reduces
# runtime dependency on the host's libstdc++.
if(UNIX AND NOT APPLE)
  include(CheckCXXCompilerFlag)
  # Only add if supported by compiler
  check_cxx_compiler_flag("-static-libstdc++" HAS_STATIC_LIBSTDCXX)
  if(HAS_STATIC_LIBSTDCXX)
    target_link_options(pywebifc PRIVATE -static-libstdc++ -static-libgcc)
  endif()
endif()

# --- CTest integration: minimal smoke + import tests ---
include(CTest)
if(BUILD_TESTING)
  enable_testing()

  # Simple smoke test to satisfy Test Explorer discovery
  add_test(NAME pywebifc_smoke
           COMMAND ${CMAKE_COMMAND} -E echo "ok")
  set_tests_properties(pywebifc_smoke PROPERTIES LABELS "pywebifc")

  # Import test: ensure the built module can be imported with PYTHONPATH
  add_test(NAME pywebifc_import
           COMMAND ${Python3_EXECUTABLE} -c
                   "import os,sys; sys.path.insert(0, os.path.abspath('.')); import pywebifc as m; print(m.get_version())")
  # Make sure Python can locate the extension in the build directory
  set_tests_properties(pywebifc_import PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
    LABELS "pywebifc")
endif()
