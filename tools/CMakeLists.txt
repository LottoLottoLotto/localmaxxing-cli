cmake_minimum_required(VERSION 3.16)
project(lmx_tools CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# llama.cpp revision the HellaSwag scorer is built against. Override with
# -DLMX_LLAMA_REF=<tag-or-commit>. Pinned to a known-good API for the scorer.
set(LMX_LLAMA_REF "8ad8aef447703e76cdcf74ed78dab16b92ec231a"
    CACHE STRING "llama.cpp git ref to build against")

# CPU-only, statically linked, portable scorer: no GPU toolchains in CI, no
# .so files to ship, and no -march=native so the artifact runs on any CPU of
# the target arch. The scorer is not throughput-critical (and the CLI defaults
# it to CPU when a llama.cpp server already holds the model in VRAM).
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)
set(GGML_CUDA            OFF CACHE BOOL "" FORCE)
set(GGML_METAL           OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL          OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN          OFF CACHE BOOL "" FORCE)
set(GGML_BLAS            OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG        ${LMX_LLAMA_REF}
)
# Local builds can skip the clone with
#   -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=/path/to/llama.cpp
FetchContent_MakeAvailable(llama_cpp)

add_executable(lmx-llama-score-hellaswag llama-score-hellaswag.cpp)
target_link_libraries(lmx-llama-score-hellaswag PRIVATE llama)
set_target_properties(lmx-llama-score-hellaswag PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
