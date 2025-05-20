cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
project(AlphaZero LANGUAGES CXX CUDA)

# Set Release as default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Add optimization flags based on compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # GCC and Clang optimizations
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto")
  
  # Additional vectorization flags
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ftree-vectorize")
  
  # Fast math (be careful with this if you need precise floating point)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
elseif(MSVC)
  # MSVC optimizations
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /GL /Gy")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
endif()

# Print the build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")

# Set the C++ standard and base compiler flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build-type specific flags
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -Wpedantic -fno-omit-frame-pointer -pg")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# CUDA flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Function to enable profiling - moved outside the if block so it's always defined
function(enable_profiling target)
    target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:-pg -fno-omit-frame-pointer>)
    target_link_options(${target} PRIVATE $<$<CONFIG:Debug>:-pg>)
endfunction()

# Set CUDA paths explicitly
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")

# Get PyTorch path and format it appropriately.
execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the PyTorch paths explicitly
set(CMAKE_PREFIX_PATH ${TORCH_PATH})
set(Torch_DIR "${TORCH_PATH}/share/cmake/Torch")

find_package(Torch REQUIRED)
message(STATUS "Torch library path: ${TORCH_LIBRARIES}")
message(STATUS "Torch include path: ${TORCH_INCLUDE_DIRS}")

# Add PyTorch libraries to include and link paths
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_PATH}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add chess subdirectory
add_subdirectory(src/games/chess)

# Source files for the main library
set(SOURCES
    src/games/tictactoe.cxx
    src/games/chess.cxx
    src/network.cxx
    src/mcts.cxx
    src/trainer.cxx
)

# Create the main library target
add_library(deeprlzero_lib ${SOURCES})
target_link_libraries(deeprlzero_lib 
    PRIVATE 
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
    chess
)
target_include_directories(deeprlzero_lib 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
)

# Build the main training executable
add_executable(train_deeprlzero src/main.cxx)
target_link_libraries(train_deeprlzero 
    PRIVATE 
    deeprlzero_lib 
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)

add_executable(explorer src/explorer.cxx)
target_link_libraries(explorer 
    PRIVATE 
    deeprlzero_lib 
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Add play executable
add_executable(play_tictactoe src/play_tictactoe.cxx)
target_link_libraries(play_tictactoe 
    PRIVATE 
    deeprlzero_lib 
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(deeprlzero_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(train_deeprlzero PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(deeprlzero_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(train_deeprlzero PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(explorer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(play_tictactoe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

enable_profiling(deeprlzero_lib)
enable_profiling(train_deeprlzero)
enable_profiling(explorer)
enable_profiling(play_tictactoe)