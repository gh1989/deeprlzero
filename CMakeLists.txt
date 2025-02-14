cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
project(AlphaZero LANGUAGES CXX CUDA)

# Set the C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -g")
set(CMAKE_BUILD_TYPE Release)

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

# Set CUDA paths explicitly
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")

# Get PyTorch path and make sure it's properly formatted
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

# Source files
set(SOURCES
    src/core/game.cpp
    src/core/tictactoe.cpp
    src/core/neural_network.cpp
    src/core/network_manager.cpp
    src/core/mcts.cpp
    src/core/self_play.cpp
    src/core/trainer.cpp
    src/core/evaluator.cpp
)

# Create the main library
add_library(alphazero_lib ${SOURCES})

# Link dependencies
target_link_libraries(alphazero_lib 
    PRIVATE 
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
)

target_include_directories(alphazero_lib 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# Add the main training executable
add_executable(train_alphazero src/main.cpp)
target_link_libraries(train_alphazero 
    PRIVATE 
    alphazero_lib 
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(alphazero_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set output directory
set_target_properties(
    train_alphazero
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

find_package(OpenMP REQUIRED)
target_link_libraries(train_alphazero PRIVATE OpenMP::OpenMP_CXX)

# Enable testing
enable_testing()

# Debug information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PyTorch cmake path: ${CMAKE_PREFIX_PATH}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")

