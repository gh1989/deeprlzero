cmake_minimum_required(VERSION 3.18)
project(AlphaZero LANGUAGES CXX CUDA)

# Set the C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -g")
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# Set Torch directory and find package
set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# CUDA settings
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/core/game.cpp
    src/core/tictactoe.cpp
    src/core/neural_network.cpp
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
    "${TORCH_LIBRARIES}"
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
    "${TORCH_LIBRARIES}"
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
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")

find_package(CUDA)

if(CUDA_FOUND)
    add_definitions(-DUSE_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-arch=sm_75)
endif()

target_link_libraries(alphazero_lib 
    PRIVATE 
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
)

