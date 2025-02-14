cmake_minimum_required(VERSION 3.18)
project(CUDATest LANGUAGES CXX CUDA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 23)
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "PyTorch cmake path: ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
message(STATUS "Torch_DIR: ${Torch_DIR}")
add_executable(cuda_test cuda_test.cpp)
target_link_libraries(cuda_test "${TORCH_LIBRARIES}")