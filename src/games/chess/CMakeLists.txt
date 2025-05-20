cmake_minimum_required(VERSION 3.18)
project(Chess)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CHESS_SOURCES
    geometry.cpp
    move.cpp
    move_generation.cpp
    square.cpp
    state.cpp
    string_transforms.cpp
)
add_library(chess STATIC ${CHESS_SOURCES})
target_include_directories(chess PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(chess PRIVATE 
    $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native>
    $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)
set_target_properties(chess PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
) 