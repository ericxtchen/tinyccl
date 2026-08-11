cmake_minimum_required(VERSION 3.24)

project(tinyccl LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CUDA_COMPILER clang++)

set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(myapp src/main.cpp)

target_compile_options(myapp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--cuda-gpu-arch=sm_89>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(myapp PRIVATE CUDA::cudart)