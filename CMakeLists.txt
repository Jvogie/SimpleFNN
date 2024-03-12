cmake_minimum_required(VERSION 3.27)
project(NeuralNet_Implementations)

set(CMAKE_CXX_STANDARD 17)

## add eigen3 library
set(EIGEN_DIR "C:\\Program Files\\eigen")
include_directories(${EIGEN_DIR})

## add_subdirectory(matplotplusplus)

add_executable(NeuralNet_Implementations main.cpp
)
##target_link_libraries(NeuralNet_Implementations PUBLIC matplot)
