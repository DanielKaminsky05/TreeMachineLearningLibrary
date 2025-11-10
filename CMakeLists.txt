cmake_minimum_required(VERSION 3.5)
project(Assignment2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(3rdparty/eigen-5.0.0)

add_executable(assignment2
    main.cpp
    code/MLSuite/Dataset.cpp
    code/MLSuite/LinRegModel.cpp
)
