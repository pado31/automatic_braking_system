cmake_minimum_required(VERSION 2.8)

project(main)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set (CMAKE_CXX_STANDARD 11)

set( project_sources
   src/main.cpp
   src/LaneLines.hpp
   src/LaneLines.cpp
   src/CarDetection.hpp
   src/CarDetection.cpp
)

add_executable(${PROJECT_NAME} ${project_sources})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


