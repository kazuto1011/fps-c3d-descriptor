project(c3d_classification)
cmake_minimum_required(VERSION 2.8)
add_definitions ("-std=c++11")
aux_source_directory(. SRC_LIST)

find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(OpenCV REQUIRED)
include_directories(
    include
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
