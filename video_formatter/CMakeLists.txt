project(video_formatter_for_c3d)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)

find_package(Boost REQUIRED COMPONENTS system filesystem thread)
include_directories(
    include
    ${Boost_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
