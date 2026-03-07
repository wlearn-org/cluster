cmake_minimum_required(VERSION 3.16)
project(cluster C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CORE_SOURCES csrc/cluster.c csrc/wl_api.c)

# Shared library
add_library(cluster SHARED ${CORE_SOURCES})
target_include_directories(cluster PUBLIC csrc)
target_compile_definitions(cluster PRIVATE _POSIX_C_SOURCE=200809L)
target_compile_options(cluster PRIVATE -Wall -Wextra -Wpedantic -O2)
target_link_libraries(cluster m)
set_target_properties(cluster PROPERTIES OUTPUT_NAME cluster)

# Test executable
if(BUILD_TESTING)
    add_executable(test_cluster test/test_c.c)
    target_link_libraries(test_cluster cluster)
    target_include_directories(test_cluster PRIVATE csrc)
    target_link_libraries(test_cluster m)
endif()
