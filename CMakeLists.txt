cmake_minimum_required(VERSION 3.17)
project(sandbox)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
        BTAS
        GIT_REPOSITORY      https://github.com/ValeevGroup/btas.git
)
FetchContent_MakeAvailable(BTAS)
FetchContent_GetProperties(BTAS
        SOURCE_DIR BTAS_SOURCE_DIR
        BINARY_DIR BTAS_BINARY_DIR
        )
#add_library(BTAS)
message(STATUS "BTAS: ${BTAS}")

add_executable(sandbox
        sandbox.cpp)

find_package(Libint2 REQUIRED)

target_link_libraries(sandbox BTAS)
target_link_libraries(sandbox Libint2::libint2_cxx)
