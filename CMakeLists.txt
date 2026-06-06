cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(Tigrad VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(Tigrad INTERFACE)

add_library(Tigrad::Tigrad ALIAS Tigrad)

target_compile_features(Tigrad INTERFACE cxx_std_20)

target_include_directories(Tigrad
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Tests
  include(CTest)
  if(BUILD_TESTING)
    # add_subdirectory(tests) # coming soon
  endif()

  # Examples
  option(TIGRAD_BUILD_EXAMPLES "Build the example executables" ON)
  if(TIGRAD_BUILD_EXAMPLES)
    add_subdirectory(examples)
  endif()
endif()
