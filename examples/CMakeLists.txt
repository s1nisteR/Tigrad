set(CMAKE_CXX_STANDARD 23)

add_executable(basic_usage basic_usage.cpp)

target_link_libraries(basic_usage PRIVATE Tigrad::Tigrad)

set_target_properties(basic_usage PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples_bin"
)

add_executable(neural_net neural_net.cpp)

target_link_libraries(neural_net PRIVATE Tigrad::Tigrad)

set_target_properties(neural_net PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples_bin"
)