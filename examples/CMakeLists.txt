set(CMAKE_CXX_STANDARD 20)

add_executable(basic_usage basic_usage.cpp)

target_link_libraries(basic_usage PRIVATE Tigrad::Tigrad)

set_target_properties(basic_usage PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples_bin"
)

add_executable(single_neuron single_neuron.cpp)

target_link_libraries(single_neuron PRIVATE Tigrad::Tigrad)

set_target_properties(single_neuron PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples_bin"
)