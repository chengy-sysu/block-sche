cmake_minimum_required(VERSION 3.18)
project(block_sche LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_library(block_sche_headers INTERFACE)
target_include_directories(block_sche_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(schedule_layout_test tests/schedule_layout_test.cu)
target_link_libraries(schedule_layout_test PRIVATE block_sche_headers)

set(GENERATED_VECTOR_ADD ${CMAKE_CURRENT_BINARY_DIR}/vector_add_generated.cu)
add_custom_command(
  OUTPUT ${GENERATED_VECTOR_ADD}
  COMMAND
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/block_sche_compiler.py
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/vector_add.cu
    --kernel vector_add
    --drop-original
    --output ${GENERATED_VECTOR_ADD}
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sche_compiler.py
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sche/compiler/transform.py
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/vector_add.cu
  VERBATIM)
set_source_files_properties(${GENERATED_VECTOR_ADD} PROPERTIES GENERATED TRUE)

add_executable(vector_add_e2e ${GENERATED_VECTOR_ADD} tests/vector_add_e2e_main.cu)
target_link_libraries(vector_add_e2e PRIVATE block_sche_headers)

set(GENERATED_VECTOR_ADD_SM_AFFINE ${CMAKE_CURRENT_BINARY_DIR}/vector_add_sm_affine_generated.cu)
add_custom_command(
  OUTPUT ${GENERATED_VECTOR_ADD_SM_AFFINE}
  COMMAND
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/block_sche_compiler.py
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/vector_add.cu
    --kernel vector_add
    --drop-original
    --sm-affine
    --trace
    --output ${GENERATED_VECTOR_ADD_SM_AFFINE}
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sche_compiler.py
    ${CMAKE_CURRENT_SOURCE_DIR}/block_sche/compiler/transform.py
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/vector_add.cu
  VERBATIM)
set_source_files_properties(${GENERATED_VECTOR_ADD_SM_AFFINE} PROPERTIES GENERATED TRUE)

add_executable(vector_add_sm_affine_e2e
  ${GENERATED_VECTOR_ADD_SM_AFFINE}
  tests/vector_add_sm_affine_e2e_main.cu)
target_link_libraries(vector_add_sm_affine_e2e PRIVATE block_sche_headers)

enable_testing()
add_test(NAME schedule_layout_test COMMAND schedule_layout_test)
add_test(NAME vector_add_e2e COMMAND vector_add_e2e)
set_tests_properties(vector_add_e2e PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME vector_add_sm_affine_e2e COMMAND vector_add_sm_affine_e2e)
set_tests_properties(vector_add_sm_affine_e2e PROPERTIES SKIP_RETURN_CODE 77)
