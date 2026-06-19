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

set(LEETCUDA_ELEMENTWISE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/LeetCUDA/kernels/elementwise/elementwise.cu)
set(LEETCUDA_SGEMM_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/LeetCUDA/kernels/sgemm/sgemm.cu)
if(EXISTS ${LEETCUDA_ELEMENTWISE_SOURCE} AND EXISTS ${LEETCUDA_SGEMM_SOURCE})
  set(LEETCUDA_ELEMENTWISE_SLICE ${CMAKE_CURRENT_BINARY_DIR}/leetcuda_elementwise_add_f32x4_slice.cu)
  add_custom_command(
    OUTPUT ${LEETCUDA_ELEMENTWISE_SLICE}
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/extract_leetcuda_kernel.py
      ${LEETCUDA_ELEMENTWISE_SOURCE}
      --kernel elementwise_add_f32x4_kernel
      --output ${LEETCUDA_ELEMENTWISE_SLICE}
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/extract_leetcuda_kernel.py
      ${CMAKE_CURRENT_SOURCE_DIR}/block_sche/compiler/transform.py
      ${LEETCUDA_ELEMENTWISE_SOURCE}
    VERBATIM)

  set(GENERATED_LEETCUDA_ELEMENTWISE ${CMAKE_CURRENT_BINARY_DIR}/leetcuda_elementwise_add_f32x4_generated.cu)
  add_custom_command(
    OUTPUT ${GENERATED_LEETCUDA_ELEMENTWISE}
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/block_sche_compiler.py
      ${LEETCUDA_ELEMENTWISE_SLICE}
      --kernel elementwise_add_f32x4_kernel
      --drop-original
      --output ${GENERATED_LEETCUDA_ELEMENTWISE}
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/block_sche_compiler.py
      ${CMAKE_CURRENT_SOURCE_DIR}/block_sche/compiler/transform.py
      ${LEETCUDA_ELEMENTWISE_SLICE}
    VERBATIM)
  set_source_files_properties(${GENERATED_LEETCUDA_ELEMENTWISE} PROPERTIES GENERATED TRUE)

  add_executable(leetcuda_elementwise_e2e
    ${GENERATED_LEETCUDA_ELEMENTWISE}
    tests/leetcuda_elementwise_e2e_main.cu)
  target_link_libraries(leetcuda_elementwise_e2e PRIVATE block_sche_headers)

  set(LEETCUDA_SGEMM_SLICE ${CMAKE_CURRENT_BINARY_DIR}/leetcuda_sgemm_naive_slice.cu)
  add_custom_command(
    OUTPUT ${LEETCUDA_SGEMM_SLICE}
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/extract_leetcuda_kernel.py
      ${LEETCUDA_SGEMM_SOURCE}
      --kernel sgemm_naive_f32_kernel
      --output ${LEETCUDA_SGEMM_SLICE}
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/extract_leetcuda_kernel.py
      ${CMAKE_CURRENT_SOURCE_DIR}/block_sche/compiler/transform.py
      ${LEETCUDA_SGEMM_SOURCE}
    VERBATIM)

  set(GENERATED_LEETCUDA_SGEMM ${CMAKE_CURRENT_BINARY_DIR}/leetcuda_sgemm_naive_generated.cu)
  add_custom_command(
    OUTPUT ${GENERATED_LEETCUDA_SGEMM}
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/block_sche_compiler.py
      ${LEETCUDA_SGEMM_SLICE}
      --kernel sgemm_naive_f32_kernel
      --drop-original
      --output ${GENERATED_LEETCUDA_SGEMM}
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/block_sche_compiler.py
      ${CMAKE_CURRENT_SOURCE_DIR}/block_sche/compiler/transform.py
      ${LEETCUDA_SGEMM_SLICE}
    VERBATIM)
  set_source_files_properties(${GENERATED_LEETCUDA_SGEMM} PROPERTIES GENERATED TRUE)

  add_executable(leetcuda_sgemm_e2e
    ${GENERATED_LEETCUDA_SGEMM}
    tests/leetcuda_sgemm_e2e_main.cu)
  target_link_libraries(leetcuda_sgemm_e2e PRIVATE block_sche_headers)
endif()

enable_testing()
add_test(NAME schedule_layout_test COMMAND schedule_layout_test)
add_test(NAME vector_add_e2e COMMAND vector_add_e2e)
set_tests_properties(vector_add_e2e PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME vector_add_sm_affine_e2e COMMAND vector_add_sm_affine_e2e)
set_tests_properties(vector_add_sm_affine_e2e PROPERTIES SKIP_RETURN_CODE 77)
if(TARGET leetcuda_elementwise_e2e AND TARGET leetcuda_sgemm_e2e)
  add_test(NAME leetcuda_elementwise_e2e COMMAND leetcuda_elementwise_e2e)
  set_tests_properties(leetcuda_elementwise_e2e PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME leetcuda_sgemm_e2e COMMAND leetcuda_sgemm_e2e)
  set_tests_properties(leetcuda_sgemm_e2e PROPERTIES SKIP_RETURN_CODE 77)
endif()
