
function(add_my_test TARGET)
  add_executable(${TARGET} ${TARGET}.cpp)
  target_link_libraries(${TARGET} PRIVATE parlay_atomic_shared_ptr gtest_main)
  target_compile_options(${TARGET} PRIVATE -mcx16)

  if(NOT MSVC)
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wfatal-errors)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_options(${TARGET} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${TARGET} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif (CMAKE_BUILD_TYPE STREQUAL "Debug")
  endif()

  add_test(${TARGET} ${TARGET})
  message(STATUS "Added test target " ${TARGET})
endfunction()

add_my_test(test_basic_shared_ptr)
add_my_test(test_basic_atomic_shared_ptr)

add_my_test(test_shared_ptr)
add_my_test(test_atomic_shared_ptr_custom)
