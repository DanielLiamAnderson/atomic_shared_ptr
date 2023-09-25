function(add_google_benchmark NAME)
    add_executable(bench_${NAME} bench_${NAME}.cpp)
    target_link_libraries(bench_${NAME} PRIVATE parlay_atomic_shared_ptr benchmark_main)
endfunction()

function(add_benchmark NAME)
    add_executable(bench_${NAME} bench_${NAME}.cpp)
    target_link_libraries(bench_${NAME} PRIVATE parlay_atomic_shared_ptr)
endfunction()

add_google_benchmark(shared_ptrs)
add_google_benchmark(atomic_shared_ptrs)

