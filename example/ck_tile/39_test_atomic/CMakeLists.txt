add_executable(test_atomic_kernel EXCLUDE_FROM_ALL test_atomic.cpp)
target_compile_options(test_atomic_kernel PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)