add_executable(tile_example_wave_specialization EXCLUDE_FROM_ALL wave_specialization.cpp)
target_compile_options(tile_example_wave_specialization PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)
