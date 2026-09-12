cmake_minimum_required(VERSION 3.18)
project(local_endgame_player LANGUAGES CXX)
find_package(Python 3.13 REQUIRED COMPONENTS Interpreter Development.Module)
get_filename_component(VENDOR "${CMAKE_CURRENT_LIST_DIR}/../../third_party/2048EndgameTablebase/native_core" ABSOLUTE)
add_subdirectory("${VENDOR}/extern/nanobind" nanobind)
nanobind_add_module(ai_core
    "${VENDOR}/src/AIPlayer.cpp" "${VENDOR}/src/egtb_query.cpp"
    "${VENDOR}/src/egtb_data_256.cpp" "${VENDOR}/src/egtb_data_512.cpp"
    "${VENDOR}/src/egtb_data_1256.cpp" "${VENDOR}/src/bindings_ai.cpp")
nanobind_add_module(mover_core "${VENDOR}/src/bindings_mover.cpp")
foreach(target ai_core mover_core)
    target_include_directories(${target} PRIVATE "${VENDOR}/include")
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE -O3 -Wno-unknown-pragmas)
    set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${VENDOR}")
endforeach()
