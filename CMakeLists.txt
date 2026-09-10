cmake_minimum_required(VERSION 3.18)
project(pypulseqpp_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The trajectory solver is a submodule rather than a copy, so a checkout that
# forgot --recursive has no sources here. Say so now: the alternative is a
# hundred lines of missing-header errors.
set(MRARBGRAD ${CMAKE_SOURCE_DIR}/external/MRArbGrad/mrarbgrad_src/ext)
if(NOT EXISTS ${MRARBGRAD}/mag/Mag.cpp)
    message(FATAL_ERROR
        "external/MRArbGrad is empty. Run:\n"
        "    git submodule update --init --recursive")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)
find_package(Threads REQUIRED)

# One extension module holding the sequence core and its binding. List the
# sources explicitly: a glob makes a new file compile without anyone deciding
# it should.
pybind11_add_module(_ext
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/analysis.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/corners.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/dedup.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/md5.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/read.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/read_binary.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/write_binary.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/sequence.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/shape.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/fov.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/kspace.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/safety.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/timing.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/waveforms.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/pulseq/write_text.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bindings/module.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bindings/pulseqpp_eventtypes.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bindings/arbgrad.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bindings/sampling.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bindings/sim.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bindings/slr.cpp
    # The three translation units of the MRArbGrad solver; the rest of it is
    # headers. See external/NOTICE.md.
    ${MRARBGRAD}/mag/Mag.cpp
    ${MRARBGRAD}/utility/global.cpp
    ${MRARBGRAD}/utility/v3.cpp
)

target_include_directories(_ext PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bindings
    ${MRARBGRAD}
)

# Deduplication partitions independent work across standard C++ threads.
target_link_libraries(_ext PRIVATE Threads::Threads)

# Built into the build tree, never beside the sources: the wheel takes the
# module from the install step, so a binary left in the source tree would be
# swept into wheels built for every other interpreter too.
set_target_properties(_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/pypulseqpp"
)

install(TARGETS _ext DESTINATION pypulseqpp)
