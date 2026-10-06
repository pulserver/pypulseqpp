# Development prerequisites

Development requires Git, Python 3.10–3.13, a C++17 compiler and CMake. Clone
all submodules; `external/MRArbGrad` is part of the native build and the
SeqEyes source is pinned separately under `viewer/`.

The reference file-format tests use the test-only dependency in
`tests/requirements.txt`. Tests requiring an unavailable reference fixture skip
rather than changing the runtime dependency set.
