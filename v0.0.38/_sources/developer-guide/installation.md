# Installation

## Prerequisites

Development requires Git, Python 3.10–3.14, a C++17 compiler and CMake. Clone
all submodules; `external/MRArbGrad` is part of the native build and the
SeqEyes source is pinned separately under `viewer/`.

The reference file-format tests use the test-only dependency in
`tests/requirements.txt`. Tests requiring an unavailable reference fixture skip
rather than changing the runtime dependency set.

## Editable installation

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' -r tests/requirements.txt
```

The `dev` extra contains lint, test and documentation tools. Gallery builds
also require the `examples` extra because optimized FSE examples use
`blochsim`:

```bash
python -m pip install -e '.[dev,examples]' -r tests/requirements.txt
```

## Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

The format-and-lint hook runs `scripts/format_and_lint.sh`, the script the
Style workflow runs with `--check`, so a commit that passes locally passes the
style check in CI. The hooks skip the submodules under `external/` and
`viewer/seqeyes/`.

A deliberate one-off bypass uses `git commit --no-verify`; one hook can be
skipped with `SKIP=<hook-id> git commit`. A bypass does not bypass CI and must
not be used to conceal a failing check.

## Documentation build

```bash
bash scripts/build_docs.sh
```

The build executes every gallery script and treats warnings as errors.
`bash scripts/build_docs_pdf.sh` renders the same build as one PDF.
