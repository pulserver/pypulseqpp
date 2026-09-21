---
name: build-and-test
description: Build pypulseqpp from this checkout and run its formatter, linter and test suite. Use before reporting any change to the package complete, and whenever a test result, a skip or a benchmark figure has to be quoted rather than assumed.
---

# Build and test

The native extension is compiled from `src/cpp/`, so a source checkout has to
be built before anything imports `pypulseqpp`. Run the whole sequence and
report the results it actually produced.

```bash
git submodule update --init --recursive
pip install -e '.[dev]' -r tests/requirements.txt
bash scripts/format_and_lint.sh --check
pytest -q
```

`scripts/format_and_lint.sh` runs `ruff format` and `ruff check`; without
`--check` it rewrites the files instead of reporting them. It is the same
entry point pre-commit and CI use, so a style failure in CI is reproducible
here.

## Interpreting the run

- `tests/requirements.txt` installs `pypulseq-matlab-like` from Git: the
  file-format authority, used for byte-parity fixtures. The tests that need it
  skip when it is absent, and the checked-in `tests/seq/` corpus still
  exercises the reader.
- Skips are expected, not failures: the reference toolbox leaves gaps in block
  numbering in some fixtures, the console package is not present here, MKL is
  optional, and the optimized fast-spin-echo trains need `torchsim` from the
  `design` extra.
- A new event kind needs a reference fixture wherever the reference supports
  it, and an intentional numerical difference from the plain implementation
  needs a test that states it.

## Changes to the bindings

Run `python benchmarks/throughput.py` before and after a change to
`src/cpp/bindings/` and report both measurements. Per-block calls use
`METH_FASTCALL`, scalar fields use `PyMemberDef` offsets, and block-table
snapshots are copy-on-write with a capacity-based ownership check; a change
that abandons one of these is a regression even when the tests pass.

## Documentation

Documentation changes are validated by building the documentation, not by
reading the source. See the `write-documentation` skill.
