# Contributing to pypulseqpp

## Getting set up

```bash
git clone --recurse-submodules https://github.com/pulserver/pypulseqpp.git
cd pypulseqpp
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
```

## Before you open a pull request

```bash
bash scripts/format_and_lint.sh   # rewrites in place
pytest -q
```

CI runs the same script with `--check`, alongside its platform-specific
builds and tests. If `pre-commit` rewrites a file, review and stage it again.

## What the tests expect

- pytest with plain functions and fixtures. No `unittest.TestCase` subclasses.
- A test name states the invariant it protects, so a failure reads as a
  sentence: `test_a_double_precision_basis_does_not_reach_the_kernel`.
- Check file-format changes against `pypulseq-matlab-like`, the MATLAB Pulseq
  transcription used by the reference fixtures. Check API behavior against
  upstream PyPulseq. Compare `.seq` output byte-for-byte and calculations
  numerically; support performance claims with measurements.

## Comments and documentation

Write for someone reading the code as it is now, with no memory of an earlier
version. Do not write text whose subject is the history of the code — no "used
to", "previously", "this replaces", no naming a bug that is fixed. A comment
earns its place by explaining a non-obvious algorithm or a choice a reader
would otherwise undo; prefer a well-named function, or a test whose name states
the invariant, because those cannot go stale silently.

## Building documentation

The development extra includes the documentation dependencies. Source pages
are Markdown; Sphinx renders the API from NumPy-style Python docstrings.

```bash
bash scripts/build_docs.sh
```

It compiles the checkout into `docs/build/site` and runs Sphinx against that
build, with warnings as errors; extra arguments are passed to `sphinx-build`.

Open `docs/build/html/index.html`. Preserve the NumPy convention and document
units, coordinate frames, mutation and non-obvious return conventions.
Private helpers need docstrings only when their contracts are not evident
from the signature and implementation. The user/developer guides and examples
remain scaffolds; API pages live in `docs/api/`.

## Releasing

Versions come from git tags via `setuptools_scm`. Push a tag matching
`v[0-9]+.[0-9]+.[0-9]+` and the release workflow builds, signs and publishes it.
