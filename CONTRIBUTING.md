# Contributing to pypulseqpp

## Getting set up

```bash
git clone --recurse-submodules https://github.com/pulserver/pypulseqpp.git
cd pypulseqpp
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]' -r tests/requirements.txt
pre-commit install
```

`tests/requirements.txt` holds the reference toolbox the file-format tests are
checked against. It is named by URL rather than published to PyPI, and a
distribution whose metadata carries a direct reference cannot be uploaded, so
it cannot live in an extra. The tests that need it skip without it.

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

Two documents govern this, and both are binding. Read them before changing a
docstring, a page under `docs/`, a code comment or a user-facing diagnostic
string.

[`docs/contributing/documentation.md`](docs/contributing/documentation.md) is
the documentation guide: what belongs in reference, in an explanation, in a
gallery example and in a how-to guide, and the register each is written in.

[`docs/contributing/terminology.md`](docs/contributing/terminology.md) is the
project's terminology and conventions: the established MRI and Pulseq terms,
the unit, frame and raster conventions, the four layers a statement can be
about, the safety-language rules and what has to be verified against which
source.

In short: use the established MRI and Pulseq term rather than a paraphrase of
it; document units, coordinate frames, composition order and rasters as part of
the API; keep safety language to exactly what is checked; and write for someone
reading the code as it is now, with no memory of an earlier version — no "used
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

Open `docs/build/html/index.html`. How-to guides live in `docs/guides/`,
conceptual pages in `docs/explanations/`, the API reference in `docs/api/`, the
executable examples in `gallery/`, and the shipped sequences are catalogued in
`docs/sequences.md`.

## Releasing

The two packages are released from tags of their own, and PyPI trusts the
workflow the tag starts rather than an API token.

`pypulseqpp` versions come from git tags via `setuptools_scm`. Push a tag
matching `v[0-9]+.[0-9]+.[0-9]+` and `tags-release.yml` builds the wheels and
the source distribution, publishes them to PyPI and signs a GitHub release.

`pypulseqpp-seqeyes` carries the version of the SeqEyes release its submodule
is pinned to, written in `viewer/pyproject.toml`. Push `viewer-v<that version>`
and `viewer-release.yml` publishes it. Release it before a `pypulseqpp` version
that raises the `plot` extra's floor, so the requirement resolves.

Trusted publishing matches the workflow the *publishing job* is written in, not
the reusable workflow it calls for the build. On PyPI, `pypulseqpp` trusts
`tags-release.yml` in the `pypi` environment and `pypulseqpp-seqeyes` trusts
`viewer-release.yml` in `pypi-viewer`.

## Documentation site

`docs.yml` publishes the built documentation to the `gh-pages` branch, which
GitHub Pages serves from its root. The site holds one directory per version:

- `latest` is main, republished on every push to it.
- `stable` is the newest release, so it is where the root redirects and where
  `docs/conf.py` points the canonical link of every released page.
- `vX.Y.Z` archives each release as it was published.

Each run replaces only the directories it publishes, so a release keeps the
pages it shipped with. A tag older than the newest release archives itself
without taking `stable` backwards. `docs/conf.py` reads the canonical
directory from `PYPULSEQPP_DOCS_VERSION`.

Pages can be deployed from a build artifact rather than a branch, but a
deployment replaces the whole site, which leaves nowhere for the other
versions to live.
