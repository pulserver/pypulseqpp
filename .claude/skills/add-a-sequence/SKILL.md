---
name: add-a-sequence
description: Add a complete sequence to examples/sequence/, or change one that is already shipped. Use when a new SequenceApp has to be written, classified in the catalogue, given a gallery page and covered by tests, so that the documentation build and the docs tests stay consistent.
---

# Add a shipped sequence

A shipped sequence is a module under `examples/sequence/`, installed as
`pypulseqpp.sequences.<name>`. Adding one touches five places; the docs tests
fail if any is left out.

## 1. The application module

One `SequenceApp` subclass per module, with `main = <App>.main` at module
level, so the module is callable as that `main`.

- `MAX_GRAD` and `MAX_SLEW` are class attributes with no default: an
  application states the limits it is designed under. Every other setting a
  user does not prescribe is a class attribute a subclass can override.
- `init_sequence` designs the events and the sampling order. Its signature is
  the protocol: the CLI derives its flags from it, and its NumPy-style
  `Parameters` section is the help text. Every parameter with a default
  records it as ``default=<repr>``, and the documented value has to match the
  signature — `tests/test_docstring_defaults.py` enforces this.
- `kernel` adds the blocks of one repetition; `loop` plays the scan by calling
  `kernel` once per repetition; `design()` wraps `loop` with a fresh sequence
  and `finalize`.
- Prescans listed by `prescans()` are written by `write()` as separate files
  linked through `NextSequence`, so each file stays one repeating unit.
- Module-level helpers may stay in the script, but nothing may be imported
  from `examples/`.

The summary line of `main`'s docstring is read verbatim into the catalogue, so
it classifies or describes the sequence and is not a tagline.

## 2. The catalogue row

Add a `SequenceDoc` to `SEQUENCES` in `docs/sequence_reference.py`: the module
name, title, family (one of `FAMILIES`), dimensionality, sampling and readout.
This is the single place a sequence's classification is stated; the family
tables in `docs/sequences.md` and the per-sequence reference page are written
from it.

## 3. The gallery script

Add `gallery/<nn>-<family>/<module>.py`, named after the module — the
reference page links its example by name, and exactly one script may match.
A built-in sequence tour calls `paper_plot()` with no arguments, exercising
the automatic selection, and leaves constraint checks to the safety example
rather than repeating them.

## 4. The family page

Add a row and a toctree entry in that family's page under
`docs/examples/built-in-sequences/`. If the family is new, it also needs an
entry in `FAMILIES`, a heading in `docs/sequences.md` and a directory listed
in `GALLERY_SECTIONS` in `docs/conf.py`.

## 5. Tests

Extend the `tests/test_zoo_<family>.py` that covers the family: timing,
gradient and k-space invariants, not a smoke test. Use pytest functions and
fixtures, never `unittest.TestCase`, and name each test after the invariant it
states.

## Validate

```bash
pytest -q
bash scripts/build_docs.sh
```

The gallery script is executed as the pages are built, so a sequence whose
example cannot run cannot be merged.
