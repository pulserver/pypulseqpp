---
name: add-a-sequence
description: Add a complete sequence to examples/sequence/, or change one that is already shipped. Use when a new sequence function has to be written, or a shipped one changed, classified in the catalogue, given a gallery page and covered by tests, so that the documentation build and the docs tests stay consistent.
---

# Add a shipped sequence

A shipped sequence is a module under `examples/sequence/`, installed as
`pypulseqpp.sequences.<name>`. Adding one touches five places; the docs tests
fail if any is left out.

## 1. The sequence module

A new sequence is a function `sequence(system, **protocol)` that returns a
`Sequence`, or a list of sequences with the prescans first and the main
sequence last. The module sets `main = <function>` at module level, so that the
module is callable as that `main`, and runs it through `cli.run` when executed
as a script.

- The keyword parameters after `system` are the protocol. The CLI derives its
  flags from the signature and its help text from the NumPy-style `Parameters`
  section, and `sequences.parameters` reads the same for protocol editors. Every
  parameter with a default records it as ``default=<repr>``, and the documented
  value has to match the signature — `tests/test_docstring_defaults.py` enforces
  this.
- A quantity's unit is the parenthesised group in the first sentence of its
  description, `Echo time (s).`, and a string parameter lists its values in its
  type, `{'slab', 'nonselective'}`.
- The module constants `MAX_GRAD` (mT/m) and `MAX_SLEW` (T/m/s) have no
  default: the function holds `system` to them with `pp.cap_system`, and the
  description of `system` states both, as in `(80 mT/m)` and `(200 T/m/s)`.
  `tests/test_examples.py` holds the statement to the constants.
- The function raises `ValueError` for a prescription it cannot design, writes
  label events with `sequences.Labels`, records the definitions a
  reconstruction reads, and returns the sequence. Settings a user does not
  prescribe are module constants.
- Prescans are further sequences in the returned list. `sequences.write` and
  `cli.run` write the list as files linked through `NextSequence`, so each file
  stays one repeating unit; a prescan's `Name` definition names its file.
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
