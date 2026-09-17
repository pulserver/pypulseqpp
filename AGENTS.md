# pypulseqpp — agent instructions

## Scope and compatibility

pypulseqpp provides PyPulseq-compatible sequence authoring over a C++ core.
Preserve supported Python signatures, event conventions and reference file
output. Not every upstream feature is implemented; do not document an absent
method as available.

The package owns sequence storage, text/binary I/O, deduplication, structural
repetition detection, timing, gradient, mechanical-resonance, PNS and SAR checks,
waveform and k-space analysis, FOV transforms, RF/gradient design and
reusable sequence modules. Tiling is deferred.

Sampling, view-ordering, angle and schedule helpers live in private modules
(`_masks`, `_sampling`, `_ordering`, `_epi`, `_angles`, `_schedules`) and are
withheld from the public namespace until the example sequences settle which of
them they need. Code that uses one imports it from its private module.

Scanner execution, segmentation, protocol contracts and consoles belong to
Pulserver. Vendor-specific execution logic does not belong here.

Runtime dependencies are NumPy, SciPy, PyPulseq and mrsd, which draws
`paper_plot` and needs nothing PyPulseq does not. The native extension links
the standard library and threads. The optional GPL viewer is distributed
separately from the MIT core.

## Layout

| Path | Purpose |
|---|---|
| `src/cpp/pulseq/` | Python-independent C++17 storage, codecs, I/O and analysis |
| `src/cpp/bindings/` | CPython/pybind11 bindings for `pypulseqpp._ext`, including `arbgrad`, `ptx`, `sampling`, `sim` and `slr` |
| `external/MRArbGrad/` | Vendored gradient solver submodule; see `external/NOTICE.md` |
| `src/pypulseqpp/` | Python facade, event conversion, sequence operations and design |
| `src/pypulseqpp/sequences/` | Reusable excitation, preparation and readout modules |
| `src/pypulseqpp/cli/` | Signature-driven command-line parsing and sequence writing |
| `src/pypulseqpp/plot/` | Figures: SeqEyes view, publication diagram, k-space and RF profiles |
| `examples/sequence/` | Complete scripts, installed as `pypulseqpp.sequences.<name>` |
| `gallery/` | sphinx-gallery example scripts, executed when the pages are built |
| `tests/` | API, numerical, format-parity and invariant tests |
| `docs/` | Markdown/Sphinx documentation, explanation pages and generated API reference |
| `viewer/` | Separate `pypulseqpp-seqeyes` package; excluded from the core distribution |

Do not edit vendored submodule contents as part of core maintenance.

## Build and test

```bash
git submodule update --init --recursive
pip install -e '.[dev]' -r tests/requirements.txt
bash scripts/format_and_lint.sh --check
pytest -q
```

Build and test steps are mandatory before reporting a change complete. Run
them and report their actual results; do not assume success. The formatting
script rewrites files when called without `--check`.

For documentation changes, also run:

```bash
bash scripts/build_docs.sh
```

The development extra includes documentation dependencies. Do not add a new
documentation or linting dependency solely for a cleanup.

## Language and performance constraints

- Native code targets C++17; Python targets 3.10+.
- Keep per-block loops in C++. Measure before adding allocations on the
  million-block hot path.
- Per-block binding calls use `METH_FASTCALL`, not pybind11 argument
  conversion. `add_block` registers all events in one native call.
- Scalar event fields use `PyMemberDef` offsets. Waveform accessors expose
  samples at their stored amplitude.
- Block-table arrays are copy-on-write snapshots. A view owns its buffer,
  survives sequence destruction and does not observe later mutations.
  Preserve the capacity-based fast-path ownership check.
- Substantial native work, including deduplication, compression and writing,
  releases the GIL.
- Every native mutation must invalidate derived analysis through the core
  revision counter. Keep the Python cache guard a revision comparison, not
  a per-block Python attribute update.
- Run `benchmarks/throughput.py` before and after changing bindings. Report
  measurements rather than asserting performance improvements.

## Facade and event contracts

`pypulseqpp` re-exports upstream authoring vocabulary alongside its own
implementations. Upstream imports and arithmetic helpers need not appear in
`__all__`. `add_ramps` and `add_custom_label` are intentionally withheld:
the upstream ramp solver is unsupported, and custom labels need no explicit
registration here.

The interoperability decorator converts compiled events to namespaces for
upstream functions and converts returned events back. This applies to event
consumers such as `calc_duration`, `align`, `split_gradient` and `rotate`,
not only factories.

A dynamic pTx pulse is one arbitrary RF event holding every transmit
channel's waveform one after another over a shared time base, so its time
shape restarts once per channel (Roos et al., Magn Reson Med 2025,
doi:10.1002/mrm.30601). The channel count is the number of samples at the
first sample time, as the reference interpreter reads it. `make_ptx_pulse`
writes the layout and `split_ptx_pulse` reads it; the core's timing check
judges one channel's time base, and flip-angle integrals take each channel on
its own time base and sum them, which is the flip where every channel has
unit, in-phase sensitivity. The rule lives in `src/cpp/pulseq/channels.hpp`.

Waveforms are normalised beside scalar amplitudes. Amplitude changes preserve
shape registrations; waveform replacement invalidates them. Registration IDs
are sequence-local. Decoded blocks are independent event snapshots.

## Storage invariants

Libraries and block indices are 1-based; zero event IDs mean absence.
Trapezoids and arbitrary gradients share file IDs but occupy separate tables.

Shape roles are a bit mask, recorded when events reference a shape.
Deduplication ORs the roles of merged shapes.

Event definitions separate fixed timing/shape data from playout parameters.
RF magnitude, phase and time shapes belong to definitions; arbitrary gradient
waveforms belong to instances. ADC and extension choices do not distinguish
block definitions. Pure delays share one definition independent of duration;
triggers and digital outputs are not pure delays.

Registration creates definitions. Deduplication rebuilds them after
renumbering shapes and events, so definition IDs are not stable across it.
Before deduplication, equal separately registered shapes can yield distinct
definitions.

## File-format invariants

`pypulseq-matlab-like`, a transcription of MATLAB Pulseq, is the file-format
authority. Upstream PyPulseq remains the Python API reference. The format
reference is a test-only Git dependency; tests requiring it skip when absent,
while the checked-in `tests/seq/` corpus remains usable.

Parse complete files before registering libraries and then blocks. Both text
and binary use the shared builder. Legacy text conversion derives missing RF
centres, gradient endpoints and pre-1.4 block durations. Missing RF uses stay
undefined unless inference is requested. Pre-1.4 shapes require forced
decoding before re-encoding because equal encoded/sample counts are ambiguous.

Default writers declare Pulseq 1.5.1. The 1.4.1 writer folds ppm offsets into
absolute offsets using gamma in Hz/T and field strength in tesla, drops RF
centre/use and gradient endpoints, warns when omitting soft delays, and
refuses rotation or RF-shim extensions.

Binary records are little-endian. Times use integer picoseconds and shape
samples use float32. Both forms support optional MD5 signatures.
`check_timing`, not writing alone, records `TotalDuration`.

Binary definition names and value counts use int32 lengths. Builtin label
IDs follow the reference order. Names beyond that table are stored in the
`CustomLabels` definition in assignment order; do not add a custom section.

## Analysis invariants

Use physical, rotated waveforms for played gradient limits. Simultaneous
vector peaks are not the norm of independently attained axis peaks.
Within-block slew and boundary continuity are separate checks.

FOV translation is expressed in logical metres. Prescription rotation is
composed after existing block rotation. The unbroken gradient integral used
for RF/ADC shift phase is distinct from excitation/reset-aware k-space used
for echo anchoring. Preserve both across consecutive ranges.

RF phase shapes store cycles; ADC modulation and event phase offsets use
radians. See `NEXT.md` for compatibility notes and the native headers for
individual range and state contracts.

## Examples and tests

Each example module defines one `SequenceApp` subclass and exposes
`main = <App>.main`, which builds and designs it; the module is callable as
that `main`. `init_sequence` designs, `loop` plays the scan by calling
`kernel` once per repetition, and `design()` wraps `loop` with a fresh
sequence and `finalize`. Prescans listed by `prescans()` are written
by `write()` as separate files linked through `NextSequence`, so each file
stays one repeating unit. Settings a user does not prescribe are class
attributes a subclass overrides; `MAX_GRAD` and `MAX_SLEW` have no default.
Module-level helpers may stay in the script, but nothing may be imported from
`examples/`. The CLI derives flags from `main`'s signature, which is
`init_sequence`'s, and help text from its NumPy-style Parameters section.

Editable package mappings can expose repository files that are not shipped.
Before importing modules discovered by walking a package path, check their
source location; importing a reachable `setup.py` can execute the build.

Use pytest functions and fixtures, never `unittest.TestCase`. Test names
state the invariant. Preserve:

- Reference byte parity with deduplication enabled and disabled. New event
  kinds need reference fixtures where the reference supports them.
- Numerical parity between native and plain implementations, with explicit
  tests for intentional differences.
- Text and binary round trips, including extensions and custom labels.
- Definition/instance separation and repetition-detection invariants.
- The fast calling convention and snapshot ownership.

Block numbering may close gaps in reference files; the noise-scan parity
test compares the remaining columns explicitly.

## Documentation

`docs/guides/developer/documentation.md` is the authoritative documentation
guide for both contributors and agents. Read and follow it before creating or
substantially modifying documentation, docstrings, examples, tutorials, or
explanatory material.

The documentation has four parts, and they are not interchangeable:
`docs/api/` is reference, `docs/explanations/` is conceptual explanation,
`gallery/` holds the executable examples sphinx-gallery builds into
`docs/generated/gallery/`, and `docs/sequences.md` catalogues the shipped
sequences. A gallery script is a `.py` file whose module docstring is the
page's title and opening; `# %%` starts a text cell, and code a reader would
not type — figure styling and print formatting — goes between
`# sphinx_gallery_start_ignore` and `# sphinx_gallery_end_ignore` so it runs
without appearing on the page. Every script is executed at build time, so an
example that cannot run cannot be merged. Explanation figures are functions in
`docs/explanation_figures.py`, registered in its `FIGURES` mapping and drawn
from the code being built rather than committed as images.

Treat its distinction between API reference, gallery examples, conceptual
explanation, and tutorials/how-to guides as a requirement. Do not transfer the
prose style or level of exposition of one documentation type into another.

Project-specific terminology and conventions take precedence over generic
examples in the guide.

When modifying existing documentation:

* verify substantive semantics against implementation, tests, authoritative
  upstream specifications/libraries, and literature where appropriate;
* do not treat existing prose as authoritative;
* preserve technically good documentation and avoid unrelated stylistic churn;
* flag unresolved semantic discrepancies rather than guessing;
* when useful material is in the wrong documentation type, move or develop it
  in the appropriate location rather than automatically deleting it.

When auditing or refactoring documentation, explicitly check for
conversational or literary LLM prose, paraphrastic replacements for
established technical terminology, personification, taglines, code narration,
and technically imprecise attempts at accessibility.

After substantial documentation work, build the documentation, run relevant
documentation tests/examples, and inspect the rendered output.
