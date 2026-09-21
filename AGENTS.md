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
| `docs/user-guide/` | Installation, supported platforms and project-use procedures; the snippets are executed as doctests |
| `docs/examples/` | The executable pages' landing pages, one per gallery directory |
| `docs/explanations/` | Conceptual explanation pages and their build-time figures |
| `docs/api/` | API reference pages; autosummary writes the stubs under `docs/generated/` |
| `docs/developer-guide/` | Contribution procedure and conventions, including the documentation guide and the project terminology |
| `docs/misc/` | Licensing, related projects and contributors |
| `viewer/` | Separate `pypulseqpp-seqeyes` package; excluded from the core distribution |

Do not edit vendored submodule contents as part of core maintenance.

`CLAUDE.md` and `GEMINI.md` point here, and `SKILLS.md` indexes the skills
under `.claude/skills/`: the procedures for building and testing the package,
writing documentation, and adding a shipped sequence. This page states the
rules; a skill states how a recurring task is carried out under them.

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

The gallery is executed as the pages are built, and the fast-spin-echo scripts
design their refocusing trains with `torchsim`, which the `design` extra brings.
`bash scripts/build_docs_pdf.sh` renders the single-file manual from the same
build and is what the release workflow attaches to a tag.

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

Two documents govern documentation, and both are binding:

- `docs/developer-guide/documentation.md` — the generic guide. What belongs in
  each form of documentation (API reference, gallery examples, conceptual
  explanation, tutorials and how-to guides) and how each should be written.
- `docs/developer-guide/terminology.md` — the pypulseqpp conventions. Terminology,
  register, units, frames, rasters, safety language and source-of-truth rules.

Read both before creating or substantially modifying documentation,
docstrings, examples, tutorials or explanatory material. Where they differ,
the project page governs terminology and conventions and the generic guide
governs documentation type and register. The summary below does not replace
either; it states the rules most often broken.

### The documentation types are not interchangeable

| Location | Type | Answers |
|---|---|---|
| `docs/user-guide/` | Task-oriented how-to | How do I install and use the project? |
| `docs/explanations/` | Conceptual explanation | Why does this work this way? |
| `gallery/` | Executable examples, built into `docs/generated/gallery/` | What does a representative scientific workflow look like? |
| `docs/api/` | Reference | What exactly does this object do? |
| `docs/developer-guide/` | Contributor procedure and conventions | How is this repository developed? |
| `docs/sequences.md` | Catalogue of the shipped sequences, grouped by family, with a reference page each | Which sequences exist, and what does one of them look like? |

Do not transfer the prose style or level of exposition of one type into
another. Explanations come before examples in the navigation, and an example
links to conceptual material rather than restating it.

A gallery example exists because running it and examining its output shows
something scientifically or computationally useful. An API demonstration, a
constructor catalogue, a conceptual introduction with incidental code,
command-line documentation, or a set of configurations whose only result is
that they run does not belong in the gallery. Prefer few strong examples.

A representative configuration of a shipped sequence, with its diagram, is
reference material and belongs on that sequence's page, not in the gallery.

### Mechanics

A gallery script is a `.py` file whose module docstring is the page's title
and opening; `# %%` starts a text cell, and code a reader would not type —
figure styling and print formatting — goes between
`# sphinx_gallery_start_ignore` and `# sphinx_gallery_end_ignore` so it runs
without appearing on the page. Every script is executed at build time, so an
example that cannot run cannot be merged.

`sphinx_gallery.gen_gallery` reads one level of subdirectories and no more, so
`gallery/` is flat: one directory per landing page, listed in `GALLERY_SECTIONS`.
The hierarchy a reader navigates is built by the pages under `docs/examples/`,
each of which carries a table of what is below it and a hidden toctree over the
same entries. Those toctrees are what nests the sidebar, which `max_navbar_depth`
renders to three levels. The tree sphinx-gallery writes is reachable through its
own root index, an `:orphan:`, so it contributes no navigation entries of its
own; a page being in both trees is a Sphinx info message, not a warning. Each
`README.rst` carries its own reStructuredText title above the `.. include::` of
its Markdown header, because a title arriving through an include leaves the
toctree beneath it outside the page's section.

Three generators run on `builder-inited` and write into `docs/generated/`, which
is not tracked.

`docs/explanation_figures.py` draws the explanation pages' figures from the code
being built.

`docs/sequence_reference.py` writes one reference page per shipped complete
sequence, plus the catalogue's family tables: its `SEQUENCES` table is the
single place a sequence's classification is stated, every description is read
from the application's own summary line, the prescription is rendered by
`autofunction` from the docstring, and a `minigallery` links the gallery page
that designs and draws it. A sequence added to `examples/sequence/` needs a row
there, a gallery script named after it in its family's directory, and a row in
that family's page under `docs/examples/built-in-sequences/`.

`docs/api_objects.py` collects the `autosummary` blocks of `docs/api/*.md` into
one `:orphan:` holder that owns their `:toctree:`. The API pages themselves
carry the object tables and no toctree, so the generated per-object stubs are
reachable and documented without every method and attribute landing in the
navigation tree.

### Terminology, in brief

- Use conventional MRI and Pulseq terminology. Do not replace an established
  term with a paraphrase of what it does.
- Keep these distinct: gradient *waveform*, gradient *event*, *amplitude*,
  *slew rate*, *area*/*moment*, and *k-space trajectory*. A waveform is not a
  trajectory, and a trajectory is not a set of ADC sampling locations.
- Keep these distinct: *RF waveform*, *RF event*, *flip angle*, and the
  playout parameters (frequency and phase offsets). An *ADC event* is the
  acquisition window; *samples*, *dwell time*, *receiver bandwidth* and
  *bandwidth per pixel* are separate quantities.
- Units, frames and rasters are API semantics, not incidental detail. State
  them.
- *interleaf* / *interleaves* are the count nouns for one shot of a multi-shot
  non-Cartesian trajectory.
- Four layers, never conflated: the **Pulseq representation** (what a `.seq`
  file holds), the **pypulseqpp abstraction** (compiled events, sequence
  modules, designers, checks), **PyPulseq interoperability** (a preserved
  upstream signature or event convention), and **scanner execution** (what an
  interpreter does on hardware, which belongs to Pulserver). Do not attribute
  a property of one layer to another.
- This package computes checks and estimates. A passing check does not
  establish that a sequence is safe to run on a scanner or on a subject.
  Never write "safe", "validated", "compliant" or "approved" of a sequence
  that passed one. Preserve the existing disclaimers verbatim.
- Write dry, declarative technical prose. Do not personify sequences,
  parameters, constraints, waveforms or files, and do not use a figurative
  verb where the technical relationship can be stated directly. Headings name
  the concept.

### Verification and revision

Verify substantive semantics against the implementation, the tests, the
`.seq` format authority (`pypulseq-matlab-like`), upstream PyPulseq, and the
Pulseq specification or primary literature — in that order. Existing prose is
not evidence.

When modifying existing documentation:

* preserve technically good material and avoid unrelated stylistic churn;
* move misplaced material to the correct documentation type rather than
  deleting it;
* rewrite only what is inaccurate, redundant, misplaced or stylistically
  inappropriate;
* flag unresolved semantic discrepancies rather than guessing.

When auditing documentation, check explicitly for conversational or literary
prose, paraphrastic replacements for established terminology, personification,
taglines, code narration, and technically imprecise attempts at accessibility.

After substantial documentation work, build the documentation, run the
documentation tests and the executed examples, and inspect the rendered
output, including the sidebar hierarchy.
