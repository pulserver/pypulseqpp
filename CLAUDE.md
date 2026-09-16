<!-- Generated from AGENTS.md by scripts/sync_agent_docs.sh. Do not edit. -->

# pypulseqpp — agent instructions

<!--
This file is the SOURCE. CLAUDE.md and GEMINI.md are generated from it by
scripts/sync_agent_docs.sh, which pre-commit runs. Edit this file, never those.
-->

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
| `tests/` | API, numerical, format-parity and invariant tests |
| `docs/` | Markdown/Sphinx documentation and generated API reference |
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

## Documentation and docstrings

`docs/documentation_style.md` is the documentation style guide, and it is
binding. Read it before writing or changing any docstring, documentation page,
code comment or user-facing diagnostic string, including the Doxygen comments
under `src/cpp/`. It carries the terminology table, the register rules, the
unit/frame/raster conventions, the safety-language rules and the
source-of-truth requirements in full. What follows is the operational summary,
not a substitute for it.

- Documentation is written for human developers: MRI researchers and sequence
  developers who know MR physics and Pulseq. Optimize for precision and
  information density. Do not pad documentation to help an LLM read the code.
- Use the established MRI, MR physics, sequence-design and Pulseq term. Do not
  replace a technical concept with a paraphrase, a personification, a metaphor,
  a tagline or an invented informal label. Repeating the correct term beats
  stylistic variation.
- Keep these distinct: gradient amplitude, slew rate, moment/area, waveform and
  event; RF waveform, RF event, flip angle, phase/frequency offset and
  slice-selection gradient; ADC event, samples, dwell time, acquisition
  duration and sampling times; gradient waveform, k-space trajectory and ADC
  sampling locations; event delay, block duration, sequence timing and
  rasterization; logical sequence axes and physical gradient axes; Pulseq
  representation, pypulseqpp abstraction, PyPulseq compatibility and scanner
  execution.
- Fixed vocabulary: *interleaf/interleaves*, not "interleave" as a count noun;
  *prewinder/prephasing* and *rewinder/rephasing*, not "bridge"; *example
  sequences* or *sequence library*, not "zoo"; *wave-encoding gradients* for
  the events and *corkscrew trajectory* for the k-space path; k-space in 1/m.
- Units, coordinate frames, composition order, rasters and normalization
  conventions are API semantics. Document them, and verify them from the
  implementation, the tests or the specification rather than guessing.
- Safety language is narrow. State exactly what is checked or estimated and
  against which limit. Passing a timing, gradient, slew, continuity,
  mechanical-resonance, PNS or SAR check never means a sequence is scanner-safe
  or patient-safe. Preserve the existing disclaimers verbatim.
- Preserve NumPy-style docstrings. Write parameter, return and attribute types
  as Python 3.10 type expressions, not prose: `float | ArrayLike`,
  `NDArray[np.float64]`, `Sequence[int]`, `tuple[float, float, float]`,
  `str | os.PathLike[str]`, keeping numpydoc's `, optional` and `, default X`
  suffixes. Every documented public function, class and method carries a brief
  `Examples` section whose doctest runs.
- Summary lines classify. Functions and methods take a verb; classes,
  properties and attributes take a noun phrase. Do not restate names,
  annotations or obvious attributes. Private helpers need no filler docstring.
  Package and module docstrings state a responsibility briefly; architectural
  rationale belongs here or in `docs/`.
- Describe the code as it is, not its history. No "used to", "previously",
  "this replaces", no fixed-bug narratives, no comparisons with removed
  designs. Preserve non-obvious invariants and protect them with tests.
- Do not print measured constants that are not guaranteed across releases or
  hardware. Benchmark tables are generated by the benchmark scripts.
- Existing documentation is not a source of truth, but correct documentation is
  not to be churned. Do not rewrite prose that already uses the conventional
  term with its units and frame. When modifying code, clean up nearby
  documentation that clearly violates the guide; do not broaden a focused
  change into a repository-wide rewrite unless that is the task.