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
repetition detection, timing and gradient checks, waveform and k-space analysis,
FOV transforms, RF/gradient design, sampling, and reusable sequence modules.
PNS, RF-power and mechanical-resonance analysis and tiling are deferred.

Scanner execution, segmentation, protocol contracts and consoles belong to
Pulserver. Vendor-specific execution logic does not belong here.

Runtime dependencies are NumPy, SciPy and PyPulseq. The native extension links
the standard library and threads. The optional GPL viewer is distributed
separately from the MIT core.

## Layout

| Path | Purpose |
|---|---|
| `src/cpp/pulseq/` | Python-independent C++17 storage, codecs, I/O and analysis |
| `src/cpp/bindings/` | CPython/pybind11 bindings for `pypulseqpp._ext`, including `arbgrad`, `sampling` and `slr` |
| `external/MRArbGrad/` | Vendored gradient solver submodule; see `external/NOTICE.md` |
| `src/pypulseqpp/` | Python facade, event conversion, sequence operations and design |
| `src/pypulseqpp/sequences/` | Reusable excitation, preparation and readout modules |
| `src/pypulseqpp/cli/` | Signature-driven command-line parsing and sequence writing |
| `examples/sequence/` | Complete scripts, installed as `pypulseqpp.sequences.<name>` |
| `tests/` | API, numerical, format-parity and invariant tests |
| `docs/` | Markdown/Sphinx documentation and generated API reference |
| `viewer/` | Separate `pypulseqpp-seqeyes` package; excluded from the core distribution |

Do not edit vendored submodule contents as part of core maintenance.

## Build and test

```bash
git submodule update --init --recursive
pip install -e '.[dev]'
bash scripts/format_and_lint.sh --check
pytest -q
```

Build and test steps are mandatory before reporting a change complete. Run
them and report their actual results; do not assume success. The formatting
script rewrites files when called without `--check`.

For documentation changes, also run:

```bash
sphinx-build -W --keep-going -b html docs docs/build/html
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

Example modules are callable through their `main` functions. The CLI derives
flags from signatures and help text from NumPy-style Parameters sections.

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

Documentation in this project is written primarily for human developers. Optimize for clarity, precision, and high information density. Do not make documentation verbose in order to help an LLM understand the code.

### General principles

Preserve NumPy-style Python docstrings. Write concise technical prose for
developers and MR scientists. Document units, frames, composition order,
state, side effects and non-obvious return conventions where useful.

Do not restate names, annotations, obvious attributes or implementation steps.
Private helpers need no filler docstrings. Package/module docstrings describe
purpose briefly; architectural constraints belong here or in dedicated docs.

Describe the code as it is, not its history. Avoid “used to”, “previously”,
“this replaces”, fixed-bug narratives and comparisons with removed designs.
Preserve non-obvious invariants a maintainer could accidentally break, and
protect them with tests where possible.

Do not print measured constants that are not guaranteed across releases or
hardware. Use symbols and their sources. Benchmark tables must be generated
by benchmark scripts rather than maintained by hand.

* Document information that is not obvious from names, signatures, type annotations, or the implementation itself.
* Prefer direct technical prose over narrative, tutorial-style, conversational, literary, or essay-like explanations.
* Do not use docstrings to record your reasoning process or to narrate how the code works line by line.
* Do not restate the signature in prose.
* Do not document parameters or attributes with descriptions that merely repeat their names or types.
* Do not add documentation solely for completeness or because a symbol exists.
* Preserve the project's established docstring format and terminology.

Conciseness is a means, not the goal. Preserve enough detail to state non-obvious contracts precisely.

### Information worth documenting

Document these when relevant and non-obvious:

* purpose and externally visible behavior;
* physical units;
* coordinate or reference frames;
* transformation/composition order;
* invariants and state transitions;
* side effects;
* important preconditions or assumptions;
* non-obvious return conventions;
* behavior at boundaries or special values;
* state whose meaning is not apparent from its name/type;
* compatibility constraints;
* surprising behavior that is intentional and must be preserved.

These details are more important than minimizing line count.

### Packages and modules

Package and module docstrings should normally be brief: usually a one-line summary or a few sentences describing the responsibility of the package/module.

Do not put a design essay, implementation walkthrough, usage tutorial, or historical rationale in a module docstring. Put substantial architectural rationale in dedicated documentation, or a focused code comment if it is local to an implementation decision.

### Classes

A class docstring should explain what abstraction the class represents and any important semantic conventions.

Document constructor parameters and public attributes when their meaning is useful and not obvious. Do not mechanically enumerate every attribute.

For stateful classes, document state variables whose interpretation or lifecycle would otherwise be unclear.

### Functions and methods

State what the operation means rather than narrating its implementation.

Document parameters, return values, exceptions, units, frames, side effects, or special cases only where they convey useful semantics beyond the signature.

A short precise statement is preferred to a long explanatory paragraph.

### Private and helper functions

Private helpers do not require docstrings merely because they are functions.

Add or retain a helper docstring when it communicates a non-obvious contract, invariant, state transition, algorithmic assumption, side effect, special return convention, or other information useful to a maintainer.

If a private helper's behavior is obvious from its name, signature, and short implementation, omit the docstring rather than adding filler.

### Comments versus docstrings

Use docstrings for the contract and semantics of an abstraction.

Use local comments for implementation details, algorithmic tricks, performance-sensitive choices, and explanations of why a particular piece of code is written in a non-obvious way.

Do not move local implementation commentary into a docstring simply to preserve it.

### Style to avoid

Avoid generated prose such as:

* extended scenarios used where a direct rule would suffice;
* phrases describing code metaphorically or narratively;
* repeated explanations of implementation mechanics;
* obvious descriptions such as "the first value", "the system options", or "helper for X";
* commentary about what is "common", "usually", or "nearly all" unless this is a meaningful documented constraint;
* large `Parameters` or `Attributes` sections containing mostly information already present in type annotations;
* statements whose primary purpose is to make the code easier for an LLM to reconstruct.

When modifying existing code, clean up nearby documentation that clearly violates these rules, but do not broaden an otherwise focused code change into a repository-wide documentation rewrite unless requested.