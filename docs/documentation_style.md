# Documentation style guide

This guide governs every docstring, documentation page, code comment and
user-facing diagnostic string in this repository, including the C++ doc
comments under `src/cpp/`. It is binding on human contributors and on
automated agents alike; `AGENTS.md` requires compliance with it.

The target register is reference documentation for MRI researchers and
sequence developers, in the tradition of NumPy, SciPy and Pulseq. The reader
is assumed to know MR physics and Pulseq. The documentation's job is to state
what an object *is*, in the field's own vocabulary, together with the
semantics a signature cannot carry: units, frames, rasters, composition order,
state and invariants.

The failure mode this guide exists to prevent is not verbosity. It is the
replacement of an established MRI or Pulseq term by a paraphrase of what the
thing does — a relative clause, a personification, or an invented informal
label. A reader then has to reconstruct the standard concept from a
description of it. Repeating the correct technical term is always preferable
to varying the wording.

## 1. Terminology is the substance

Use the established term. Do not explain around it.

| Do not write | Write |
|---|---|
| the RF moments | RF pulse centre times |
| when the pulses act | RF pulse centre times, with frequency and phase offsets |
| where each ADC sample sits in k-space | the k-space location of each ADC sample |
| making an event land on the raster | rounding an event's timing to the raster |
| turning a k-space path into a gradient | converting a k-space trajectory to a gradient waveform |
| a block as the events it plays | the events a block contains |
| whether each gradient carries on from the block before it | gradient waveform continuity across block boundaries |
| the scanner to weigh it against | the system limits used for the check |
| nothing slews too fast | no axis exceeds `max_slew` |
| the sequence leaves its gradients at zero | all gradient waveforms end at zero amplitude |
| what the sequence is, as named statistics | sequence timing, encoding and gradient statistics |
| everything `[DEFINITIONS]` will carry | the definitions written to the `[DEFINITIONS]` section |
| what `key` says | the value recorded for `key` |
| the name `extension_id` stands for | the name `extension_id` maps to |
| what each soft delay stands for if nobody sets it | each soft delay's default value |
| the wave corkscrew | the wave-encoding gradients (the *trajectory* is the corkscrew) |
| a moment bridge | a prewinder / prephasing gradient, or a rewinder / rephasing gradient |
| the zoo | the example sequences, the sequence library |
| a wait that turns `minimum` into `requested` | the delay (s) that extends `minimum` to `requested` |
| size the two blocks | compute the durations of the two blocks |
| closes the TR on the read axis | rewinds and spoils the readout axis at the end of the TR |

`moment` is reserved for a gradient moment, the integral of a gradient
waveform over time. Never use it for a point in time.

### Distinctions that must not be blurred

These are separate concepts with separate names. Documentation that conflates
them is wrong, not merely informal.

**Gradients.** *Amplitude* (Hz/m or mT/m) is the instantaneous value.
*Slew rate* (Hz/m/s or T/m/s) is its time derivative. *Area* or *moment*
(1/m) is its time integral. A *waveform* is the sampled or piecewise-linear
shape. A *gradient event* is the Pulseq object — a trapezoid or an arbitrary
gradient with a channel, delay and amplitude — that plays a waveform in a
block. Say which one is meant.

**RF.** An *RF waveform* is the complex envelope on the RF raster. An *RF
event* is the Pulseq object holding that waveform together with delay,
amplitude, frequency and phase offsets, dead time, ringdown time, centre and
use. *Flip angle* is the integral property the waveform's amplitude is scaled
to reach. *Frequency offset* (Hz) and *phase offset* (rad) are playout
parameters, not properties of the waveform. A *slice-selection gradient* is a
gradient event played with the pulse; it is not part of the RF event.

**ADC.** An *ADC event* is the acquisition window. *Samples* is the count.
*Dwell time* is the sampling interval (s). *Acquisition duration* is
`num_samples * dwell`. *Sampling times* are the sample instants. *Receiver
bandwidth* is `1 / dwell`; *bandwidth per pixel* is `1 / (num_samples * dwell)`.
State which of the two a value is.

**Encoding.** A *gradient waveform* is what the hardware plays. A *k-space
trajectory* is its time integral, in 1/m. *ADC sampling locations* are the
points of that trajectory at the ADC sample times. A gradient waveform is not
a trajectory, and a trajectory is not a set of sampling locations.

**Timing.** An *event delay* offsets an event within its block. A *block
duration* is the block's own length. *Sequence timing* is the resulting
schedule. *Rasterization* is quantization onto the RF, gradient, ADC or block
duration raster. Do not use one of these words for another.

**Axes.** *Logical axes* are the sequence's own x, y, z gradient channels —
in imaging terms readout, phase encode and slice/partition. *Physical
gradient axes* are the scanner's, reached by applying each block's rotation
and any prescription rotation. Every amplitude, slew, PNS, resonance or
k-space statement must say which frame it is in.

**Layers.** The *Pulseq representation* is what a `.seq` file holds — blocks,
event libraries, shapes, definitions and extensions. The *pypulseqpp
abstraction* is what this package adds on top: compiled event objects,
sequence modules, designers and checks. *PyPulseq compatibility* means a
supported upstream signature and event convention is preserved. *Scanner
execution* is what an interpreter does on the hardware, and belongs to
Pulserver, not here. Do not attribute a property of one layer to another.

### Fixed vocabulary

- **interleaf / interleaves** for one shot of a multi-shot non-Cartesian
  trajectory. Not "interleave" as a count noun. "Interleave" remains correct
  as a verb, and public API identifiers are not renamed for this guide.
- **prewinder** or **prephasing gradient**; **rewinder** or **rephasing
  gradient**. Not "bridge".
- **example sequences**, **sequence implementations**, **sequence library**.
  Not "zoo". `SequenceApp` is a class name; refer to it by name or as the base
  class for complete, runnable sequence implementations. "App" is not project
  vocabulary.
- **wave-encoding gradients** for the events; **corkscrew trajectory** for the
  k-space path they produce.
- **k-space** coordinates are written in **1/m**. State once, where it helps,
  that this is cycles per metre; do not alternate between the two spellings.

## 2. Register

Write dry, declarative technical prose.

**No personification.** A shot does not walk, a gradient does not ask for
anything, a limit does not judge, a window is not still there, an encode does
not ride alongside another. Objects have properties and functions have
behavior.

**No literary compression.** Do not describe an object by a relative clause
where a noun exists: not "the container a sequence is written into" but "the
sequence container"; not "every figure is a function that takes what it draws"
but "each figure is a function of the object it draws"; not "what a complete
sequence asks of the scanner and of the subject" but the names of the checks.

**No taglines.** An API summary classifies its object. Write "Construct a
trapezoidal gradient event", not "the workhorse gradient builder".

**No metaphor for mechanism.** Describe the mechanism.

**No conversational or tutorial voice.** No second person, no "simply", "just",
"note that", "under the hood", "powerful", "handy", "seamless". No rhetorical
questions, no scene-setting.

**No history.** Describe the code as it is. No "used to", "previously", "this
replaces", no named fixed bugs, no comparison with removed designs.

**No hedging about the code's own naming.** If a parameter's name is
misleading, document what the parameter means. Do not write "despite the
parameter suffix" or otherwise apologise in the reference.

**Variation is not a virtue.** Use the same term for the same concept every
time it appears.

## 3. Summary lines

The first line is one sentence, on one line, ending in a period.

- **Functions and methods** take an imperative or third-person declarative
  verb that classifies the operation: "Construct…", "Return…", "Check…",
  "Convert…", "Design…", "Plot…". State what the operation means, not how it
  is implemented.
- **Classes** take a noun phrase naming what the class represents:
  "Rheobase-chronaxie nerve model, one coefficient set per physical axis."
  This is the numpydoc convention and is correct — do not convert existing
  noun-phrase class summaries into verb phrases.
- **Properties and attributes** take a noun phrase with its unit:
  "Gradient raster time in seconds."
- **Modules and packages** take one line naming the responsibility. A module
  docstring is not the place for a design essay, a tutorial or a rationale;
  those belong in `docs/` or in a local comment.

Do not restate the signature, the annotations or the parameter names in prose.

## 4. Units, frames and rasters are API semantics

Every documented quantity carries its unit. The package conventions are:

| Quantity | Unit |
|---|---|
| Gradient amplitude | Hz/m (mT/m only where explicitly converted) |
| Slew rate | Hz/m/s (T/m/s only where explicitly converted) |
| Gradient area / moment | 1/m |
| k-space coordinate | 1/m |
| RF amplitude | Hz |
| Time, duration, delay, dwell, raster | s |
| Phase offset, ADC phase modulation | rad |
| RF phase *shape* as stored in a shape library | cycles |
| Frequency offset | Hz |
| Gyromagnetic ratio | Hz/T |
| Field strength `B0` | T |
| SAR | W/kg |
| Mechanical-resonance amplitude | mT/m |

Departures from these conventions exist and are intentional. They must be
documented as the unit they use, and left alone:

- `make_wave_gradients` takes and reports its wave amplitude in **T/m**.
- `_arbgrad` designs return gradients in native **Hz/pixel** before conversion.
- The `safety` mechanical-resonance check reports amplitudes in **mT/m**.

State the coordinate frame wherever a gradient, slew, k-space or PNS quantity
appears: logical axes, or physical axes after block rotations. State whether a
rotation has already been applied and in what order rotations compose.

State the raster a time is quantized to, and which raster: RF, gradient, ADC
or block duration. A function that rounds must say which direction it rounds
and how ties are broken. Do not present a raster value as a default that
tracks `Opts` unless it actually does.

## 5. Checking and safety language

This package computes timing, gradient amplitude, slew rate, gradient
continuity, mechanical-resonance, PNS and SAR **checks and estimates**. It
does not establish that a sequence is safe to run on a scanner or on a
subject.

Rules:

- State exactly what is evaluated and against which limit. "Returns True when
  the largest per-axis gradient amplitude does not exceed `max_grad`" is
  correct; "returns True when the gradients are safe" is not.
- Never generalize a passing check into a safety claim. Do not write "safe",
  "scanner-safe", "patient-safe", "validated", "compliant" or "approved" of a
  sequence that passed a check.
- Do not imply a check covers more than it does. If a quantity is computed and
  reported but not compared against a limit — as the simultaneous vector
  gradient and slew magnitudes are — say so.
- Name the model and its source: SAFE, rheobase–chronaxie, VOP, a vendor
  forbidden-band table, an IEC limit. Say when a model is synthetic and for
  demonstration only.
- The existing disclaimers in `README.md`, `NEXT.md`, `pypulseqpp.safety` and
  `docs/api/safety.md` are correct. Preserve them; do not soften or reword
  them for style.

Diagnostics printed to users follow the same rules. Message templates
transcribed from the reference toolbox (`pypulseq-matlab-like`) are a
compatibility surface: do not reword them. Messages this package adds are
ours, and must be as precise as the reference's.

## 6. Source of truth

Existing documentation is not evidence. Before writing or changing a
substantive statement, verify it against, in order of authority:

1. the implementation being documented, including the C++ core;
2. the tests that protect it, particularly parity and invariant tests;
3. `pypulseq-matlab-like`, the file-format authority, for anything about the
   `.seq` format, its conventions or its diagnostics;
4. upstream PyPulseq, for Python API behavior and event conventions;
5. the Pulseq specification and the primary literature, cited by DOI.

Do not infer semantics from a name, a type annotation or another docstring.
If a contract cannot be established from these sources, say what is known and
leave the rest undocumented rather than guessing.

Cite literature for any designed pulse, trajectory, ordering or model, and
retain existing citations and third-party attributions verbatim.

Do not print measured constants that are not guaranteed across releases or
hardware. Benchmark numbers come from the benchmark scripts, never from a
hand-maintained table.

Every documented public function, class and method carries a brief `Examples`
section whose doctest runs under `pytest tests/test_docstrings.py`. Write
examples that demonstrate the contract — a unit, a shape, an invariant — not
that a call succeeds.

## 7. What to document, and what not to

Document, when it is not obvious from the name, signature or annotation:
externally visible behavior; units; coordinate frames; composition and
transformation order; invariants and state transitions; side effects;
preconditions; non-obvious return conventions; behavior at boundaries and
special values; compatibility constraints; and intentional surprises that must
be preserved.

Do not document a symbol merely because it exists. Do not restate names,
annotations or obvious attributes. Do not mechanically enumerate every
attribute of a class. Private helpers need a docstring only when they carry a
non-obvious contract, invariant, assumption, side effect or return convention.

Use docstrings for the contract of an abstraction. Use local comments for
implementation detail, algorithmic tricks, performance-sensitive choices and
the reason a piece of code is written in a non-obvious way. Do not move local
implementation commentary into a docstring to preserve it.

Conciseness is a means. Preserve enough detail to state a non-obvious contract
precisely.

## 8. Format

- NumPy-style docstrings throughout, rendered by `sphinx.ext.napoleon`.
- Parameter, return and attribute types are Python 3.10 type expressions, not
  prose: `float | ArrayLike`, `NDArray[np.float64]`, `Sequence[int]`,
  `tuple[float, float, float]`, `str | os.PathLike[str]`, keeping numpydoc's
  `, optional` and `, default X` suffixes.
- Documentation pages are Markdown with MyST roles and directives.
- C++ documentation comments are Doxygen-style and follow this guide's
  terminology and register rules in full.
- Use equations only after verifying the convention they express, and define
  every symbol.

## 9. Do not churn

Much of this repository's documentation is already correct. Rewriting correct
technical prose for stylistic preference wastes review effort and risks
introducing errors.

Leave alone:

- Documentation that already uses the conventional term, states its units and
  frame, and classifies its object.
- Noun-phrase class summaries and noun-phrase property docstrings.
- `Parameters`, `Attributes`, `Returns`, `Raises` and `Notes` sections that
  carry real contracts.
- Citation and third-party attribution blocks.
- The safety disclaimers of §5.
- `NEXT.md`, `docs/conf.py`, `scripts/`, and the release and documentation-site
  sections of `CONTRIBUTING.md`.
- Error message templates transcribed from the reference toolbox.

When modifying existing code, clean up nearby documentation that clearly
violates this guide. Do not broaden a focused change into a repository-wide
documentation rewrite unless that is the task.
