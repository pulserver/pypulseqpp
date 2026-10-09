# Sequence module and function internals

The contracts of sequence modules and sequence functions. The concepts are in
{doc}`../../explanations/sequence-modules`.

## Sequence modules

A {class}`~pypulseqpp.sequences.SequenceModule` solves a reusable block layout
independently of the acquisition loop. It holds block tuples, named mutable
event templates and a timing reference.

```text
SequenceModule
├── blocks: [(event, ...), ...]
├── events: named mutable event templates
└── center: timing reference (s)
```

| Module family | Blocks and events | Responsibility |
| --- | --- | --- |
| Excitation | RF event, selection gradient, rephaser | Excitation or refocusing geometry and pulse timing. |
| Preparation | RF and gradient preparation blocks | Inversion, T2 preparation, saturation, diffusion, or magnetisation transfer. |
| Readout | Prephasing, ADC event, readout gradients, rewinding or spoiling | Echo timing, acquisition bandwidth, and sampling trajectory. |

Every excitation and preparation module is an
{class}`~pypulseqpp.sequences.RfModule`, which adds `sim_rf`, the
off-resonance simulation of the module's pulse, by default the first RF event
in block order.

### Construction

A subclass implements `init_module`, assigns `self.seq` (a
{class}`pypulseqpp.Sequence`; an upstream `Sequence` is refused), adds blocks to
it, and sets `center` when the module is timed against something other than its
start. Assigning `seq` makes it record the event identities and block tuples
added to it. A module that assigns no `seq`, or adds no blocks, raises
`TypeError`. The constructor forwards its arguments to `init_module`, and
`inspect.signature` and the class docstring report that method's parameters,
with the keywords a subclass forwards to its parent's `init_module` spelt out.
A family states its parameters once, and a variant narrows them.

### Publication

After `init_module` returns, the locals of every `init_module` frame in the
chain of subclasses are scanned; a local that names an event a block played is
published as an attribute of the module and on `events`. The innermost frame
is read first, so where a subclass and its base bind one name, the subclass's
meaning is published.

- Repeated references are deduplicated by identity: one distinct object
  becomes a scalar event, several become a list in first-seen order.
- `register(name=event)` publishes events no block played and preserves the
  supplied structure, including one-element lists; it takes precedence over
  automatic names.
- `publish()` publishes the caller's locals, for construction helpers whose
  locals are not captured automatically.
- A module that publishes nothing warns. A published name that is also a
  module attribute warns, and the event stays reachable as `events.<name>`; the
  module's own attribute wins.

### Reading a module

`blocks` returns a copy of the block tuples in play order, holding the
original event objects. Mutating those objects changes the published
templates and what the loop adds afterwards; it does not rewrite the stored
sequence, so the analyses forwarded to `seq` (`calculate_kspace`,
`check_timing`, `paper_plot`, `test_report`, `waveforms_and_times`) answer for
the layout as constructed. `duration` is summed from the blocks unless the
module derived its timing analytically and assigned it. An unknown attribute
raises `AttributeError` naming the module.

### Representation boundary

Module construction separates the fixed waveform and timing design from the
per-view encoding. This mirrors the Pulseq distinction between event
definitions and playout instances, described in {doc}`../../explanations/pulseq-file`.
Sampling order and repetition structure belong to the sequence function.

## Sequence functions

A complete sequence is designed from the system limits and a prescription: the
field of view, matrix, timing, flip angle and sampling order of the
acquisition. A sequence function is that design as a callable, from which a
script, the command line and a protocol editor obtain the sequence and the
prescription.

The signature is `sequence(system, **protocol)`. `system` is the
{class}`~pypulseqpp.Opts` the sequence is designed under. A function designed
for lower gradient or slew limits lowers `system` to them with
{func}`~pypulseqpp.cap_system`, which never raises a limit. The keyword
parameters after `system` are the protocol, and each states its unit in its
description. The function returns a {class}`~pypulseqpp.Sequence`, or, for a
scan with prescans, a list of sequences in play order. A new example module
exposes the function as `main`, which makes the module callable as it.

| Concern | Sequence function |
| --- | --- |
| Prescription | Keyword parameters after `system`, documented in the Parameters section |
| System limits | `system`, capped to the design's limits with `cap_system` |
| Modules, timing and sampling arrays | The body, before the loop |
| Sampling order | The loop |
| One repetition | The body of the loop |
| Pulseq definitions required by reconstruction | `set_definition` after the loop |
| Prescans | The elements of the returned list before the main sequence |

The Python module of a shipped sequence defines its design limits as the
constants `MAX_GRAD` (mT/m) and `MAX_SLEW` (T/m/s). Changing one redesigns the
gradient waveforms and may alter echo spacing, acquisition duration and the
constraint estimates.

### Protocol

A parameter's type and default come from its annotation and default value, and
its unit, choices and description from the NumPy-style Parameters section. The
unit is the parenthesised group in the first sentence of a description,
`Echo time (s).`, and the choices of a string parameter are the values its
documented type lists in braces. An annotation that admits `None` leaves the
value to the design. {func}`~pypulseqpp.sequences.parameters` returns one
{class}`~pypulseqpp.sequences.ProtocolParameter` per keyword parameter, in
signature order, which is what a protocol editor presents.

{func}`pypulseqpp.cli.run` derives one command-line option from each scalar
keyword parameter and takes its help text from the first sentence of the
parameter's description. The limit options `--max-grad-mtm` (mT/m) and
`--max-slew-tm-s` (T/m/s) build the `Opts` passed as `system`, and the result
is written with {func}`~pypulseqpp.sequences.write`.

(encoding-labels)=

### Encoding labels

A loop records encoding indices with Pulseq label extensions. The principal
labels are `LIN`, `PAR`, `ECO`, `SEG`, `REP` and `SET`. A label keeps its value
across blocks until an event changes it, so a block carries the events of the
labels it changes and no others. {class}`~pypulseqpp.sequences.Labels` holds
that state: calling it with the new value of each label returns the events to
add to the block. An unchanged value writes nothing, a change equal to the
label's previous change is an INC, and any other change is a SET.
{meth}`~pypulseqpp.Sequence.evaluate_labels` recovers the ADC order directly
from the sequence, so sampling figures can use the implemented acquisition
order rather than reconstructing it from the prescription.

### Prescans and chains

A prescan is a sequence played before the main sequence, such as a calibration
acquisition. {func}`~pypulseqpp.sequences.write` writes the list as separate
files: the first at the given path, and each later one beside it as
`<stem>_<Name>.seq`, where `Name` is the sequence's `Name` definition, or its
position in the list, counted from 0 at the first, when it has none. A later
file whose name an earlier file has taken is written as
`<stem>_<Name>_<position>.seq`, so every sequence has a file of its own. Each
file but the last names the next with `NextSequence`. The chain represents one
acquisition while each file keeps one period for
{meth}`~pypulseqpp.Sequence.repetition` and the analyses that use it.
{func}`~pypulseqpp.sequences.duration` is the sum of the durations of the
sequences in the chain.

### Checking a prescription

Calling a sequence function designs every block, and the number of blocks grows
with the matrix. A prescription the design cannot meet, such as an echo time
shorter than the readout admits, raises `ValueError`. The call does not
evaluate the waveforms against gradient, PNS or SAR limits, which take the
designed sequence ({doc}`../../explanations/safety-checks`).

The function records the prescription as designed in the definitions it sets
after the loop. A value the design chooses, such as the shortest echo time for
`te=None`, is recorded as `TE`, and prescribing the recorded value as `te`
designs a sequence that records the same `TE`. The receiver bandwidth as
designed is the reciprocal of the dwell time of the ADC events, which is
rounded to the ADC raster.
