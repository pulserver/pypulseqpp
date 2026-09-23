# Sequence applications

```{admonition} TL;DR
:class: tldr

- A {class}`~pypulseqpp.sequences.SequenceApp` separates the prescription
  (`init_sequence`), the sampling order (`loop`), one repetition (`kernel`) and
  the Pulseq definitions required by reconstruction (`finalize`).
- {meth}`~pypulseqpp.sequences.SequenceApp.design` creates a fresh sequence,
  runs `loop` and applies `finalize`. A direct application call adds one
  kernel, so single-repetition inspection and complete design use the same code
  path.
- The `init_sequence` signature is the prescription exposed by `protocol`, the
  command line and protocol editors. Fixed design choices are class attributes,
  and construction caps the supplied system limits to the application's
  `MAX_GRAD` and `MAX_SLEW`.
- `kernel` records encoding indices as Pulseq labels.
  {meth}`~pypulseqpp.Sequence.evaluate_labels` recovers their ADC order from
  the sequence, so sampling figures can use the implemented acquisition order.
- Prescans are written by {meth}`~pypulseqpp.sequences.SequenceApp.write` as
  separate files, each naming the next with `NextSequence`, so each file
  retains a single repeating unit.
```

A {class}`~pypulseqpp.sequences.SequenceApp` combines a prescription, sampling
order, and repetition kernel into a complete sequence.

## Application structure

| Concern | Method | Result |
| --- | --- | --- |
| Prescription | `init_sequence` | Modules, timing, and sampling arrays. |
| Sampling order | `loop` | Repetition order for the complete acquisition. |
| One repetition | `kernel` | Blocks and labels for one shot or TR. |
| Metadata | `finalize` | Pulseq definitions required by reconstruction. |

Construction runs `init_sequence` without adding acquisition blocks.
{meth}`~pypulseqpp.sequences.SequenceApp.design` creates a fresh sequence,
runs `loop`, and applies `finalize`. Direct application calls add one kernel,
so single-repetition inspection and complete design use the same code path.

## Prescription and settings

The `init_sequence` signature is the prescription exposed by `protocol`, the
command line, and protocol editors. Its NumPy-style Parameters section defines
units and defaults. Fixed design choices are class attributes. Every concrete
application specifies `MAX_GRAD` and `MAX_SLEW`; construction caps the supplied
system limits to these values.

```python
class GentleEpi(Epi2DApp):
    MAX_SLEW = 60.0
```

Changing a design limit redesigns the gradient waveforms and may alter echo
spacing, acquisition duration, and constraint estimates.

## Encoding labels

`kernel` records encoding indices with Pulseq label extensions. The principal
labels are `LIN`, `PAR`, `ECO`, `SEG`, `REP`, and `SET`.
{meth}`~pypulseqpp.Sequence.evaluate_labels` recovers their ADC order directly
from the sequence. Sampling figures can therefore use the implemented
acquisition order rather than reconstructing it from the prescription.

## Prescans

Loops returned by {meth}`~pypulseqpp.sequences.SequenceApp.prescans` are
written by {meth}`~pypulseqpp.sequences.SequenceApp.write` as separate files
before the imaging sequence. Each file names the
next with `NextSequence`. The chain represents one acquisition while retaining
a single repeating unit per file for repetition-based analyses.

## See also

* {doc}`sequence-module` — reusable block layouts.
* {doc}`sampling` — acquired support and temporal ordering.
* {doc}`../../api/apps` — exact application interface.
* {doc}`../../sequences` — shipped complete sequences.
