# Examples

Executable pages, run when the documentation is built, so every figure and
printed number is produced by the code as it stands. Concepts are in
{doc}`/explanations/index`, interfaces in {doc}`/api/index`, and the
prescriptions of the shipped sequences in the {doc}`catalogue </sequences>`.

The Course is the shortest coherent path that gives a new user the framework's
core mental model and enough practical competence to work independently. Tours
are useful applications, advanced branches or specialised workflows that are
not necessary for that core competence.

## Course

Ten lessons read in order: each starts from what the previous one finished with
and adds one construct.

| Section | Lessons | Covers |
| --- | --- | --- |
| {doc}`/examples/pulseq-basics` | 1–3 | Events, blocks and timing, from a free induction decay to a slice-selective gradient echo. |
| {doc}`/examples/spoiling` | 4 | Gradient and RF spoiling of the residual transverse magnetisation. |
| {doc}`/examples/gre-to-epi` | 5 | A train of echoes with phase-encode blips: single-shot echo planar imaging. |
| {doc}`/examples/non-cartesian` | 6 | Radial spokes in place of phase encodes, and golden-angle ordering. |
| {doc}`/examples/sequence-modules` | 7–8 | The excitation and readout modules, and the sequence function that plays them. |
| {doc}`/examples/checks` | 9 | The constraint checks applied to a finished sequence. |
| {doc}`/examples/custom-modules` | 10 | A new module written against the base-class contract. |

## Tours

Standalone examples that build on the Course; each names the lessons it
assumes.

| Section | Covers |
| --- | --- |
| {doc}`/examples/tours` | Multi-echo and segmented echo planar readouts, a spiral readout, and two custom readout modules. |

## Sequence catalogue

| Section | Covers |
| --- | --- |
| {doc}`/examples/built-in-sequences/index` | Each complete sequence at a representative prescription: diagram, sampling and acquisition order. |

```{toctree}
:hidden:

/examples/pulseq-basics
/examples/spoiling
/examples/gre-to-epi
/examples/non-cartesian
/examples/sequence-modules
/examples/checks
/examples/custom-modules
/examples/tours
/examples/built-in-sequences/index
```
