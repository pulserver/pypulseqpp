# Examples

Executable pages, run when the documentation is built, so every figure and
printed number is produced by the code as it stands. Concepts are in
{doc}`/explanations/index`, interfaces in {doc}`/api/index`, and the
prescriptions of the shipped sequences in the {doc}`catalogue </sequences>`.

## Pulseq course

Lessons read in order: each starts from the sequence the previous one finished
with and adds one construct.

| Section | Lessons | Covers |
| --- | --- | --- |
| {doc}`/examples/pulseq-basics` | 3 | Events, blocks and timing, up to a slice-selective gradient echo. |
| {doc}`/examples/spoiling` | 2 | Gradient and RF spoiling of the residual transverse magnetisation. |
| {doc}`/examples/gre-to-epi` | 3 | Several echoes per excitation, up to single-shot echo planar imaging. |
| {doc}`/examples/non-cartesian` | 2 | Radial spokes and spiral interleaves, and the limits on their traversal. |
| {doc}`/examples/sequence-modules` | 3 | The excitation and readout modules, and the application that plays them. |
| {doc}`/examples/checks` | 1 | The constraint checks applied to a finished sequence. |
| {doc}`/examples/custom-modules` | 3 | New modules written against the base-class contract. |

## Shipped sequences

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
/examples/built-in-sequences/index
```
