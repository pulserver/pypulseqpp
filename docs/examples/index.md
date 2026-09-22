# Examples

Executable pages, run when the documentation is built, so every figure and
every printed number on them is produced by the code as it stands.

The first seven sections are a course. Each page starts from the sequence the
previous one finished with and adds one thing to it, states what that thing
changes, and measures it, so the sections are written to be read in order. The
last section is not part of the course: it is one page per shipped sequence,
each with a representative prescription and the figures that prescription
produces.

| Section | What it covers |
| --- | --- |
| {doc}`/examples/pulseq-basics` | Events, blocks and timing, up to a slice-selective gradient echo. |
| {doc}`/examples/spoiling` | What a repetition leaves behind, and the gradient and RF spoiling that keep it out of the next one. |
| {doc}`/examples/gre-to-epi` | Several echoes per excitation, up to a single-shot echo planar acquisition. |
| {doc}`/examples/non-cartesian` | Radial spokes and spiral arms, and the limits that bound their traversal. |
| {doc}`/examples/sequence-modules` | The excitation and readout designers, and the application that plays them. |
| {doc}`/examples/checks` | The checks a finished sequence is put through, and what each one reports. |
| {doc}`/examples/custom-modules` | New modules written against the base-class contract. |
| {doc}`/examples/built-in-sequences/index` | Every complete sequence the package ships, with its diagram and its acquisition order. |

The concepts these pages rely on are in {doc}`/explanations/index`: the Pulseq
representation, the design abstractions, and the gradient, stimulation and SAR
constraints. The interfaces they call are documented in {doc}`/api/index`, and
the prescriptions of the shipped sequences in the {doc}`catalogue </sequences>`.

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
