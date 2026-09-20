# Guides

Task-oriented procedures for sequence construction, analysis and constraint
checking. Conceptual background is
in {doc}`../explanations/index`, and the exhaustive interface semantics in
{doc}`../api/index`.

{doc}`first-sequence`
: Build a two-dimensional spoiled gradient echo from individual Pulseq events,
  check its timing, and write it as a `.seq` file.

{doc}`analysing-a-sequence`
: Inspect a finished sequence: its structure and encoding indices, its timing,
  its waveforms and k-space trajectory, and the figures that show them.

{doc}`checking-constraints`
: Run the gradient amplitude, slew-rate, continuity, nerve-stimulation,
  mechanical-resonance and SAR checks, and read their reports.

{doc}`custom-module`
: Assemble excitation, preparation and readout modules into a scan loop, and
  turn the assembly into a reusable sequence application.

{doc}`command-line`
: Run a shipped sequence from a shell, write its file, and give a sequence of
  your own the same interface.

```{toctree}
:hidden:

first-sequence
analysing-a-sequence
checking-constraints
custom-module
command-line
```
