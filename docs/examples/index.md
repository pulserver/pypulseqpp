# Examples

Executable pages, run when the documentation is built, so every figure and
every printed number on them comes from the code as it stands.

| Section | What it holds |
| --- | --- |
| {doc}`/examples/getting-started` | The minimal Pulseq workflow, event by event. |
| {doc}`/examples/sequence-modules/index` | The reusable modules the package designs sequences from. |
| {doc}`/examples/built-in-sequences/index` | Every complete sequence the package ships, with its diagram and its acquisition order. |
| {doc}`/examples/building-sequences` | Assembling modules, a prescription and a scan loop into an application. |
| {doc}`/examples/custom-modules` | Implementing new modules against the base-class contract. |

The concepts these pages rely on are in {doc}`/explanations/index`: the Pulseq
representation, the design abstractions, and the gradient, stimulation and SAR
constraints. The interfaces they call are documented in {doc}`/api/index`, and
the prescriptions of the shipped sequences in the {doc}`catalogue </sequences>`.

```{toctree}
:hidden:

/examples/getting-started
/examples/sequence-modules/index
/examples/built-in-sequences/index
/examples/building-sequences
/examples/custom-modules
```
