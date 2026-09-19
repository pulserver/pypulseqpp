# Sequence modules

A sequence module designs one group of blocks once and publishes its events for
a scan loop to place per view. These pages construct the shipped modules, print
their published event templates and resulting block layouts.

| Group | Module type |
| --- | --- |
| {doc}`/examples/sequence-modules/composition` | The module interface itself, and the interval two modules compose at. |
| {doc}`/examples/sequence-modules/rf` | One RF pulse and the gradients that select with it. |
| {doc}`/examples/sequence-modules/preparation` | The magnetisation the readout that follows will sample. |
| {doc}`/examples/sequence-modules/cartesian-readouts` | The views one excitation reads on a regular grid. |
| {doc}`/examples/sequence-modules/non-cartesian-readouts` | A trajectory solved against the gradient system and the receiver. |

```{toctree}
:hidden:

/examples/sequence-modules/composition
/examples/sequence-modules/rf
/examples/sequence-modules/preparation
/examples/sequence-modules/cartesian-readouts
/examples/sequence-modules/non-cartesian-readouts
```
