# Built-in sequences

Each shipped sequence is presented with a representative prescription,
resulting timing, sequence diagram, sampling geometry and acquisition order.

The prescription each sequence accepts, parameter by parameter, is on its
reference page in the {doc}`catalogue </sequences>`.

| Family | Contents |
| --- | --- |
| {doc}`/examples/built-in-sequences/gradient-echo` | Excitation and readout, spoiled between repetitions |
| {doc}`/examples/built-in-sequences/spin-echo` | Excitation and one refocusing pulse; readout at the spin echo |
| {doc}`/examples/built-in-sequences/fast-spin-echo` | Excitation and a CPMG refocusing train, one view per echo |
| {doc}`/examples/built-in-sequences/mprage` | Inversion, inversion time and a spoiled gradient-echo train per partition |
| {doc}`/examples/built-in-sequences/bssfp` | Excitation with all gradient axes balanced in each TR |
| {doc}`/examples/built-in-sequences/epi` | Excitation and a train of readout lobes of alternating polarity |
| {doc}`/examples/built-in-sequences/zte` | Hard pulse played with the readout gradient already on |

```{toctree}
:hidden:

/examples/built-in-sequences/gradient-echo
/examples/built-in-sequences/spin-echo
/examples/built-in-sequences/fast-spin-echo
/examples/built-in-sequences/mprage
/examples/built-in-sequences/bssfp
/examples/built-in-sequences/epi
/examples/built-in-sequences/zte
```
