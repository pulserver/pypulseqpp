# Built-in sequences

Every complete sequence the package ships, designed and drawn at a
representative prescription. A page states the prescription, prints what the
design produced, and shows the sequence diagram and the order the acquisition
covers k-space in.

The prescription each application accepts, parameter by parameter, is on its
reference page in the {doc}`catalogue </sequences>`.

| Family | What the repetition holds |
| --- | --- |
| {doc}`/examples/built-in-sequences/gradient-echo` | One excitation, no refocusing pulse, spoiled between repetitions. |
| {doc}`/examples/built-in-sequences/spin-echo` | One excitation and one refocusing pulse, acquired at the refocused echo. |
| {doc}`/examples/built-in-sequences/fast-spin-echo` | One excitation and a CPMG train, one view per echo. |
| {doc}`/examples/built-in-sequences/mprage` | An inversion, an inversion time, and a spoiled gradient-echo train over one partition. |
| {doc}`/examples/built-in-sequences/bssfp` | One excitation with every gradient axis balanced across the repetition. |
| {doc}`/examples/built-in-sequences/epi` | One excitation and a train of readout lobes of alternating polarity. |
| {doc}`/examples/built-in-sequences/zte` | A hard pulse transmitted with the readout gradient already at amplitude. |

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
