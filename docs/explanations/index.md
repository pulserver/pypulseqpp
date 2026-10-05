# Explanations

Conceptual background for the interfaces documented in {doc}`../api/index` and
applied in the {doc}`examples <../examples/index>`. These pages state the
vocabulary, the models and the conventions the rest of the documentation
assumes.

| Explanation | What it covers |
| --- | --- |
| {doc}`pulseq-representation` | What a block holds, what each kind of event carries, how an extension is chained onto one, and the repeating unit of a sequence. |
| {doc}`shapes-and-storage` | How events and shapes are stored, deduplicated and signed, and what changes between format revisions. |
| {doc}`timing-and-rasters` | The rasters an event time is addressed on, how a block duration follows from them, and what the timing check establishes. |
| {doc}`sequence-design` | Sequence modules, sequence functions, and the sampling support and ordering that feed them. |
| {doc}`constraint-checks` | The checks, the system each is evaluated against, the frames, report units and evaluation intervals they share. |
| {doc}`gradient-constraints` | Gradient amplitude, slew rate within a block, and continuity across block boundaries. |
| {doc}`pns` | Peripheral nerve stimulation models and the response the check compares with a threshold. |
| {doc}`mechanical-resonance` | The windowed gradient spectrum and the forbidden bands it is compared with. |
| {doc}`sar` | Window-averaged local and global SAR from virtual observation points. |

## Checks and their limits

The checks in {mod}`pypulseqpp.safety` evaluate a finished sequence against
stated limits and models. They are **design-time estimates**. They do not
replace the scanner's own gate before download, nor its hardware monitor during
the scan, and they do not establish patient safety.

```{toctree}
:hidden:

pulseq-representation
shapes-and-storage
timing-and-rasters
sequence-design
constraint-checks
gradient-constraints
pns
mechanical-resonance
sar
```
