# Events and blocks

The block events other than RF and gradients, and the operations on blocks and
events. Every factory returns a compiled event that
{meth}`Sequence.add_block` accepts; the RF and gradient factories are on
{doc}`rf` and {doc}`gradients`. Times are in s, frequency offsets in Hz and
phase offsets in rad; {doc}`../explanations/pulseq/events-and-blocks`
describes the block and extension structure.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Acquisition and control events

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.make_adc` | Samples, dwell or duration, delay, offsets | ADC event | Acquisition window. |
| {obj}`~pypulseqpp.make_delay` | Duration (s) | Delay event | Minimum block duration. |
| {obj}`~pypulseqpp.make_digital_output_pulse` | Output channel, delay, duration | Trigger event (output) | Digital output pulse. |
| {obj}`~pypulseqpp.make_label` | Label name, `SET` or `INC`, value | Label event | `LABELSET` / `LABELINC` extension. |
| {obj}`~pypulseqpp.make_rf_shim` | Complex weights, one per transmit channel | RF shim extension event | Static per-channel RF weights. |
| {obj}`~pypulseqpp.make_rotation` | Angles (rad), axis-angle, quaternion or matrices | Rotation extension event, or one per matrix | Block gradient rotation. |
| {obj}`~pypulseqpp.make_soft_delay` | Hint, ID, offset, factor, default duration | Soft delay extension event | Console-adjustable block duration. |
| {obj}`~pypulseqpp.make_trigger` | Input channel, delay, duration | Trigger event (input) | Wait for an external signal. |

## Block and event operations

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.align` | Events keyed by `left`, `center`, `right` | Events with delays set | Alignment within a block. |
| {obj}`~pypulseqpp.block_to_events` | Block or events | Tuple of events | Event extraction from a block. |
| {obj}`~pypulseqpp.calc_duration` | Events or block | Duration (s) | Longest event extent. |
| {obj}`~pypulseqpp.rotate` | Gradient events, angle (rad), axis `x`/`y`/`z` | Rotated gradient events | Rotation about a channel axis. |

## Labels and tracing

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.get_supported_labels` | — | Tuple of label names | Supported label identifiers. |
| {obj}`~pypulseqpp.enable_trace` | Stack depth | `None` | Record event and block source locations. |
| {obj}`~pypulseqpp.disable_trace` | — | `None` | Stop recording source locations. |

## Interoperability with PyPulseq

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.interoperating` | Callable taking or returning PyPulseq events | Wrapped callable with event conversion | Compiled events through upstream functions. |
| {obj}`~pypulseqpp.convert` | PyPulseq namespace event | Compiled event; other objects unchanged | Namespace to compiled event. |
| {obj}`~pypulseqpp.as_namespace` | Compiled event | `SimpleNamespace`; other objects unchanged | Compiled event to namespace. |
