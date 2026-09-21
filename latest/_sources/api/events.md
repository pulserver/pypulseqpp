# Events and blocks

`pypulseqpp`: the block events other than RF and gradients, and the operations
on blocks and events. Every factory returns a compiled event that
{meth}`Sequence.add_block` accepts; the RF and gradient factories have pages of
their own.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Acquisition and control events

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.make_adc` | Create an ADC readout event. |
| {obj}`~pypulseqpp.make_delay` | Create a delay event. |
| {obj}`~pypulseqpp.make_digital_output_pulse` | Create a digital-output pulse on a supported channel. |
| {obj}`~pypulseqpp.make_label` | Create a label event. |
| {obj}`~pypulseqpp.make_rf_shim` | Create complex per-transmit-channel weights for a block's RF envelope. |
| {obj}`~pypulseqpp.make_rotation` | Create a rotation extension event for a block's gradients. |
| {obj}`~pypulseqpp.make_soft_delay` | Create a soft delay extension event for dynamic timing adjustment. |
| {obj}`~pypulseqpp.make_trigger` | Create an external-input trigger event. |

## Block and event operations

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.align` | Align event start, centre or end times within a block. |
| {obj}`~pypulseqpp.block_to_events` | Split a block into its events, or pass events through unchanged. |
| {obj}`~pypulseqpp.calc_duration` | Calculate the duration of an event or block. |
| {obj}`~pypulseqpp.rotate` | Rotate gradient events about a logical axis. |

## Labels and tracing

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.get_supported_labels` | Return the supported label identifiers. |
| {obj}`~pypulseqpp.enable_trace` | Record source locations when events and blocks are created. |
| {obj}`~pypulseqpp.disable_trace` | Stop recording event and block source locations. |

## Interoperability with PyPulseq

Upstream PyPulseq functions take and return plain namespaces.
{func}`interoperating` wraps such a function so that it accepts and returns
compiled events; {func}`convert` and {func}`as_namespace` convert one event
each way.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.interoperating` | Wrap a callable with recursive event conversion. |
| {obj}`~pypulseqpp.convert` | Convert a PyPulseq event to a compiled event; return other objects unchanged. |
| {obj}`~pypulseqpp.as_namespace` | Convert a compiled event to a PyPulseq-compatible SimpleNamespace. |
