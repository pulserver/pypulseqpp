# Events and blocks

`pypulseqpp`: the block events other than RF and gradients, and the operations
on blocks and events. Every factory returns a compiled event that
{meth}`Sequence.add_block` accepts; the RF and gradient factories have pages of
their own.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Acquisition and control events

```{eval-rst}
.. autosummary::
   :nosignatures:

   make_adc
   make_delay
   make_digital_output_pulse
   make_label
   make_rf_shim
   make_rotation
   make_soft_delay
   make_trigger
```

## Block and event operations

```{eval-rst}
.. autosummary::
   :nosignatures:

   align
   block_to_events
   calc_duration
   rotate
```

## Labels and tracing

```{eval-rst}
.. autosummary::
   :nosignatures:

   get_supported_labels
   enable_trace
   disable_trace
```

## Interoperability with PyPulseq

Upstream PyPulseq functions take and return plain namespaces.
{func}`interoperating` wraps such a function so that it accepts and returns
compiled events; {func}`convert` and {func}`as_namespace` convert one event
each way.

```{eval-rst}
.. autosummary::
   :nosignatures:

   interoperating
   convert
   as_namespace
```
