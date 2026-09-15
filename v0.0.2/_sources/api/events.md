# Events and blocks

`pypulseqpp`: the events a block plays besides RF and gradients, and the
operations on blocks and events. Every factory returns a compact event that
{meth}`Sequence.add_block` accepts; RF and gradient factories have their own
pages.

```{eval-rst}
.. currentmodule:: pypulseqpp
```

## Acquisition and control events

```{eval-rst}
.. autosummary::
   :toctree: ../generated
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
   :toctree: ../generated
   :nosignatures:

   align
   block_to_events
   calc_duration
   rotate
```

## Labels and tracing

```{eval-rst}
.. autosummary::
   :toctree: ../generated
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
   :toctree: ../generated
   :nosignatures:

   interoperating
   convert
   as_namespace
```
