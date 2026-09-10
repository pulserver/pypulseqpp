# Events and blocks

```{eval-rst}
.. currentmodule:: pypulseqpp
```

Factories return compact event objects accepted directly by
{meth}`Sequence.add_block`. Control events occupy the sequence timeline or
carry instructions interpreted alongside it; RF and gradient factories are
listed on their own pages.

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
