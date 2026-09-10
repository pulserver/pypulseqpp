# Sampling and ordering

```{eval-rst}
.. currentmodule:: pypulseqpp
```

Sampling functions choose which Cartesian positions are acquired. Ordering
functions arrange those positions into shots or echo trains; schedules supply
the RF phase or refocusing angle paired with each repetition.

## Sampling masks

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_uniform_mask
   make_random_mask
   make_poisson_disc_mask
   make_caipirinha_mask
   calc_sampled_lines
   calc_calibration_lines
   calc_sampled_pairs
```

## View ordering

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_linear_order
   make_centric_order
   make_radial_order
   make_radial_adaptive_order
   make_shuffling_order
   calc_traversal_order
   calc_epi_order
```

## RF schedules

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   make_phase_cycling_schedule
   make_rf_spoiling_schedule
   make_traps_schedule
```
