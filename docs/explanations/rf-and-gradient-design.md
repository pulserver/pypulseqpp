# RF and gradient design

```{admonition} TL;DR
:class: tldr

- Designers return RF and gradient events from a prescription and the system limits.
- `sim_bloch` and `sim_rf` simulate the result without relaxation.
- A designed waveform is a computed shape, not a measured profile on a scanner.
```

An excitation has to select a slice with a given time-bandwidth product,
tolerate the B1 field your coil delivers, and fit inside the RF and gradient
limits. You state the flip angle, duration and limits to a designer, simulate
the event it returns against off-resonance or B1, and place it in a block, as
in the excitation of the {doc}`Tour on slice selection
<../generated/gallery/02-tours/02_slice_selective_excitation>`.

## What pypulseqpp does

:::{container} capabilities

- **Designs Shinnar-Le Roux pulses with five filter types and optional root flipping.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_slr_pulse` with `filter_type=` and `root_flip=`. Tests: *every filter design selects the slice it was designed for*; *root flipping never raises the peak*; *the compiled search picks the pattern an exhaustive numpy search does* (`test_slr_design.py`).

  </details>
- **Designs minimum-phase excitation and records its effective centre.**

  <details><summary>Show code and tests</summary>

  Code: `filter_type="min"` and `center_pos=` of {func}`~pypulseqpp.make_slr_pulse`. Tests: *minimum and maximum phase pulses load opposite ends*; *an slr pulse records its centre at its magnitude peak*; *the rephaser refocuses from the centre the pulse records* (`test_slr_design.py`).

  </details>
- **Designs recursive SLR trains, multiband modulation and spectral-spatial and 2D-selective pulses.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_recursive_slr_pulses`, {func}`~pypulseqpp.make_sms_pulse`, {func}`~pypulseqpp.make_spsp_pulse`, {func}`~pypulseqpp.make_2d_selective_pulse`. Tests: *every pulse of a recursive train excites the same magnetisation*; *a phase schedule lowers the peak of the band sum*; *a spsp pulse rides an alternating gradient that ends balanced* (`test_b1_and_recursive_pulses.py`, `test_slr_design.py`, `test_pypulseq_base_factories.py`).

  </details>
- **Designs adiabatic pulses: hyperbolic secant, WURST, BIR-4, GOIA-WURST and half passages.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_adiabatic_pulse`, {func}`~pypulseqpp.make_half_passages`. Tests: *a bir4 pulse tips by its flip angle whatever the b1*; *a goia wurst pulse inverts its slice and leaves the rest*; *the half passages are exact mirrors* (`test_adiabatic.py`, `test_design_preparation.py`).

  </details>
- **Designs B1-encoding pulses, including adiabatic Bloch-Siegert.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_bloch_siegert_pulse`, {func}`~pypulseqpp.make_b1_selective_pulse`, {func}`~pypulseqpp.make_b1_gslider_pulse`, {func}`~pypulseqpp.make_b1_hadamard_pulse`. Tests: *a bloch siegert phase grows with the square of b1*; *a bloch siegert pulse leaves the magnetisation along z* (`test_b1_and_recursive_pulses.py`).

  </details>
- **Stores a dynamic pTx pulse as one RF event holding every channel in turn.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_ptx_pulse`, {func}`~pypulseqpp.split_ptx_pulse`. Tests: *the time base restarts once per channel*; *a pulse lasts one channel however many it drives*; *the layout survives writing and reading*; *upstream reads the same channels* (`test_ptx.py`).

  </details>
- **Sums channels in flip angle, power and bandwidth, each channel on its own time base.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.test_report`, {func}`~pypulseqpp.calc_rf_power`, {func}`~pypulseqpp.calc_rf_bandwidth`, {func}`~pypulseqpp.calc_rf_center`. Tests: *the report sums channels played in phase*; *the report cancels channels played in antiphase*; *a pulse is as wide as the sum of its channels*; *a pulse is centred on the peak of the root sum square of its channels* (`test_ptx.py`).

  </details>
- **Designs RF shims, spokes and 2D-selective pTx pulses from B1+ maps.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.calc_rf_shim`, {func}`~pypulseqpp.make_rf_shim`, {func}`~pypulseqpp.make_spokes_pulse`. Tests: *a shim delivers the magnitude asked for*; *spokes flatten a b1 that one spoke cannot*; *a spokes pulse plays what its design predicts*; *two blocks shimmed alike share one row* (`test_ptx_design.py`, `test_rf_shim.py`).

  </details>
- **Simulates a pulse without relaxation, across positions or per-position fields.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.sim_bloch`, {func}`~pypulseqpp.sim_rf`. Tests: *sim bloch is the step by step rotation*; *each position can see its own field*; *sim rf is the quaternion composition simrf performs* (`test_sim_bloch.py`).

  </details>
- **Times a k-space path into the shortest gradient within the amplitude and slew limits.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.traj_to_grad`, {func}`~pypulseqpp.calc_spiral_trajectory`, {func}`~pypulseqpp.calc_rosette_trajectory`, {func}`~pypulseqpp.make_arbitrary_grad`. Tests: *the time optimal gradient obeys both limits everywhere*; *resampling the same path does not change the gradient*; *differentiating reproduces upstream exactly* (`test_pypulseq_base_factories.py`).

  </details>

:::

## What PyPulseq does

- **The basic factories, kept with their signatures.**
  {func}`~pypulseqpp.make_sinc_pulse`, {func}`~pypulseqpp.make_gauss_pulse`,
  {func}`~pypulseqpp.make_block_pulse`, {func}`~pypulseqpp.make_trapezoid`
  and {func}`~pypulseqpp.make_extended_trapezoid` are upstream's designs,
  re-exported and converted to compiled events.
- **The adiabatic sweeps `hypsec` and `wurst`.** `make_adiabatic_pulse` calls
  PyPulseq's factory for those two.
- **Its time-optimal solver is not used.** With `time_optimal=False`,
  `traj_to_grad` differentiates the path as upstream does.

## How it works

```{figure} ../generated/figures/designed_pulse_and_gradient.png
A linear-phase and a minimum-phase SLR excitation of one time-bandwidth
product, the transverse magnetisation `sim_bloch` gives for each across
off-resonance, and the gradient magnitude `traj_to_grad` assigns to a spiral
against `max_grad`.
```

### Designs end in a played event

A designer returns an RF event, and with `return_gz` the selection gradient
and its rephaser, on the system's rasters. The rephaser refocuses from the
effective centre the event records, which a minimum-phase pulse places at its
end, so `center_pos` sets both the rephaser area and the instant an echo time
is measured from.

### A pTx pulse is one event

The channels of a dynamic pTx pulse lie one after another in a single RF
event, so a file and an interpreter that know nothing of pTx still read one
pulse of one channel's duration. The channel count is read back from the
samples, not stored.

### The path fixes the geometry, the solver the timing

A k-space path carries no timing. `traj_to_grad` resamples it and assigns the
time at which each point is reached, so the vector gradient and slew
magnitudes reach their limits wherever the geometry allows, as in the figure, where the
spiral ramps at the slew limit and then holds `max_grad`.

The full rules, units and the pTx layout are in
{doc}`../developer-guide/internals/rf-and-gradient-design`.

## See it run

- {doc}`../generated/gallery/02-tours/02_slice_selective_excitation`: a
  designed slice profile against the simulated one.
- {doc}`../generated/gallery/02-tours/09_minimum_phase_excitation`: a
  minimum-phase excitation module against the linear-phase one.
- {doc}`../generated/gallery/02-tours/06_spiral`: a spiral readout designed
  at several slew limits and sampling rates.
- {doc}`../api/rf`, {doc}`../api/gradients` and {doc}`../api/trajectories`:
  the designers and their units.
