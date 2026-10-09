# Specific absorption rate

```{admonition} TL;DR
:class: tldr

- {func}`~pypulseqpp.safety.check_sar` computes time-averaged local and global
  SAR in W/kg from a virtual-observation-point model and compares them with
  `local_limit` and `global_limit`, by default 10 W/kg and 3.2 W/kg, the IEC
  60601-2-33 normal-mode head values. The check does not use the gradient
  system limits.
- Channel $c$ is driven with $v_c(t) = d_c\,s_c\,b_c(t)$, the RF waveform in Hz
  scaled by `drive_per_hz` and the block's RF shim. Local SAR in a window $W$
  is the largest time-averaged quadratic form over the VOPs times the model's
  safety factor, $M \max_k \mathrm{SAR}_k(W)$.
- The averaging windows are the repetitions detected from the block
  definitions, reported as `tr_size` blocks, or the whole sequence when its
  blocks do not divide into repetitions. The check does not aggregate the
  per-window values over a regulatory averaging interval such as the 6-minute
  interval of IEC 60601-2-33.
- With `reference`, the report adds `local_to_head`, the sequence's local SAR
  over the reference's global SAR, and `global_sar_ratio`, global SAR over the
  reference's body model by body model, each with an energy form. At the
  drive scale where the reference meets the global limit, the two ratios give
  the sequence's local and global SAR in units of that limit. The scale
  of `drive_per_hz` and of the matrices cancels; relative channel gains and
  the safety factor do not.
- A `True` result states only that the computed window-averaged SAR values do
  not exceed the supplied limits under the stated VOP model and drive
  calibration. {func}`~pypulseqpp.safety.example_vops` is a synthetic model for
  demonstration only.
```

RF transmission deposits energy in tissue. The specific absorption rate (SAR,
W/kg) is regulated as a **global** value over the exposed mass and a **local**
value over 10 g of tissue, each averaged over a stated time and bounded by
IEC 60601-2-33 according to the operating mode.
{func}`~pypulseqpp.safety.check_sar` computes time-averaged local and global
SAR from a virtual-observation-point model and compares them with the
`local_limit` and `global_limit` arguments. The check does not use the
gradient system limits.

## Virtual observation points

For a transmit array with $N_c$ channels driven by a phasor vector
$\mathbf{v}$, local SAR at position $\mathbf{r}$ is a Hermitian quadratic form,

$$
\mathrm{SAR}(\mathbf{r}) = \mathbf{v}^{\mathsf H} \, Q(\mathbf{r}) \, \mathbf{v},
$$

with one matrix $Q$ per position from an electromagnetic simulation on a body
model. Virtual observation points (VOPs) compress these matrices into a small
set $\{Q_k\}$ whose largest value bounds the largest value over the body
model:[^eichfelder] each $Q_k$ dominates a cluster of the averaged matrices, so
for every drive

$$
\max_{\mathbf{r}} \mathrm{SAR}(\mathbf{r})
\;\le\; \max_k \mathbf{v}^{\mathsf H} Q_k \mathbf{v}.
$$

{class}`~pypulseqpp.safety.VopModel` holds the $(N, N_c, N_c)$ VOP stack, in
W/kg per unit channel drive squared, optional global matrices, one or one per
body model, and the file's metadata, whose
`safety_factor` $M \ge 1$ multiplies every local SAR for what the model does
not carry, such as positioning, anatomy outside its population and the coil
model's error against measurement. {func}`~pypulseqpp.safety.read_vops` reads
them from a `.mat` or `.npz` file, MARIE's point-first stack and MATLAB's
channel-first one alike, including the population file that mariepy's
`vop.write` stores.
{func}`~pypulseqpp.safety.example_vops` returns a synthetic eight-channel model
of a loop array around a uniform cylinder, with no tissue, coil coupling or
conservative field, for demonstration only.

## Channel drive and time average

An RF event states its amplitude in Hz of $B_1^+$. The conversion to channel
drive is a property of the transmit chain and loading, and is supplied as
`drive_per_hz`, one value or one per channel, in the drive unit of the VOPs.
Channel $c$ is driven with

$$
v_c(t) = d_c \, s_c \, b_c(t),
$$

where $d_c$ is `drive_per_hz`, $b_c$ the RF waveform in Hz resampled every
microsecond as {func}`~pypulseqpp.calc_rf_power` does, and $s_c$ the block's
RF shim, or `default_shim` for a single-channel pulse without one. A
single-channel pulse is played as the same waveform on every channel. For each
averaging window $W$ of duration $T_W$,

$$
\mathrm{SAR}_k(W) = \frac{1}{T_W} \int_W \mathbf{v}(t)^{\mathsf H} Q_k \, \mathbf{v}(t)\,\mathrm{d}t,
\qquad
\mathrm{SAR}_{\mathrm{local}}(W) = M \max_k \mathrm{SAR}_k(W),
$$

and global SAR is the same integral with the global matrix, or the largest
over the body models where the file carries one matrix per model.

## Averaging windows

`check_sar` evaluates RF energy over the repetitions
{meth}`~pypulseqpp.Sequence.repetition` detects from the sequence's block
definitions, reported as `tr_size` blocks: consecutive windows of `tr_size`
blocks from the first block, or the whole sequence as one window when its
blocks do not divide into repetitions. A `TRSize` definition the sequence
records is used when the blocks repeat with it.
The result is `True` when every window's local SAR is at most `local_limit`
and, with a global matrix, every window's global SAR is at most `global_limit`.
The defaults, 10 W/kg and 3.2 W/kg, are the IEC 60601-2-33 normal-mode head
values.

The report states every window's first and last block, duration, local SAR,
VOP index, global SAR and the body model that global SAR belongs to. These per-window quantities may subsequently be
aggregated over a regulatory averaging interval, such as the 6-minute interval
of IEC 60601-2-33; the check itself does not perform that aggregation.

## Comparison with a reference sequence

A scanner controls the SAR of a pulse it knows, such as a hard pulse in the
coil's circularly polarised mode, by its own prediction of that pulse's global
SAR: at its shortest repetition time, the reference deposits at most the
global limit $L_G$. That fixes the drive scale of the model without a power
calibration, and `reference` compares a sequence with such a pulse at that
scale. Played under the same model, drive and default shim, the reference
gives its global SAR $\mathrm{SAR}^{\mathrm{ref}}_{G_b}$ in each body model,
in its window of largest global SAR, of duration $T^{\mathrm{ref}}$. The report
adds

$$
r_{\mathrm{local}} = \max_{W} \frac{\mathrm{SAR}_{\mathrm{local}}(W)}{\min_b \mathrm{SAR}^{\mathrm{ref}}_{G_b}},
\qquad
r_{\mathrm{global}} = \max_{W,b} \frac{\mathrm{SAR}_{G_b}(W)}{\mathrm{SAR}^{\mathrm{ref}}_{G_b}},
$$

as `local_to_head` and `global_sar_ratio`, and the same with each window's term
times $T_W / T^{\mathrm{ref}}$ as `local_to_head_energy` and
`global_energy_ratio`; `windows.local_to_head` and `windows.global_ratio` hold
each window's term. The reference reaches the global limit at the largest drive
in the body model where it deposits least, so the smallest reference SAR over
the models sets the local term. The global matrices are not compressed, and a
subject's global SAR is that of one body, which the largest over all bodies on
each side does not bound, so the global term is taken body model by body model.

At the drive scale that holds the reference at $L_G$, a window's peak local SAR
is at most $r_{\mathrm{local}} L_G$, safety factor included. For a reference
lasting its shortest repetition time, the sequence meets both limits when each
of its repetitions lasts at least

$$
T^{\mathrm{ref}} \max\!\left(r^{E}_{\mathrm{global}},\; r^{E}_{\mathrm{local}} \, \frac{L_G}{L_L}\right),
$$

with $L_L$ the local limit and $r^{E}$ the energy forms. The bound rests on the
scanner's prediction for the reference not falling below the reference's true
global SAR, and on the VOPs and the safety factor bounding local SAR; it uses
no local SAR of the reference. The scale of `drive_per_hz` and of the matrices
cancels in every ratio; relative channel gains do not, and neither does the
safety factor, which multiplies the sequence's side only. The report of an
earlier call on the reference can stand for it.

## Limitations

The estimate covers the RF energy of the sequence's own waveforms in a stated
VOP model and drive calibration. It does not cover RF coil heating, gradient
heating, the scanner's own assessment before a scan or transmit monitoring, and it
makes no statement about a particular subject. A `True` result states only that
the computed window-averaged SAR values do not exceed the supplied limits under
that model and calibration.

## See also

* {func}`~pypulseqpp.safety.check_sar`,
  {func}`~pypulseqpp.safety.read_vops` and
  {func}`~pypulseqpp.safety.example_vops` — the calls.
* {func}`~pypulseqpp.calc_rf_power` and
  {meth}`~pypulseqpp.Sequence.calc_rf_power` — RF power in Pulseq's Hz units.
* {doc}`/generated/gallery/01-course/07_hardware_and_safety_checks` — running the check over a
  sequence and reading its report.

## References

[^eichfelder]: Eichfelder G, Gebhardt M. Local specific absorption rate control for parallel transmission by virtual observation points. *Magnetic Resonance in Medicine*. 2011;66(5):1468–1476. [doi:10.1002/mrm.22927](https://doi.org/10.1002/mrm.22927).
