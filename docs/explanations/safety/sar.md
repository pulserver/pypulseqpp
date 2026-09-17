# Specific absorption rate

RF transmission deposits energy in tissue. The regulated quantity is the
specific absorption rate, in watts per kilogram, averaged over a stated mass and
a stated time window: a **global** value over the exposed volume and a **local**
value over any ten grams of tissue, both bounded by IEC 60601-2-33 according to
the operating mode. {func}`~pypulseqpp.safety.check_sar` estimates both from a
virtual-observation-point model and compares them with limits.

## Virtual observation points

Local SAR is a field in space. For a transmit array with $N_c$ channels driven
by a phasor vector $\mathbf{v}$, the SAR at a position $\mathbf{r}$ is a
Hermitian quadratic form,

$$
\mathrm{SAR}(\mathbf{r}) = \mathbf{v}^{\mathsf H} \, Q(\mathbf{r}) \, \mathbf{v},
$$

with one matrix $Q$ per position, obtained from an electromagnetic simulation on
a body model. Evaluating the peak over every voxel for every candidate drive is
not affordable online, and is not necessary: the matrices can be compressed into
a small set of **virtual observation points**, each an upper bound over a cluster
of positions, such that the largest value over the set bounds the largest value
over the body (Eichfelder and Gebhardt, Magn Reson Med 2011,
doi:10.1002/mrm.22927).

{class}`~pypulseqpp.safety.VopModel` stores that set as an $(N, N_c, N_c)$ stack
of Hermitian matrices, in W/kg per unit channel drive squared, and optionally a
global matrix evaluated the same way over the whole exposed volume. The model is
read from a `.mat` or `.npz` file by {func}`~pypulseqpp.safety.read_vops`.
{func}`~pypulseqpp.safety.example_vops` returns a synthetic eight-channel model
of a loop array around a uniform cylinder, together with the drive calibration
and circularly polarized shim that go with it. It models no tissue, no coil
coupling and no conservative field, so its numbers are plausible in scale and
nothing more, and it is for demonstrations only.

## Drive calibration from RF amplitude

A Pulseq RF event states an amplitude in Hz, which is a statement about the
$B_1^+$ field it produces, not about the voltage or current that produces it.
Converting between the two is a property of the transmit chain and the loading,
and it is the calibration the check has to be given: `drive_per_hz`, one value
or one per channel, in whatever unit the VOPs were computed in.

A pulse then drives channel $c$ with $\texttt{drive}_c \, s_c \, b_c(t)$, where
$b$ is the waveform in Hz, resampled every microsecond as
{func}`~pypulseqpp.calc_rf_power` does, and $s$ is the block's RF shim, or
`default_shim` where a single-channel pulse defines none. A single-channel pulse
is treated as the same waveform on every channel, weighted by the shim.

## Averaging window

SAR is defined per unit time, so the check needs an interval to average over.
It uses the repetitions the sequence's own block definitions repeat with: the
blocks before the first full repetition, each repetition, and any blocks after
the last, or the whole sequence when it does not repeat. The largest local value
over those windows, and the largest global value, are the verdict.

A repetition rather than a fixed six-minute window is the right unit here
because a scan's repetitions are what a longer average is built from, and
because the worst repetition bounds every window a longer average could
contain. The report states every window's values, so a caller averaging over a
regulatory interval has the per-repetition energies required to do so.

The window is the sequence's own structure, so a sequence written with dummy
repetitions or an unusual prologue reports those as their own windows rather
than folding them into the steady-state ones.

## Relative comparison against a reference sequence

Absolute SAR from a stated model is only as good as the calibration behind it.
Comparing two sequences under the *same* model and calibration is much more
robust, because the scale of `drive_per_hz` and of the VOPs cancels in the
ratio.

`reference` takes a second sequence, usually a CP-mode free induction decay, or
the report of an earlier call. The report then states
`sar_ratio`, the largest ratio over windows and VOPs of a VOP's SAR to the same
VOP's in the reference, and `energy_ratio`, that ratio weighted by the window
durations. With a reference lasting its own minimum repetition time,
`energy_ratio` scales that minimum to this sequence's repetition at the energy
each repetition deposits, which is the form a protocol's SAR headroom is
usually stated in. Relative channel gains do not cancel, so the reference has to
be evaluated in the same calibration.

## Scope and limitations

The estimate covers the RF energy the sequence's own waveforms deposit in a
model of one subject. It does not cover RF coil heating, gradient heating, the
scanner's own predownload assessment or its transmit monitor, and it does not
make a statement about any particular patient. What it establishes is whether
the sequence, under a stated model and calibration, is within a stated limit.

## Related pages

* {func}`~pypulseqpp.safety.check_sar`,
  {func}`~pypulseqpp.safety.read_vops` and
  {func}`~pypulseqpp.safety.example_vops` — the calls.
* {func}`~pypulseqpp.calc_rf_power` and
  {meth}`~pypulseqpp.Sequence.calc_rf_power` — the RF power the check
  integrates, in Pulseq's Hz units.
* {doc}`../../guides/checking-constraints` — running the check over a
  sequence and reading its report.
