# Bloch simulation

```{admonition} TL;DR
:class: tldr

- {class}`~pypulseqpp.Isochromats` integrates the Bloch equation with
  relaxation for a set of isochromats, in the frame rotating at the reference
  frequency. The magnetisation turns clockwise about the field seen from its
  tip, as the moment of a nucleus of positive gyromagnetic ratio precesses: a
  90° pulse along $+x$ takes $+z$ to $+y$, and free precession at a positive
  frequency gives $M_x + iM_y$ a negative phase.
- Free precession under a piecewise-linear gradient is integrated exactly. An
  RF pulse is played in steps of constant $b_1$, each a rotation about the
  step's mean field between two half steps of relaxation.
- A Pulseq RF pulse enters as the complex conjugate of the waveform Pulseq
  defines. A frequency offset and the same phase ramp in the shape are then one
  pulse, and a positive offset excites isochromats precessing at a positive
  frequency.
- Each ADC sample is the sum over the isochromats of the receive sensitivity
  times $M_x + iM_y$, multiplied by $e^{i\theta}$, with $\theta$ the ADC phase
  offset and phase modulation advancing at the ADC frequency offset. A
  gradient echo then samples $\sum_j \rho_j e^{-2\pi i\,\mathbf{k}\cdot\mathbf{r}_j}$
  at the $\mathbf{k}$ that {meth}`~pypulseqpp.Sequence.calculate_kspace`
  reports.
- Diffusion, flow, motion, concomitant fields and the scanner's hardware are
  not modelled.
```

A Bloch simulation computes the magnetisation that a sequence leaves in an
object and the signal it induces in the receive coils. The object is
represented by isochromats: spin packets that share one position, one
precession frequency and one pair of relaxation times. This page states the
equation the engine integrates, how each Pulseq event enters it, and how the
resulting signal relates to the k-space trajectory.

## The Bloch equation in the rotating frame

In the frame rotating at the scanner's reference frequency, an isochromat at
position $\mathbf{r}$ sees the field

$$
\mathbf{b}(t) = \bigl(\operatorname{Re} b_1(t),\ \operatorname{Im} b_1(t),\
\mathbf{G}(t)\cdot\mathbf{r} + \Delta f\bigr),
$$

in Hz: $b_1$ is the transverse RF field, $\mathbf{G}$ the gradient in Hz/m and
$\Delta f$ the isochromat's off-resonance, field inhomogeneity and chemical
shift together. The magnetisation obeys

$$
\frac{d\mathbf{M}}{dt} = 2\pi\,\mathbf{M}\times\mathbf{b}
- \frac{M_x\hat{\mathbf{x}} + M_y\hat{\mathbf{y}}}{T_2}
- \frac{(M_z - M_0)\,\hat{\mathbf{z}}}{T_1},
$$

with $M_0$ the proton density. The cross product in this order is the
precession of a nucleus of positive gyromagnetic ratio: the magnetisation
turns about $\mathbf{b}$ clockwise, seen from the tip of $\mathbf{b}$. A 90°
pulse along $+x$ takes $+z$ to $+y$, and free precession gives the transverse
magnetisation $M_{xy} = M_x + iM_y$ the factor $e^{-2\pi i\,b_z t}$.

{func}`~pypulseqpp.sim_bloch` turns the magnetisation the other way, right-handedly
about $\mathbf{b}$. The two are mirror images under $y \to -y$: the engine's
result for a field $b_1$ is `sim_bloch`'s for $b_1^*$ with $M_y$ negated.

## Free precession

Between RF steps, $b_z = \mathbf{G}(t)\cdot\mathbf{r} + \Delta f$ turns the
magnetisation about $z$ only, and the turns add. Over an interval $\tau$,

$$
M_{xy} \leftarrow M_{xy}\, e^{-2\pi i\,(\mathbf{r}\cdot\mathbf{A} + \Delta f\,\tau)}\,e^{-\tau/T_2},
\qquad
M_z \leftarrow M_z\,e^{-\tau/T_1} + M_0\,(1 - e^{-\tau/T_1}),
$$

where $\mathbf{A} = \int \mathbf{G}\,dt$ is the gradient area over the
interval, in 1/m. A Pulseq gradient is piecewise linear, so $\mathbf{A}$ is
exact from its corners, and so is the update, for any $\tau$. The engine
accumulates $\mathbf{A}$ and $\tau$ and applies them when the magnetisation is
next needed, by an RF pulse, an ADC sample or a read: blocks that hold neither
cost nothing per isochromat.

## RF pulses

During an RF pulse $b_1$ varies, and the field no longer points along $z$. The
engine holds $b_1$ constant over steps of duration $\Delta t$. Step $k$ turns
the magnetisation by $2\pi\,|\mathbf{b}_k|\,\Delta t$ about $\mathbf{b}_k$,
whose $z$ component is the mean over the step,
$\mathbf{r}\cdot\Delta\mathbf{A}_k/\Delta t + \Delta f$. Relaxation over the
step is split into two half steps around the rotation,

$$
\mathbf{M} \leftarrow E(\tfrac{\Delta t}{2})\,R_k\,E(\tfrac{\Delta t}{2})\,\mathbf{M},
$$

a symmetric splitting whose error is of second order in $\Delta t$. The steps
compose into one affine map, $\mathbf{M} \leftarrow A\,\mathbf{M} + \mathbf{c}\,M_0$,
per pulse.

The map depends on the isochromat only through $\Delta f$, $T_1$, $T_2$, the
transmit sensitivities and $\mathbf{r}\cdot\Delta\mathbf{A}_k$. When the steps'
gradient areas all lie along one direction, as they do under a slice-selection
gradient, the last depends on the position along that direction alone.
Isochromats equal in the others and within $10^{-12}$ m along that direction
share one map. For a slice of a phantom with a few tissues, a pulse then costs
a few maps rather than one per isochromat.

## Pulseq events as fields

{meth}`~pypulseqpp.Sequence.simulate` plays each block's events as follows.

Gradients
: The gradients each block plays after its rotation, on the axes the
  isochromats' positions are given along. For a sequence with rotations, these
  are the logical axes.

RF pulses
: Pulseq defines a pulse's complex waveform as its amplitude times its
  magnitude and phase shapes, times a carrier of the phase offset advancing at
  the frequency offset from the pulse's start,
  $w(t) = a\,m(t)\,e^{2\pi i\,p(t)}\,e^{i(\phi + 2\pi f t)}$, with the ppm
  offsets resolved at the system's `gamma` and `B0`. The engine plays
  $b_1 = w^*$. A phase ramp in $p(t)$ and an equal frequency offset are then
  one pulse, as the format defines them, and a positive $f$ excites
  isochromats at a positive $\Delta f$: under a positive slice-selection
  gradient, a slice at a positive position. Samples on the RF raster are held
  over their raster intervals. A pulse with a time shape is joined linearly
  between its samples and held over steps of the RF raster, or of its shortest
  interval where that is shorter.

Parallel transmission
: With transmit sensitivities $S_c(\mathbf{r})$, a pTx pulse's channels add as
  $b_1 = \sum_c S_c\,b_{1,c}$, and a single-channel pulse plays on every
  channel, weighted by the block's RF shim where it has one. Without them, the
  channels are summed and a shim is not applied: every channel has unit,
  in-phase sensitivity, and a pulse turns by the flip angle
  {meth}`~pypulseqpp.Sequence.rf_flip_angles` reports.

ADC samples
: Coil $c$ receives
  $s_c(t) = \sum_j R_{jc}\,M_{xy,j}(t)$, with $R_{jc}$ the receive sensitivity
  of isochromat $j$. Each sample is multiplied by $e^{i\theta}$, where $\theta$
  is the ADC phase offset plus its phase modulation, advancing at its frequency
  offset from the start of the window.

With these conventions, an ADC phase offset equal to the phase offset of the
excitation cancels it, which RF spoiling relies on. Isochromats excited by a
90° pulse along $+x$, without relaxation, give

$$
s(t) = i \sum_j \rho_j\, e^{-2\pi i\,\mathbf{k}(t)\cdot\mathbf{r}_j},
$$

with $\mathbf{k}$ the trajectory {meth}`~pypulseqpp.Sequence.calculate_kspace`
reports, and the inverse discrete Fourier transform of Cartesian samples places
each isochromat at its own position.

## Relation to KomaMRI

KomaMRI turns the magnetisation in the same sense, and plays a pulse's
frequency offset as a frame rotating in the same sense as here. Its Pulseq
reader adds a pulse's phase offset and phase shape to $b_1$ with the opposite
sign, refers the frequency offset's phase to the pulse's centre, and
demodulates a sample by the ADC phase offset alone, with the opposite sign and
without the ADC frequency offset or phase modulation. The two therefore agree,
up to their time discretisation, where every pulse has a real waveform, a
phase offset of 0 or π and no frequency offset, and every ADC has a phase
offset of 0 or π and no frequency offset or phase modulation. Elsewhere the
phase of a pulse or of the demodulation differs between them.

## What is not modelled

The engine computes the signal of an ideal system playing the sequence as
written: no gradient delays, eddy currents, gradient nonlinearity, concomitant
fields or receiver noise. Isochromats neither diffuse, flow nor move, and the
decay of a voxel's signal by intravoxel dephasing appears only where several
isochromats represent the voxel.

## See also

* {doc}`../api/simulation` — the isochromats and the simulation of a
  sequence's blocks.
* {doc}`pulseq/events-and-blocks` — the RF, gradient and ADC events a block
  holds.
* {func}`~pypulseqpp.sim_rf` — the off-resonance profile of one RF pulse.
