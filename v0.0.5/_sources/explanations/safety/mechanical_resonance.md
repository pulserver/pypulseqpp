# Mechanical resonance

The Lorentz force on a gradient coil in the static field is proportional to its
current, and the gradient assembly has narrowly resonant mechanical modes.
Sustained gradient drive at a mode's frequency produces acoustic output and
mechanical stress well above those at other frequencies.[^hedeen] Vendors state
the frequency ranges that must not be driven as **forbidden bands**.
{func}`~pypulseqpp.safety.check_mech_resonance` compares a windowed amplitude
spectrum of the physical-axis gradient waveforms with those bands.

## Windowed amplitude spectrum

Each physical axis is sampled at the centres of the sequence's own gradient
raster, after each block's rotation and then the `rotation` argument. A window
of `window_width` (40 ms by default) starts every `stride` (half the window by
default); the last window is zero-filled to the end of the sequence. Each window
is mean-subtracted, multiplied by a Hann taper $w$, zero-padded to
`frequency_oversampling` times its length and transformed with a real FFT. The
amplitude of bin $k$ is

$$
A_k = \frac{2\,|X_k|}{\sum_n w_n},
$$

so a sustained sinusoid of amplitude $A$ at a bin frequency reads $A$, in mT/m.
The window length sets the time scale over which drive counts as sustained; a
single transform of the whole sequence would not distinguish one brief crossing
of a band from drive held inside it. Mean subtraction removes the constant
component, the taper limits leakage of low-frequency content into distant bins,
and zero-padding interpolates the spectrum without adding information.
{func}`~pypulseqpp.safety.mech_resonance_spectrum` returns the spectrum of one
window from the same pass.

## Spectra of echo-planar and spiral readouts

A train of trapezoids of alternating polarity at echo spacing $\Delta t$ is
periodic, with fundamental

$$
f = \frac{1}{2\,\Delta t}
$$

and harmonics at odd multiples. The fundamental's amplitude lies between
$8/\pi^2$ (triangular) and $4/\pi$ (square) of the plateau amplitude; a 500 µs
echo spacing places it at 1 kHz for the length of the train.

```{figure} ../../generated/figures/gradient_spectra.png
The readout-axis spectrum of a 40 ms window at the middle of two sequences, both
64 x 64. The spoiled gradient echo spreads its power below a few hundred hertz;
the single-shot echo-planar train concentrates it in one line at the reciprocal
of twice its echo spacing, several times higher than anything the gradient echo
reaches.
```

Echo spacing is therefore the parameter a forbidden band constrains for an
echo-planar readout. A spiral readout sweeps its instantaneous frequency and
spreads its power over a range; the window length determines whether the
interval the sweep spends inside a band is resolved.

## Bands and thresholds

A {class}`~pypulseqpp.safety.ForbiddenBand` is an axis (`None` for all three),
an inclusive frequency range in Hz and a tolerance in mT/m. The threshold is the
tolerance where it is positive and `min_threshold` (10 mT/m by default)
otherwise. A band is violated by each window whose largest amplitude on a bin
inside the band exceeds the threshold on any axis the band guards; a band
narrower than one bin is read at the bin nearest its centre. The report states
each band's worst window whether or not it violates.

| Source read by {func}`~pypulseqpp.safety.read_forbidden_bands` | Band | Axis | Tolerance |
| --- | --- | --- | --- |
| Siemens `.asc` acoustic resonances | centre ± bandwidth / 2 | all | 0, so `min_threshold` applies |
| GE `epiesp.dat` forbidden echo-spacing ranges | $1/(2\,\mathrm{ESP})$ over the range | x, y or z, by table section | plateau amplitude, G/cm converted to mT/m |

The `epiesp.dat` tolerance is a plateau amplitude and is compared unscaled with
the spectral amplitude, which for an alternating train is between $8/\pi^2$ and
$4/\pi$ of the plateau.

## Limitations

The criterion is spectral and is applied to the commanded waveform. It does not
model the coil's transfer function, the acoustic output of the assembly or the
scanner's own predownload assessment, and a passing result does not establish
that a sequence is quiet or within any acoustic-noise regulation. It states
that no window's gradient amplitude spectrum exceeds a threshold inside a band
of the supplied table.

## See also

* {func}`~pypulseqpp.safety.check_mech_resonance`,
  {func}`~pypulseqpp.safety.mech_resonance_spectrum`,
  {func}`~pypulseqpp.safety.read_forbidden_bands` and
  {class}`~pypulseqpp.safety.ForbiddenBand` — the calls.
* {doc}`../../examples/checks` — running the check over a
  sequence and reading its report.

## References

[^hedeen]: Hedeen RA, Edelstein WA. Characterization and prediction of gradient acoustic noise in MR imagers. *Magnetic Resonance in Medicine*. 1997;37(1):7–10. [doi:10.1002/mrm.1910370103](https://doi.org/10.1002/mrm.1910370103).
