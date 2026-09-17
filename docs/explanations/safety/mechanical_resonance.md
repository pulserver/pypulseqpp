# Mechanical resonance

A gradient coil in the static field experiences a Lorentz force proportional to
its current, and the assembly it is mounted in has mechanical modes with narrow
resonances. A gradient waveform whose spectrum contains sustained power at a
mode's frequency excites that mode, producing acoustic output far above what
the same amplitude produces elsewhere in the spectrum, together with mechanical
stress on the assembly. Vendors publish the frequency ranges the coil must not
be driven in as **forbidden bands**, and
{func}`~pypulseqpp.safety.check_mech_resonance` compares the sequence's gradient
spectrum with them.

## Sustained drive and the analysis window

What excites a resonance is a **sustained** drive near the mode's frequency, not
a single transition. A lightly damped mode reaches its steady-state amplitude
over many cycles, so a waveform that crosses a band for one repetition and
exits it deposits far less energy than one that remains within it for a second.
A single transform of the whole sequence cannot express that distinction: it
reports the total power at each frequency without saying whether it arrived all
at once or was spread over the scan.

The check therefore slides a window along each physical axis and transforms each
window separately. The window length is the time scale over which a drive counts
as sustained, and it is a parameter, `window_width`, rather than a constant: a
mode with a high quality factor is driven by a shorter burst than a heavily
damped one.

Consecutive windows overlap by half the window length by default, so a burst
falling on a window boundary is still contained whole in one window. `stride`
sets the overlap.

## Spectral amplitude normalization

Each window is mean-subtracted, tapered with a Hann window and zero-padded to
`frequency_oversampling` times its length before a real FFT. The amplitude
reported for bin $k$ is

$$
A_k = \frac{2\,|X_k|}{\sum_n w_n},
$$

where $w$ is the taper. The normalization makes the reading
interpretable as a gradient amplitude: a sustained sinusoid of amplitude $A$ at
a bin frequency reads $A$, so a threshold stated in mT/m is compared with a
quantity in mT/m rather than with a spectral density.

Mean subtraction removes the constant component of the window, which is not a
drive at any resonance. Tapering keeps a strong low-frequency component
from leaking across the whole spectrum and producing a reading inside a band
that no gradient in the window put there. Zero-padding does not add
information; it interpolates the spectrum so that a line falling between bins
is read at close to its true amplitude rather than split between neighbours.

## Spectral content of echo-planar and spiral readouts

An alternating readout train is periodic. A train of trapezoids of alternating
polarity at echo spacing $\Delta t$ has its fundamental at

$$
f = \frac{1}{2\,\Delta t},
$$

with harmonics at odd multiples of it, and the amplitude at the fundamental is
a fixed fraction of the plateau amplitude — between $8/\pi^2$ for a triangular
waveform and $4/\pi$ for a square one. An echo-planar train at a 500 µs echo
spacing therefore concentrates most of its gradient power in a narrow line at
1 kHz, and sustains it for the length of the train. A band either contains that
line or does not. Echo spacing is therefore the parameter a vendor's table
constrains, and some vendors publish their bands as forbidden **echo-spacing**
ranges rather than as frequencies.

```{figure} ../../generated/figures/gradient_spectra.png
The readout-axis spectrum of a 40 ms window at the middle of two sequences, both
64 x 64. The spoiled gradient echo spreads its power below a few hundred hertz;
the single-shot echo-planar train concentrates it in one line at the reciprocal
of twice its echo spacing, several times higher than anything the gradient echo
reaches.
```

A spiral readout sweeps its instantaneous frequency as the trajectory winds out,
so it spreads its power over a range instead of concentrating it. The window has to be short
enough to resolve the interval the sweep spends inside a band, and long enough
that such an interval is not read as a sustained drive.

A conventional Cartesian gradient echo puts most of its gradient power below a
few hundred hertz and is at zero amplitude for most of each repetition, so it
rarely reaches a band at all.

## Thresholds

A band is an axis, an inclusive frequency range and a tolerance, as
{class}`~pypulseqpp.safety.ForbiddenBand`. An axis of `None` guards all three.
Where the table states a tolerance, it is the largest amplitude allowed inside
the band; where it states none, the check uses `min_threshold`. A band is
violated when any window exceeds its threshold on any axis the band guards, and
the report states the worst window of every band whether or not it violates.

{func}`~pypulseqpp.safety.read_forbidden_bands` reads two vendor forms. A
Siemens `.asc` hardware file states acoustic resonances as a centre frequency
and a bandwidth, with no axis and no tolerance, so every band it yields guards
every axis at tolerance zero. A GE `epiesp.dat` table states, per axis,
forbidden echo-spacing ranges with a plateau amplitude; the range maps to the
frequencies $1/(2\,\mathrm{ESP})$ of the alternating train it describes.

## Scope and limitations

The criterion is a spectral one applied to the commanded waveform. It does not
model the coil's transfer function, the acoustic output of the assembly, or the
scanner's own predownload gate, and a sequence that passes it is not thereby
established as quiet or as within any acoustic-noise regulation. What it
establishes is that the sequence does not sustain gradient amplitude inside a
frequency range the table identifies as forbidden.

## See also

* {func}`~pypulseqpp.safety.check_mech_resonance`,
  {func}`~pypulseqpp.safety.read_forbidden_bands` and
  {class}`~pypulseqpp.safety.ForbiddenBand` — the calls.
* {doc}`../../guides/checking-constraints` — running the check over a
  sequence and reading its report.
