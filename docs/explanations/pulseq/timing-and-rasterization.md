# Timing and rasterization

A sequencer starts and stops events on a discrete time grid. A time that is not
an integer multiple of the grid period cannot be addressed, so every event time
in a `.seq` file is quantized before it is written. This page states which grid
applies to which time, what follows from the grids interacting, and what
{meth}`~pypulseqpp.Sequence.check_timing` establishes.

## The four rasters

`[DEFINITIONS]` declares four raster periods, and each class of time is
quantized to one of them:

| Raster | Quantizes |
|---|---|
| RF raster | RF waveform sample spacing, RF event delays and centre times |
| Gradient raster | arbitrary-gradient sample spacing, gradient delays, trapezoid rise, flat and fall times |
| ADC raster | ADC dwell time and the ADC event delay |
| Block-duration raster | the duration of every block |

{class}`~pypulseqpp.Opts` holds all four. The defaults are 2 µs for the RF and
ADC rasters and 20 µs for the gradient and block-duration rasters, but they are
properties of the system the sequence is designed against, and a file records
the values it was written with.

{func}`~pypulseqpp.round_to_raster` and {func}`~pypulseqpp.ceil_to_raster`
quantize a single interval. Rounding up rather than to nearest is the usual
choice for a delay that must not fall below a computed minimum.

## Block duration against event extent

The duration recorded for a block is independent of the extent of its events,
subject to being at least as long. The difference is dead time on every
channel, which is how an echo time or a repetition time is realized.

Two consequences follow. A block longer than its events is legal and silent: no
check reports it, because it is a delay. A gradient that ends at a nonzero
amplitude before its block ends is not silent, because the amplitude over the
remaining interval is undefined; `check_timing` reports it as
`GRADIENT_END_NONZERO`.

## Coupling between the ADC and gradient rasters

An acquisition window is `num_samples * dwell` long. The dwell is quantized to
the ADC raster, and the readout gradient's flat top is quantized to the
gradient raster. When the window has to coincide with the flat top — which it
does whenever the samples must be taken at constant gradient amplitude — both
conditions apply at once, and they constrain the achievable receiver bandwidth.

Write $a$ for the ADC raster, $g$ for the gradient raster and
$r = g / a$, an integer for any realistic pair. Admissible dwell times are the
multiples $d = k a$ for which $N d$ is a multiple of $g$, and the smallest is

$$
d_{\min} = \frac{a\,r}{\gcd(N, r)} .
$$

The sample count therefore participates in setting the maximum receiver
bandwidth, alongside the rasters themselves. At the default rasters
($a = 2\ \mu\mathrm{s}$, $g = 20\ \mu\mathrm{s}$, $r = 10$), a readout of 128
samples has $\gcd(128, 10) = 2$ and admits no dwell shorter than 10 µs, a
receiver bandwidth of 100 kHz; a readout of 100 samples has $\gcd(100, 10) =
10$ and admits 2 µs, a bandwidth of 500 kHz. A request for 250 kHz is met in
the second case and not in the first, and nothing about the first prescription
is otherwise unusual.

{func}`~pypulseqpp.calc_adc_timing` performs this search and returns the dwell
together with the acquisition duration, so the achieved bandwidth is
`1 / dwell` and may be lower than the one requested. The readout modules report
what they achieved as `bandwidth_hz` rather than echoing the request.

## Dead times and ringdown

Raster addressability is a separate question from what the transmit and receive
chains can do. Three intervals in {class}`~pypulseqpp.Opts` bound the latter:
`rf_dead_time` before a pulse, `rf_ringdown_time` after it, and `adc_dead_time`
after an acquisition window. They are not quantization constraints, and a
sequence can satisfy every raster and still violate them.

`check_timing` reports both classes. Raster violations appear as `RASTER`;
chain violations appear as `RF_DEAD_TIME`, `RF_RINGDOWN_TIME`, `ADC_DEAD_TIME`
and `POST_ADC_DEAD_TIME`. The same pass also reports `BLOCK_DURATION_MISMATCH`,
`NEGATIVE_DELAY`, `GRADIENT_START_DELAY`, `GRADIENT_END_NONZERO`, the
soft-delay consistency conditions, and `ADC_SAMPLES_DIVISOR` where a vendor
requires the sample count to be divisible by a fixed factor.

A sequence whose gradient waveforms are within every amplitude and slew limit
can still be unplayable because one delay is off the raster. The constraint
checks of {doc}`../safety/index` and `check_timing` answer different questions
and are separate calls.

## Related pages

* {doc}`events-and-blocks` — what a block contains.
* {doc}`../../api/timing` — the quantization and ADC timing helpers.
* {doc}`../../guides/checking-constraints` — running the checks over a finished
  sequence.
