"""Reference sequences, built with the toolbox pypulseqpp replaces.

Each builder returns an upstream :class:`pypulseq.Sequence`. They are what the
parity tests compare against: upstream writes the file, pypulseqpp writes the
file, and the two must agree byte for byte.

Between them they reach every library section a `.seq` file has: the three RF
pulse shapes, trapezoids on all three axes, arbitrary gradients, ADC events,
both label kinds, soft delays, and the extension chain that carries the last
two.
"""

from __future__ import annotations

import math

import numpy as np
import pypulseq as pp
from pypulseq import Sequence


def gauss_pulses() -> Sequence:
    """Gaussian pulses spanning flip angle, duration, offsets and apodization."""
    seq = Sequence()
    for pulse in (
        pp.make_gauss_pulse(flip_angle=1, use="excitation"),
        pp.make_gauss_pulse(flip_angle=1, delay=1e-3, use="excitation"),
        pp.make_gauss_pulse(flip_angle=math.pi / 2, use="excitation"),
        pp.make_gauss_pulse(flip_angle=math.pi / 2, duration=1e-3, use="excitation"),
        pp.make_gauss_pulse(
            flip_angle=math.pi / 2,
            duration=2e-3,
            phase_offset=math.pi / 2,
            use="excitation",
        ),
        pp.make_gauss_pulse(
            flip_angle=math.pi / 2,
            duration=1e-3,
            phase_offset=math.pi / 2,
            freq_offset=1e3,
            use="excitation",
        ),
        pp.make_gauss_pulse(
            flip_angle=math.pi / 2, duration=1e-3, time_bw_product=1, use="excitation"
        ),
        pp.make_gauss_pulse(
            flip_angle=math.pi / 2, duration=1e-3, apodization=0.1, use="excitation"
        ),
    ):
        seq.add_block(pulse)
        seq.add_block(pp.make_delay(1))
    return seq


def sinc_pulses() -> Sequence:
    """Sinc pulses spanning the same parameter space."""
    seq = Sequence()
    for pulse in (
        pp.make_sinc_pulse(flip_angle=1, use="excitation"),
        pp.make_sinc_pulse(flip_angle=1, delay=1e-3, use="excitation"),
        pp.make_sinc_pulse(flip_angle=math.pi / 2, use="excitation"),
        pp.make_sinc_pulse(flip_angle=math.pi / 2, duration=2e-3, use="excitation"),
        pp.make_sinc_pulse(
            flip_angle=math.pi / 2,
            duration=2e-3,
            phase_offset=math.pi / 2,
            use="excitation",
        ),
        pp.make_sinc_pulse(
            flip_angle=math.pi / 2,
            duration=2e-3,
            phase_offset=math.pi / 2,
            freq_offset=1e3,
            use="excitation",
        ),
        pp.make_sinc_pulse(
            flip_angle=math.pi / 2, duration=2e-3, time_bw_product=1, use="excitation"
        ),
        pp.make_sinc_pulse(
            flip_angle=math.pi / 2, duration=2e-3, apodization=0.1, use="excitation"
        ),
    ):
        seq.add_block(pulse)
        seq.add_block(pp.make_delay(1))
    return seq


def block_pulses() -> Sequence:
    """Hard pulses, whose envelope needs no shape at all."""
    seq = Sequence()
    for pulse in (
        pp.make_block_pulse(flip_angle=1, duration=4e-3, use="excitation"),
        pp.make_block_pulse(flip_angle=1, delay=1e-3, duration=4e-3, use="excitation"),
        pp.make_block_pulse(flip_angle=math.pi / 2, duration=4e-3, use="excitation"),
        pp.make_block_pulse(flip_angle=math.pi / 2, duration=1e-3, use="excitation"),
        pp.make_block_pulse(
            flip_angle=math.pi / 2,
            duration=2e-3,
            phase_offset=math.pi / 2,
            use="excitation",
        ),
        pp.make_block_pulse(
            flip_angle=math.pi / 2,
            duration=1e-3,
            phase_offset=math.pi / 2,
            freq_offset=1e3,
            use="excitation",
        ),
        pp.make_block_pulse(
            flip_angle=math.pi / 2, duration=1e-3, time_bw_product=1, use="excitation"
        ),
    ):
        seq.add_block(pulse)
        seq.add_block(pp.make_delay(1))
    return seq


def gradients_all_axes() -> Sequence:
    """Trapezoids on every axis, including pairs that collide after rounding."""
    seq = Sequence()
    seq.add_block(pp.make_block_pulse(math.pi / 4, duration=1e-3, use="excitation"))
    seq.add_block(pp.make_trapezoid("x", area=1000))
    seq.add_block(pp.make_trapezoid("y", area=-500.00001))
    seq.add_block(pp.make_trapezoid("z", area=100))
    seq.add_block(pp.make_trapezoid("x", area=-1000), pp.make_trapezoid("y", area=500))
    seq.add_block(pp.make_trapezoid("y", area=-500), pp.make_trapezoid("z", area=1000))
    seq.add_block(
        pp.make_trapezoid("x", area=-1000), pp.make_trapezoid("z", area=1000.00001)
    )
    return seq


def spin_echo() -> Sequence:
    """Excitation, refocusing and a readout with an ADC."""
    seq = Sequence()
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3))
    seq.add_block(pp.make_trapezoid("x", area=1000))
    seq.add_block(pp.make_trapezoid("x", area=-1000))
    seq.add_block(pp.make_block_pulse(math.pi, duration=1e-3))
    seq.add_block(pp.make_trapezoid("x", area=-500))
    seq.add_block(
        pp.make_trapezoid("x", area=1000, duration=10e-3),
        pp.make_adc(num_samples=100, duration=10e-3),
    )
    return seq


def gre_with_label_inc() -> Sequence:
    """A gradient echo whose line counter is incremented by a LABELINC."""
    seq = Sequence()
    for i in range(10):
        seq.add_block(pp.make_block_pulse(math.pi / 8, duration=1e-3))
        seq.add_block(pp.make_trapezoid("x", area=1000))
        seq.add_block(pp.make_trapezoid("y", area=-500 + i * 100))
        seq.add_block(pp.make_trapezoid("x", area=-500))
        seq.add_block(
            pp.make_trapezoid("x", area=1000, duration=10e-3),
            pp.make_adc(num_samples=100, duration=10e-3),
            pp.make_label(label="LIN", type="INC", value=1),
        )
    return seq


def gre_with_label_set() -> Sequence:
    """The same scan with the line counter set outright by a LABELSET."""
    seq = Sequence()
    for i in range(10):
        seq.add_block(pp.make_block_pulse(math.pi / 8, duration=1e-3))
        seq.add_block(pp.make_trapezoid("x", area=1000))
        seq.add_block(pp.make_trapezoid("y", area=-500 + i * 100))
        seq.add_block(pp.make_trapezoid("x", area=-500))
        seq.add_block(
            pp.make_trapezoid("x", area=1000, duration=10e-3),
            pp.make_adc(num_samples=100, duration=10e-3),
            pp.make_label(label="LIN", type="SET", value=i),
        )
    return seq


def gre_with_noise_scan() -> Sequence:
    """A gradient echo preceded by a noise acquisition, with slice selection."""
    system = pp.Opts()
    seq = Sequence(system)
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=math.pi / 8, duration=1e-3, slice_thickness=3e-3, return_gz=True
    )
    gx = pp.make_trapezoid(
        channel="x", flat_area=32 / 0.3, flat_time=32e-4, system=system
    )
    adc = pp.make_adc(
        num_samples=32, duration=gx.flat_time, delay=gx.rise_time, system=system
    )
    gx_pre = pp.make_trapezoid(
        channel="x", area=-gx.area / 2, duration=1e-3, system=system
    )
    phase_areas = -(np.arange(32) - 16) / 0.3

    seq.add_block(
        pp.make_label(label="LIN", type="SET", value=0),
        pp.make_label(label="SLC", type="SET", value=0),
    )
    seq.add_block(
        pp.make_adc(num_samples=1000, duration=1e-3),
        pp.make_label(label="NOISE", type="SET", value=True),
    )
    seq.add_block(pp.make_label(label="NOISE", type="SET", value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    for line in range(32):
        gy_pre = pp.make_trapezoid(
            channel="y", area=phase_areas[line], duration=1e-3, system=system
        )
        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy_pre, gzr)
        seq.add_block(gx, adc, pp.make_label(label="LIN", type="SET", value=line))
        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(gx_pre, gy_pre, pp.make_delay(10e-3))
    return seq


def gre_with_soft_delay() -> Sequence:
    """A gradient echo whose echo time is left adjustable at the console."""
    seq = Sequence()
    for i in range(10):
        seq.add_block(pp.make_block_pulse(math.pi / 8, duration=1e-3))
        seq.add_block(pp.make_trapezoid("x", area=1000))
        seq.add_block(pp.make_trapezoid("y", area=-500 + i * 100))
        seq.add_block(pp.make_trapezoid("x", area=-500))
        seq.add_block(
            pp.make_soft_delay(
                numID=0, hint="TE", offset=1, factor=1.0, default_duration=10e-6
            )
        )
        seq.add_block(
            pp.make_trapezoid("x", area=1000, duration=10e-3),
            pp.make_adc(num_samples=100, duration=10e-3),
        )
    return seq


def arbitrary_gradients() -> Sequence:
    """Waveforms given sample by sample, so the file carries them as shapes."""
    system = pp.Opts()
    seq = Sequence(system)
    turns = np.linspace(0, 6 * math.pi, 600)
    # Windowed so every waveform starts and ends at zero: a gradient that
    # stopped mid-amplitude would not be continuous with the next block.
    envelope = np.hanning(600)
    scale = 0.4 * system.max_grad
    seq.add_block(
        pp.make_arbitrary_grad("x", scale * envelope * np.cos(turns), system=system),
        pp.make_arbitrary_grad("y", scale * envelope * np.sin(turns), system=system),
    )
    seq.add_block(pp.make_delay(1e-3))
    seq.add_block(pp.make_arbitrary_grad("z", scale * np.hanning(400), system=system))
    return seq


def extended_trapezoids() -> Sequence:
    """Gradients given as corner points, which carry a time shape with them."""
    system = pp.Opts()
    seq = Sequence(system)
    amplitude = 0.3 * system.max_grad
    seq.add_block(
        pp.make_extended_trapezoid(
            "x",
            amplitudes=np.array([0.0, amplitude, amplitude, 0.0]),
            times=np.array([0.0, 200e-6, 1.2e-3, 1.5e-3]),
            system=system,
        )
    )
    seq.add_block(
        pp.make_extended_trapezoid(
            "y",
            amplitudes=np.array([0.0, -amplitude, amplitude, 0.0]),
            times=np.array([0.0, 300e-6, 900e-6, 1.4e-3]),
            system=system,
        )
    )
    return seq


def arbitrary_rf() -> Sequence:
    """An RF pulse given sample by sample, with its own magnitude and phase."""
    system = pp.Opts()
    seq = Sequence(system)
    samples = np.linspace(-3, 3, 500)
    signal = np.sinc(samples) * np.exp(1j * 0.4 * math.pi * samples)
    seq.add_block(
        pp.make_arbitrary_rf(
            signal=signal,
            flip_angle=math.pi / 4,
            dwell=10e-6,
            use="excitation",
            system=system,
        )
    )
    seq.add_block(pp.make_delay(10e-3))
    seq.add_block(
        pp.make_arbitrary_rf(
            signal=np.hanning(250).astype(complex),
            flip_angle=math.pi,
            dwell=10e-6,
            use="refocusing",
            system=system,
        )
    )
    return seq


def adc_with_phase_modulation() -> Sequence:
    """An ADC whose phase is stepped sample by sample, carried as a shape."""
    system = pp.Opts()
    seq = Sequence(system)
    samples = 128
    modulation = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    gx = pp.make_trapezoid("x", flat_area=1000, flat_time=2.56e-3, system=system)
    seq.add_block(
        gx,
        pp.make_adc(
            num_samples=samples,
            duration=gx.flat_time,
            delay=gx.rise_time,
            phase_modulation=modulation,
            system=system,
        ),
    )
    return seq


def trapezoid_only() -> Sequence:
    """One trapezoid: the file ends on its `[TRAP]` section."""
    seq = Sequence()
    seq.add_block(pp.make_trapezoid("x", area=1000))
    return seq


def adc_only() -> Sequence:
    """One ADC: the file ends on its `[ADC]` section."""
    seq = Sequence()
    seq.add_block(pp.make_adc(num_samples=100, duration=10e-3))
    return seq


def extension_only() -> Sequence:
    """An ADC labelled as noise: the file ends on its `[EXTENSIONS]` section."""
    seq = Sequence()
    seq.add_block(
        pp.make_adc(num_samples=1000, duration=1e-3),
        pp.make_label(label="NOISE", type="SET", value=True),
    )
    return seq


#: Every reference sequence, by the name its test case reports.
ZOO = {
    "gauss_pulses": gauss_pulses,
    "sinc_pulses": sinc_pulses,
    "block_pulses": block_pulses,
    "gradients_all_axes": gradients_all_axes,
    "spin_echo": spin_echo,
    "gre_with_label_inc": gre_with_label_inc,
    "gre_with_label_set": gre_with_label_set,
    "gre_with_noise_scan": gre_with_noise_scan,
    "gre_with_soft_delay": gre_with_soft_delay,
    "arbitrary_gradients": arbitrary_gradients,
    "extended_trapezoids": extended_trapezoids,
    "arbitrary_rf": arbitrary_rf,
    "adc_with_phase_modulation": adc_with_phase_modulation,
    "trapezoid_only": trapezoid_only,
    "adc_only": adc_only,
    "extension_only": extension_only,
}
