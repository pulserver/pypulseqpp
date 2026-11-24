"""Fat saturation of the whole transmit volume."""

from __future__ import annotations

__all__ = [
    "FAT_INVERSION_BANDWIDTH_PPM",
    "FAT_SAT_BANDWIDTH_PPM",
    "FAT_T1_S",
    "FatInversion",
    "FatSaturation",
    "fat_null_flip_deg",
    "fat_null_inversion_time",
]

import math

import numpy as np

import pypulseqpp as pp

from ..._adiabatic import HYPSEC_MU
from ..excitation._base import RfModule, adiabatic_fitted, rf_reference
from ._common import spoiler_gradients

#: Main fat resonance relative to water (ppm); negative, because fat precesses
#: more slowly than water.
FAT_SHIFT_PPM = -3.45

#: Spectral passband of the saturation (ppm of the field): wide enough to hold
#: fat that the head's own field moves by a few hundred hertz at 7 T, and
#: still clear of water.
FAT_SAT_BANDWIDTH_PPM = 2.0

#: Inverted band of the fat inversion (ppm of the field): the methylene peaks
#: from -3.8 to -2.6 ppm with the head's field spread on either side.
FAT_INVERSION_BANDWIDTH_PPM = 2.2

#: Centre of the fat inversion's band (ppm from water), between the main
#: methylene peak and the methyl peak at -3.8 ppm.
FAT_INVERSION_CENTRE_PPM = -3.5

#: Product of the fat inversion's duration and its band (Hz s): the
#: hyperbolic secant's transition narrows as the pulse lengthens, and at 13 it
#: leaves water untouched 2.3 ppm from the band's edge.
FAT_INVERSION_DURATION_BANDWIDTH = 13.0

#: T1 of fat (s): subcutaneous fat at 3 T, 371 ms (Gold et al., AJR Am J
#: Roentgenol 183:343, 2004).
FAT_T1_S = 0.37


def fat_null_flip_deg(
    delay_s: float, period_s: float, *, t1_s: float = FAT_T1_S
) -> float:
    """Return the saturation flip angle (degrees) that nulls fat at the excitation.

    In the steady state of one saturation pulse every ``period_s``, fat is
    zero ``delay_s`` after the pulse when the flip ``a`` satisfies
    ``cos(a) = (1 - exp(delay/T1)) / (1 - exp(-(period - delay)/T1))``, a
    little over 90 degrees.

    Parameters
    ----------
    delay_s : float
        Time from the saturation pulse's centre to the excitation's (s).
    period_s : float
        Time between two saturation pulses (s).
    t1_s : float, default=FAT_T1_S
        T1 of fat (s).

    Raises
    ------
    ValueError
        If a time is not positive, the delay is not shorter than the period,
        or the fat recovers so fast that no flip leaves it at zero.

    Examples
    --------
    >>> from pypulseqpp.sequences import fat_null_flip_deg
    >>> round(fat_null_flip_deg(5e-3, 50e-3), 1)
    96.8
    """
    if min(delay_s, period_s, t1_s) <= 0 or delay_s >= period_s:
        raise ValueError("times must be positive and the delay shorter than the period")
    cosine = (1.0 - math.exp(delay_s / t1_s)) / (
        1.0 - math.exp(-(period_s - delay_s) / t1_s)
    )
    if cosine < -1.0:
        raise ValueError(
            f"fat recovers too fast between the pulses, {period_s * 1e3:.1f} ms apart, "
            f"for any flip to leave it at zero {delay_s * 1e3:.1f} ms after one"
        )
    return math.degrees(math.acos(cosine))


def fat_null_inversion_time(period_s: float, *, t1_s: float = FAT_T1_S) -> float:
    """Return the time (s) from a fat inversion's centre at which fat crosses zero.

    In the steady state of one complete inversion every ``period_s``, fat
    recovers from ``-(1 - E) / (1 + E)``, ``E = exp(-period / T1)``, and
    crosses zero at ``T1 ln(2 / (1 + E))``, a little under half the period.

    Parameters
    ----------
    period_s : float
        Time between two inversions (s).
    t1_s : float, default=FAT_T1_S
        T1 of fat (s).

    Raises
    ------
    ValueError
        If a time is not positive.

    Examples
    --------
    >>> from pypulseqpp.sequences import fat_null_inversion_time
    >>> round(fat_null_inversion_time(80e-3) * 1e3, 1)
    37.8
    """
    if min(period_s, t1_s) <= 0:
        raise ValueError("times must be positive")
    return t1_s * math.log(2.0 / (1.0 + math.exp(-period_s / t1_s)))


class FatSaturation(RfModule):
    """Spectrally selective fat saturation of the whole transmit volume, with a three-axis spoiler.

    The pulse plays with no gradient, so a field-of-view offset or rotation
    applied afterwards does not change it.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    freq_offset_ppm : float, default=FAT_SHIFT_PPM
        Fat offset from water (ppm), carried on the pulse as a ppm offset,
        which an interpreter resolves against the field of the scan.
    flip_angle_deg : float, default=110.0
        Saturation flip angle (degrees).
    bandwidth_hz : float or None, default=None
        Spectral passband (Hz); ``FAT_SAT_BANDWIDTH_PPM`` of ``system.B0``
        when not given. The pulse lasts ``time_bw_product / bandwidth_hz``,
        rounded up to an even number of RF raster samples.
    time_bw_product : float, default=2.0
        Time-bandwidth product.
    spoiling_cycles : float, default=4.0
        Cycles of dephasing of each spoiler across ``voxel_size_m``.
    voxel_size_m : float, default=0.001
        Length the dephasing is counted over (m).

    Attributes
    ----------
    rf_prep : RfEvent
        The saturation pulse.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        Spoiler gradient of each axis.
    freq_offset_ppm : float
        The fat offset the pulse carries (ppm).
    freq_offset_hz : float
        The offset at ``system.B0`` (Hz); informational, the pulse carries
        ``freq_offset_ppm``.

    Raises
    ------
    ValueError
        If a bandwidth, time-bandwidth product or voxel size is not positive,
        or ``spoiling_cycles`` is not positive.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> fatsat = design.FatSaturation(pp.Opts(B0=3.0))
    >>> len(fatsat.blocks), round(fatsat.freq_offset_hz)
    (2, -441)

    >>> [event.type for event in fatsat.blocks[0]]
    ['rf']

    The pulse with its spoiler, and the longitudinal magnetisation left
    across the spectrum. The band sits at the fat resonance for the field
    the module was designed at, and water at zero offset is untouched:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp.plot import plot_rf
       from pypulseqpp.sequences import FatSaturation

       module = FatSaturation(pp.Opts(B0=3.0))
       module.seq.paper_plot()
       plot_rf(module, whole=True, extent=(-800, 400), plot_now=False)
       plt.show()
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        freq_offset_ppm: float = FAT_SHIFT_PPM,
        flip_angle_deg: float = 110.0,
        bandwidth_hz: float | None = None,
        time_bw_product: float = 2.0,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
    ) -> None:
        if bandwidth_hz is None:
            bandwidth_hz = 1e-6 * system.gamma * system.B0 * FAT_SAT_BANDWIDTH_PPM
        if bandwidth_hz <= 0 or time_bw_product <= 0:
            raise ValueError("bandwidth_hz and time_bw_product must be positive")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if spoiling_cycles <= 0:
            raise ValueError(
                "a fat saturation has to be spoiled: without it the fat the pulse just "
                "tipped over stays in the transverse plane and is read with the water"
            )

        samples = 2 * math.ceil(
            time_bw_product / bandwidth_hz / system.rf_raster_time / 2
        )
        rf_prep = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=samples * system.rf_raster_time,
            slice_thickness=0.0,
            time_bw_product=time_bw_product,
            pulse_type="sat",
            freq_ppm=freq_offset_ppm,
            return_gz=False,
            use="saturation",
            system=system,
        )
        gx_spoil, gy_spoil, gz_spoil = spoiler_gradients(
            system, spoiling_cycles, voxel_size_m
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep)
        self.seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        self.center = rf_reference(rf_prep)
        self.freq_offset_ppm = float(freq_offset_ppm)
        self.freq_offset_hz = float(1e-6 * system.gamma * system.B0 * freq_offset_ppm)


class FatInversion(RfModule):
    """Spectrally selective adiabatic inversion of fat (SPAIR), with a three-axis spoiler.

    A hyperbolic secant without gradient sweeps a band of
    ``bandwidth_ppm`` centred ``freq_offset_ppm`` from water, so it inverts
    fat over the whole transmit volume, every fat peak in the band alike, and
    the inversion holds while B1 stays above the adiabatic threshold. The
    caller places the excitation :func:`fat_null_inversion_time` after the
    pulse's centre. A pulse whose peak exceeds the derated ``system.max_b1``
    is lengthened as :class:`InversionPreparation` lengthens one.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    freq_offset_ppm : float, default=FAT_INVERSION_CENTRE_PPM
        Centre of the band from water (ppm), carried on the pulse as a ppm
        offset.
    bandwidth_ppm : float, default=FAT_INVERSION_BANDWIDTH_PPM
        Inverted band (ppm of ``system.B0``).
    duration_s : float or None, default=None
        Pulse duration (s); ``FAT_INVERSION_DURATION_BANDWIDTH`` over the band
        in Hz when not given, rounded up to an even number of RF raster
        samples.
    adiabaticity : float, default=4.0
        Margin over the adiabatic condition at nominal B1. The inversion
        holds down to about ``1 / sqrt(adiabaticity)`` of nominal B1; RF
        energy grows in proportion.
    spoiling_cycles : float, default=4.0
        Cycles of dephasing of each spoiler across ``voxel_size_m``.
    voxel_size_m : float, default=0.001
        Length the dephasing is counted over (m).

    Attributes
    ----------
    rf_prep : RfEvent
        The inversion pulse.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        Spoiler gradient of each axis.
    pulse_duration : float
        Duration the pulse was designed at (s).
    adiabaticity : float
        Adiabaticity the pulse was designed at.
    freq_offset_ppm : float
        Centre of the band (ppm).
    bandwidth_hz : float
        The band at ``system.B0`` (Hz).

    Raises
    ------
    ValueError
        If the band, duration, adiabaticity, spoiling or voxel size is not
        positive.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> spair = design.FatInversion(pp.Opts(B0=7.0))
    >>> len(spair.blocks), round(spair.pulse_duration * 1e3, 1)
    (2, 19.8)
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        freq_offset_ppm: float = FAT_INVERSION_CENTRE_PPM,
        bandwidth_ppm: float = FAT_INVERSION_BANDWIDTH_PPM,
        duration_s: float | None = None,
        adiabaticity: float = 4.0,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
    ) -> None:
        bandwidth_hz = 1e-6 * system.gamma * system.B0 * bandwidth_ppm
        if bandwidth_hz <= 0 or adiabaticity <= 0:
            raise ValueError("bandwidth_ppm and adiabaticity must be positive")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if spoiling_cycles <= 0:
            raise ValueError(
                "a fat inversion has to be spoiled: an imperfect inversion leaves "
                "fat in the transverse plane, read with the water"
            )
        if duration_s is None:
            duration_s = FAT_INVERSION_DURATION_BANDWIDTH / bandwidth_hz
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        raster = 2 * system.rf_raster_time
        duration_s = raster * math.ceil(duration_s / raster - 1e-9)
        # The sweep spans mu * beta rad/s, pi times the band in Hz.
        beta = math.pi * bandwidth_hz / HYPSEC_MU

        def make(duration: float, margin: float):
            stretch = duration / duration_s
            return pp.make_adiabatic_pulse(
                "hypsec",
                duration=duration,
                beta=beta / stretch,
                mu=HYPSEC_MU * stretch,
                adiabaticity=margin,
                freq_ppm=freq_offset_ppm,
                use="inversion",
                system=system,
            )

        self.pulse_duration, self.adiabaticity, rf_prep = adiabatic_fitted(
            system, duration_s, adiabaticity, make, raster=raster
        )
        gx_spoil, gy_spoil, gz_spoil = spoiler_gradients(
            system, spoiling_cycles, voxel_size_m
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep)
        self.seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        self.center = rf_reference(rf_prep)
        self.freq_offset_ppm = float(freq_offset_ppm)
        self.bandwidth_hz = float(bandwidth_hz)
