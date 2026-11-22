"""Fat saturation of the whole transmit volume."""

from __future__ import annotations

__all__ = ["FAT_T1_S", "FatSaturation", "fat_null_flip_deg"]

import math
from collections.abc import Sequence

import numpy as np

import pypulseqpp as pp

from ..excitation._base import RfModule, rf_reference
from ._common import spoiler_gradients

#: Main fat resonance relative to water (ppm); negative, because fat precesses
#: more slowly than water.
FAT_SHIFT_PPM = -3.45

#: T1 of fat (s): subcutaneous fat at 3 T, 371 ms (Gold et al., AJR Am J
#: Roentgenol 183:343, 2004).
FAT_T1_S = 0.37


#: Passband of the fat saturation, as a fraction of the fat offset from water.
FAT_BANDWIDTH_FRACTION = 0.7


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


def fat_null_flip_deg_cycle(
    period_s: float,
    saturations_s: Sequence[float],
    *,
    inversions_s: Sequence[float] = (),
    excitation_deg: float = 90.0,
    t1_s: float = FAT_T1_S,
) -> float:
    """Return the saturation flip angle (degrees) that nulls fat at an excitation repeated every ``period_s``, given every pulse fat sees between.

    Fat in the excited slice recovers with ``t1_s``; each saturation tips it
    by the flip sought and its transverse part is spoiled, each inversion
    negates it, and the excitation tips it by ``excitation_deg``. In the
    steady state the flip is the one, from 90 to 180 degrees and nearest 90,
    that leaves it at zero when the excitation comes; where none does, the
    one that leaves it smallest.

    Parameters
    ----------
    period_s : float
        Time between two excitations of the slice (s).
    saturations_s : sequence of float
        Time from the centre of each saturation to the centre of the next
        excitation (s), each in ``(0, period_s)``.
    inversions_s : sequence of float, default=()
        The same for each inversion.
    excitation_deg : float, default=90.0
        Flip angle of the excitation (degrees).
    t1_s : float, default=FAT_T1_S
        T1 of fat (s).

    Raises
    ------
    ValueError
        If no saturation is given, or a time is outside ``(0, period_s)``.

    Examples
    --------
    >>> from pypulseqpp.sequences import fat_null_flip_deg, fat_null_flip_deg_cycle
    >>> round(fat_null_flip_deg_cycle(50e-3, [5e-3]), 1) == round(fat_null_flip_deg(5e-3, 50e-3), 1)
    True
    """
    if not saturations_s:
        raise ValueError("fat needs at least one saturation in the period")
    for time in (*saturations_s, *inversions_s):
        if not 0 < time < period_s:
            raise ValueError(
                f"a pulse {time * 1e3:.3f} ms before the excitation is outside the "
                f"{period_s * 1e3:.3f} ms period"
            )
    events = sorted(
        [(period_s - t, "saturation") for t in saturations_s]
        + [(period_s - t, "inversion") for t in inversions_s]
    )
    excitation = math.cos(math.radians(excitation_deg))

    def before_excitation(flip_deg: float) -> float:
        saturation = math.cos(math.radians(flip_deg))

        def period(mz: float) -> float:
            mz, now = mz * excitation, 0.0
            for time, kind in events:
                mz = 1.0 - (1.0 - mz) * math.exp(-(time - now) / t1_s)
                mz = mz * (saturation if kind == "saturation" else -1.0)
                now = time
            return 1.0 - (1.0 - mz) * math.exp(-(period_s - now) / t1_s)

        offset = period(0.0)
        slope = period(1.0) - offset
        return offset / (1.0 - slope)

    flips = np.linspace(90.0, 180.0, 1801)
    values = np.array([before_excitation(flip) for flip in flips])
    crossing = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
    if crossing.size == 0:
        return float(flips[np.argmin(np.abs(values))])
    low, high = flips[crossing[0]], flips[crossing[0] + 1]
    for _ in range(40):
        middle = 0.5 * (low + high)
        if np.sign(before_excitation(middle)) == np.sign(before_excitation(low)):
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


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
    bandwidth_hz : float | None, default=None
        Spectral passband (Hz); the pulse duration is
        ``time_bw_product / bandwidth_hz``. ``None`` takes
        :data:`FAT_BANDWIDTH_FRACTION` of the fat offset at ``system.B0``,
        which keeps the passband off water at any field.
    time_bw_product : float, default=4.0
        Time-bandwidth product. At 2 the passband has no flat top: scalp fat
        that the field of a head moves 50 to 90 Hz off its peak at 3 T keeps a
        fifth to a half of its magnetization.
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
        time_bw_product: float = 4.0,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
    ) -> None:
        if bandwidth_hz is None:
            bandwidth_hz = FAT_BANDWIDTH_FRACTION * abs(
                1e-6 * system.gamma * system.B0 * freq_offset_ppm
            )
        if bandwidth_hz <= 0 or time_bw_product <= 0:
            raise ValueError("bandwidth_hz and time_bw_product must be positive")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if spoiling_cycles <= 0:
            raise ValueError(
                "a fat saturation has to be spoiled: without it the fat the pulse just "
                "tipped over stays in the transverse plane and is read with the water"
            )

        rf_prep = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=time_bw_product / bandwidth_hz,
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
