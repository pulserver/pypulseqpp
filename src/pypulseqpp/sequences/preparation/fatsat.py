"""Fat saturation of the whole transmit volume."""

from __future__ import annotations

__all__ = ["FatSaturation"]

import numpy as np

import pypulseqpp as pp

from ..excitation._base import RfModule, rf_reference
from ._common import spoiler_gradients

#: Where the main fat resonance sits relative to water. Negative: fat precesses
#: more slowly, so it is downfield of water in frequency.
FAT_SHIFT_PPM = -3.45


class FatSaturation(RfModule):
    """Spectrally selective fat saturation of the whole transmit volume, with a three-axis spoiler.

    The pulse plays no gradient, so a field-of-view offset or rotation applied
    afterwards leaves it as it is: it saturates fat wherever the prescription
    puts the image.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    freq_offset_ppm : float, default=FAT_SHIFT_PPM
        Fat offset from water (ppm). Carried on the pulse as a ppm offset
        rather than as hertz, so the interpreter resolves it against the
        field the scan actually runs at.
    flip_angle_deg : float, default=110.0
        Saturation flip angle (degrees).
    bandwidth_hz : float, default=250.0
        Spectral passband (Hz). With ``time_bw_product`` it fixes the pulse
        duration, which is ``time_bw_product / bandwidth_hz``.
    time_bw_product : float, default=2.0
        Time-bandwidth product.
    spoiling_cycles : float, default=4.0
        Cycles of dephasing each spoiler axis winds across ``voxel_size_m``.
    voxel_size_m : float, default=0.001
        Length the dephasing is counted over (m).

    Attributes
    ----------
    rf_prep : RfEvent
        The saturation pulse.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        Closing spoiler on all three axes.
    freq_offset_ppm : float
        The fat offset the pulse carries (ppm).
    freq_offset_hz : float
        What that comes to at ``system.B0`` (Hz), for reference; the pulse
        itself is field-independent.

    Raises
    ------
    ValueError
        If a bandwidth, time-bandwidth product or voxel size is not positive,
        or the spoiler was asked for zero cycles.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> fatsat = design.FatSaturation(pp.Opts(B0=3.0))
    >>> len(fatsat.blocks), round(fatsat.freq_offset_hz)
    (2, -441)

    >>> [event.type for event in fatsat.blocks[0]]
    ['rf']

    The pulse with its spoiler, and the longitudinal magnetization left
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
        bandwidth_hz: float = 250.0,
        time_bw_product: float = 2.0,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
    ) -> None:
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
