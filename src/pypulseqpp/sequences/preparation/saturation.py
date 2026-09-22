"""Off-resonance MT, ihMT and Bloch-Siegert preparations."""

from __future__ import annotations

__all__ = [
    "BlochSiegertPreparation",
    "IhMtPreparation",
    "MtPreparation",
    "OffResonanceSaturation",
]

from typing import Any

import numpy as np

import pypulseqpp as pp

from ..excitation._base import RfModule, rf_reference
from ._common import spoiler_gradients


class OffResonanceSaturation(RfModule):
    """Repeat one off-resonance pulse, followed by an optional three-axis spoiler.

    Parameters
    ----------
    system : Opts
        System limits.
    rf_prep : RfEvent
        The pulse to play.
    n_pulses : int, default=1
        Pulses in the train.
    spoiling_cycles : float, default=4.0
        Dephasing of the final spoiler, in cycles across ``voxel_size_m``.
        Zero omits the spoiler, as required for a pulse used for its phase
        effect rather than for saturation.
    voxel_size_m : float, default=0.001
        Length the dephasing is counted over (m).
    labels : Sequence[str], default=None
        Counters emitted on the first pulse's block.

    Attributes
    ----------
    rf_prep : RfEvent
        The pulse, published once however many times it is played.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        The closing spoiler, when there is one.
    prep_labels : LabelEvent | list[LabelEvent]
        One per name in ``labels``.

    Raises
    ------
    ValueError
        If ``n_pulses`` is below one, or a spoiling argument is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts()
    >>> pulse = pp.make_gauss_pulse(
    ...     flip_angle=np.deg2rad(500), duration=8e-3, freq_offset=-1500.0,
    ...     use="saturation", system=system,
    ... )
    >>> prep = design.OffResonanceSaturation(system, pulse, n_pulses=3)
    >>> len(prep.blocks)
    4

    >>> prep.blocks[0] == prep.blocks[1] == prep.blocks[2]
    True

    Three pulses back to back, and the longitudinal magnetization they
    leave across the spectrum:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import numpy as np
       import pypulseqpp as pp
       from pypulseqpp.plot import plot_rf
       from pypulseqpp.sequences import OffResonanceSaturation

       system = pp.Opts()
       pulse = pp.make_gauss_pulse(
           flip_angle=np.deg2rad(500),
           duration=8e-3,
           freq_offset=-1500.0,
           use="saturation",
           system=system,
       )
       module = OffResonanceSaturation(system, pulse, n_pulses=3)
       module.seq.paper_plot()
       plot_rf(module, whole=True, extent=3000, plot_now=False)
       plt.show()
    """

    def init_module(
        self,
        system: pp.Opts,
        rf_prep: Any,
        *,
        n_pulses: int = 1,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        n_pulses = int(n_pulses)
        if n_pulses < 1:
            raise ValueError("n_pulses must be >= 1")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")

        prep_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        for index in range(n_pulses):
            self.seq.add_block(rf_prep, *(prep_labels if index == 0 else ()))
        if spoiling_cycles:
            gx_spoil, gy_spoil, gz_spoil = spoiler_gradients(
                system, spoiling_cycles, voxel_size_m
            )
            self.seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        self.center = rf_reference(rf_prep)


def saturation_pulse(
    system: pp.Opts,
    flip_angle_deg: float,
    duration_s: float,
    freq_offset_hz: float,
    time_bw_product: float,
) -> Any:
    """Design the non-spatially-selective SLR envelope used by MT preparations.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits the pulse is designed against.
    flip_angle_deg : float
        Nominal flip angle, in degrees.
    duration_s : float
        Pulse duration, in s.
    freq_offset_hz : float
        Offset from water, in Hz.
    time_bw_product : float
        Time-bandwidth product, which with ``duration_s`` sets the saturated
        band.

    Returns
    -------
    object
        The RF event, with no selection gradient and ``use="saturation"``.
    """
    return pp.make_slr_pulse(
        np.deg2rad(flip_angle_deg),
        duration=duration_s,
        slice_thickness=0.0,
        time_bw_product=time_bw_product,
        pulse_type="sat",
        freq_offset=freq_offset_hz,
        use="saturation",
        system=system,
    )


class MtPreparation(OffResonanceSaturation):
    """Off-resonance SLR saturation followed by an optional spoiler.

    Parameters
    ----------
    system : Opts
        System limits.
    flip_angle_deg : float, default=500.0
        Nominal flip angle (degrees). Far above 90: the point is deposited
        power, not a tip.
    freq_offset_hz : float, default=-1500.0
        Offset from water (Hz); must not be zero.
    duration_s : float, default=0.008
        Pulse duration (s).
    time_bw_product : float, default=4.0
        Time-bandwidth product, which with ``duration_s`` sets how wide a band
        of the bound pool is saturated.

    Raises
    ------
    ValueError
        If ``freq_offset_hz`` is zero or a duration is not positive.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> mt = design.MtPreparation(pp.Opts())
    >>> len(mt.blocks), round(float(mt.rf_prep.freq_offset))
    (2, -1500)

    One band below resonance, with the free pool at zero offset not directly
    saturated:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp.plot import plot_rf
       from pypulseqpp.sequences import MtPreparation

       module = MtPreparation(pp.Opts())
       module.seq.paper_plot()
       plot_rf(module, whole=True, extent=3000, plot_now=False)
       plt.show()
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        flip_angle_deg: float = 500.0,
        freq_offset_hz: float = -1500.0,
        duration_s: float = 8e-3,
        time_bw_product: float = 4.0,
        **kwargs: Any,
    ) -> None:
        if freq_offset_hz == 0:
            raise ValueError(
                "freq_offset_hz must not be zero: an on-resonance saturation excites water "
                "rather than the bound pool"
            )
        if duration_s <= 0 or time_bw_product <= 0:
            raise ValueError("duration_s and time_bw_product must be positive")

        rf_prep = saturation_pulse(
            system, flip_angle_deg, duration_s, freq_offset_hz, time_bw_product
        )
        super().init_module(system, rf_prep, **kwargs)


class IhMtPreparation(OffResonanceSaturation):
    """Simultaneous dual-offset saturation for ihMT measurements.

    Each band receives 1/sqrt(2) of the matched single-offset amplitude.
    Compare against positive- and negative-offset MtPreparation acquisitions
    at matched deposited power.

    Parameters
    ----------
    system : Opts
        System limits.
    flip_angle_deg : float, default=500.0
        Flip angle (degrees) of the single-offset arm this one is matched
        against. Each band gets ``1 / sqrt(2)`` of it.
    freq_offset_hz : float, default=1500.0
        Distance of each band from water (Hz), positive. Both signs are built.
    duration_s : float, default=0.008
        Pulse duration (s).
    time_bw_product : float, default=4.0
        Time-bandwidth product of each band.

    Attributes
    ----------
    rf_prep : RfEvent
        The dual-band pulse.
    band_offsets_hz : NDArray[np.float64]
        Where the two bands actually landed (Hz), for a spectral check.

    Raises
    ------
    ValueError
        If ``freq_offset_hz`` is not positive or a duration is not positive.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> ihmt = design.IhMtPreparation(pp.Opts())
    >>> ihmt.band_offsets_hz.tolist()
    [-1500.0, 1500.0]

    Both sidebands at once, with the same total RF power as one band played
    alone:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp.plot import plot_rf
       from pypulseqpp.sequences import IhMtPreparation

       module = IhMtPreparation(pp.Opts())
       module.seq.paper_plot()
       plot_rf(module, whole=True, extent=3000, plot_now=False)
       plt.show()
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        flip_angle_deg: float = 500.0,
        freq_offset_hz: float = 1500.0,
        duration_s: float = 8e-3,
        time_bw_product: float = 4.0,
        **kwargs: Any,
    ) -> None:
        if freq_offset_hz <= 0:
            raise ValueError(
                "freq_offset_hz must be positive; both signs are built from it"
            )
        if duration_s <= 0 or time_bw_product <= 0:
            raise ValueError("duration_s and time_bw_product must be positive")

        # Two bands and no on-resonance one: the modulation puts them at 0 and
        # +2f, and the pulse's own offset then slides the pair to -f and +f.
        # Each band carries 1/sqrt(2) of the single-offset amplitude, so the two
        # together deposit the same power -- see the class docstring.
        single = saturation_pulse(
            system, flip_angle_deg / np.sqrt(2.0), duration_s, 0.0, time_bw_product
        )
        rf_prep, offsets, _ = pp.make_sms_pulse(single, 2, 2.0 * freq_offset_hz)
        rf_prep.freq_offset = -freq_offset_hz

        super().init_module(system, rf_prep, **kwargs)

        self.band_offsets_hz = np.asarray(offsets, dtype=float) - freq_offset_hz


class BlochSiegertPreparation(OffResonanceSaturation):
    """Off-resonance Fermi pulse for Bloch-Siegert B1 mapping.

    By default, no spoiler is played, preserving transverse phase between
    excitation and readout. Calibration constants describe phase per squared
    RF amplitude.

    Parameters
    ----------
    system : Opts
        System limits.
    freq_offset_hz : float, default=4000.0
        Offset from water (Hz); flip its sign for the second acquisition.
    duration_s : float, default=0.008
        Pulse duration (s).
    peak_b1_hz : float, default=500.0
        Plateau amplitude, in Pulseq units.
    flat_fraction : float, default=0.8
        Fraction of the duration held at ``peak_b1_hz``, in ``(0, 1)``.
    transition_fraction : float, default=0.02
        Width of each shoulder, as a fraction of the duration.
    dwell_s : float, default=1e-05
        RF raster (s).
    spoiling_cycles : float, default=0.0
        Cycles of dephasing the closing spoiler winds across ``voxel_size_m``.
        Zero, the default here, omits the spoiler, for the reason above.

    Attributes
    ----------
    rf_prep : RfEvent
        The Fermi pulse.
    kbs : float
        Phase shift per unit squared Pulseq amplitude (rad).
    kbs_per_gauss2 : float
        The same constant in rad/G^2.

    Raises
    ------
    ValueError
        If ``freq_offset_hz`` is zero, a duration or amplitude is not
        positive, or the envelope fractions are out of range.

    References
    ----------
    Sacolick et al., *B1 mapping by Bloch-Siegert shift*, Magn Reson Med
    2010;63:1315-1322, ``10.1002/mrm.22357``.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> pulse = design.BlochSiegertPreparation(pp.Opts())
    >>> len(pulse.blocks), round(pulse.kbs_per_gauss2, 1)
    (1, 86.6)

    The pulse is placed far enough off resonance to produce a phase shift
    with a small flip angle; the residual dip at its offset is the saturation
    that remains at that frequency offset:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp.plot import plot_rf
       from pypulseqpp.sequences import BlochSiegertPreparation

       module = BlochSiegertPreparation(pp.Opts())
       module.seq.paper_plot()
       plot_rf(module, whole=True, extent=(-1000, 8000), plot_now=False)
       plt.show()
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        freq_offset_hz: float = 4000.0,
        duration_s: float = 8e-3,
        peak_b1_hz: float = 500.0,
        flat_fraction: float = 0.8,
        transition_fraction: float = 0.02,
        dwell_s: float = 10e-6,
        spoiling_cycles: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if freq_offset_hz == 0:
            raise ValueError(
                "freq_offset_hz must not be zero: the shift is inversely in it"
            )
        if duration_s <= 0 or peak_b1_hz <= 0 or dwell_s <= 0:
            raise ValueError("duration_s, peak_b1_hz and dwell_s must be positive")
        if not 0 < flat_fraction < 1 or transition_fraction <= 0:
            raise ValueError(
                "flat_fraction must lie in (0, 1) and transition_fraction must be positive"
            )

        n_samples = max(2, round(duration_s / dwell_s))
        duration_s = n_samples * dwell_s
        times = (np.arange(n_samples) + 0.5) * dwell_s
        half_flat = 0.5 * flat_fraction * duration_s
        transition = transition_fraction * duration_s
        envelope = peak_b1_hz / (
            1.0 + np.exp((np.abs(times - 0.5 * duration_s) - half_flat) / transition)
        )
        rf_prep = pp.make_arbitrary_rf(
            signal=envelope,
            flip_angle=1.0,
            no_signal_scaling=True,
            dwell=dwell_s,
            freq_offset=freq_offset_hz,
            system=system,
            use="preparation",
        )

        super().init_module(system, rf_prep, spoiling_cycles=spoiling_cycles, **kwargs)

        # The shift is (gamma B1)^2 / (2 * 2 pi * offset), integrated over the
        # pulse. Reported twice: once in the amplitude units the sequence uses,
        # once per gauss squared, which is how the literature quotes it.
        denominator = 2.0 * 2.0 * np.pi * freq_offset_hz
        gamma_hz_per_gauss = system.gamma * 1e-4
        self.kbs = float(np.sum((2.0 * np.pi * envelope) ** 2 / denominator) * dwell_s)
        self.kbs_per_gauss2 = float(
            np.sum(
                (2.0 * np.pi * gamma_hz_per_gauss * envelope / peak_b1_hz) ** 2
                / denominator
            )
            * dwell_s
        )
