"""Excitation selective in frequency: a spectral band, or a band and a slice."""

from __future__ import annotations

__all__ = ["FrequencySelectiveExcitation", "SpspExcitation"]

import numpy as np

import pypulseqpp as pp

from ._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class FrequencySelectiveExcitation(RfModule):
    """SLR excitation of a spectral band without spatial selection.

    Duration is time_bw_product / bandwidth_hz, rounded up to the RF raster.
    Absolute-Hz and field-relative ppm offsets are additive.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float
        Nominal flip angle (degrees).
    bandwidth_hz : float
        Spectral passband (Hz).
    freq_offset_ppm : float, optional
        Centre of the band, relative to water (ppm). Water itself by default.
    freq_offset_hz : float, optional
        Centre of the band as a frequency (Hz), added to whatever
        ``freq_offset_ppm`` asks for.
    time_bw_product : float, optional
        Time-bandwidth product. Higher is a squarer band and a longer pulse.
    pulse_type : {'st', 'ex', 'se', 'inv', 'sat'}, optional
        SLR design family.
    use : str, optional
        Pulseq RF-use tag, used by trajectory integration.
    passband_ripple, stopband_ripple : float, optional
        Ripple allowed in each band of the spectral profile.

    Attributes
    ----------
    rf : RfEvent
        The pulse.
    duration_s : float
        Its length (s), which the passband fixed.
    bandwidth_hz : float
        The passband asked for.

    Raises
    ------
    ValueError
        If ``bandwidth_hz`` or ``time_bw_product`` is not positive.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> water = design.FrequencySelectiveExcitation(pp.Opts(), 90.0, bandwidth_hz=200.0)
    >>> len(water.blocks), round(water.duration_s * 1e3, 3)
    (1, 20.0)

    >>> narrow = design.FrequencySelectiveExcitation(pp.Opts(), 90.0, bandwidth_hz=100.0)
    >>> narrow.duration_s == 2 * water.duration_s
    True
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float,
        *,
        bandwidth_hz: float,
        freq_offset_ppm: float = 0.0,
        freq_offset_hz: float = 0.0,
        time_bw_product: float = 4.0,
        pulse_type: str = "st",
        use: str = "excitation",
        passband_ripple: float = 0.01,
        stopband_ripple: float = 0.01,
    ) -> None:
        if bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")
        if time_bw_product <= 0:
            raise ValueError("time_bw_product must be positive")

        duration_s = pp.ceil_to_raster(
            time_bw_product / bandwidth_hz, system.rf_raster_time
        )
        rf = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=duration_s,
            time_bw_product=time_bw_product,
            pulse_type=pulse_type,
            passband_ripple=passband_ripple,
            stopband_ripple=stopband_ripple,
            freq_offset=freq_offset_hz,
            freq_ppm=freq_offset_ppm,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf)

        self.center = rf_reference(rf)
        self.duration_s = duration_s
        self.bandwidth_hz = bandwidth_hz


class SpspExcitation(RfModule):
    """Spectral-spatial excitation using SLR subpulses on an alternating gradient.

    The spectral time-bandwidth product and bandwidth set total duration.
    Subpulse count controls spectral repetition spacing.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float
        Nominal flip angle (degrees).
    thickness_m : float
        Slice thickness (m).
    spectral_bandwidth_hz : float
        Spectral passband (Hz). With ``spectral_time_bw_product`` it fixes the
        total duration.
    freq_offset_hz : float, optional
        Centre of the spectral band (Hz).
    is_slab : bool, optional
        Merge any rephaser into the selection gradient rather than a second block.
    rephase : bool, optional
        Include a slice rephaser.
    spatial_time_bw_product : float, optional
        Time-bandwidth product of each spatial subpulse.
    spectral_time_bw_product : float, optional
        Time-bandwidth product of the spectral envelope.
    n_subpulses : int, optional
        Subpulses in the train.
    axis : {'z', 'x', 'y'}, optional
        Selection axis.
    use : str, optional
        Pulseq RF-use tag, used by trajectory integration.

    Attributes
    ----------
    rf : RfEvent
        The pulse.
    gz : GradEvent
        The alternating selection gradient.
    gz_reph : TrapEvent
        Its rephaser, only when needed, rephase=True and is_slab=False.

    Raises
    ------
    ValueError
        If a size is not positive, ``axis`` is not a gradient channel, or the
        requested selectivity exceeds a gradient limit.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> water = design.SpspExcitation(
    ...     system, 30.0, thickness_m=10e-3, spectral_bandwidth_hz=300.0
    ... )
    >>> len(water.blocks), water.gz.channel
    (2, 'z')

    >>> water.gz.type
    'grad'
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float,
        *,
        thickness_m: float,
        spectral_bandwidth_hz: float,
        freq_offset_hz: float = 0.0,
        is_slab: bool = False,
        rephase: bool = True,
        spatial_time_bw_product: float = 4.0,
        spectral_time_bw_product: float = 3.0,
        n_subpulses: int = 10,
        axis: str = "z",
        use: str = "excitation",
    ) -> None:
        if thickness_m <= 0:
            raise ValueError("thickness_m must be positive")
        if spectral_bandwidth_hz <= 0:
            raise ValueError("spectral_bandwidth_hz must be positive")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf, gz, gz_reph = pp.make_spsp_pulse(
            np.deg2rad(flip_angle_deg),
            slice_thickness=thickness_m,
            spectral_bandwidth=spectral_bandwidth_hz,
            freq_offset=freq_offset_hz,
            spatial_time_bandwidth_product=spatial_time_bw_product,
            spectral_time_bandwidth_product=spectral_time_bw_product,
            n_subpulses=n_subpulses,
            system=system,
            use=use,
        )
        gz.channel = axis
        if gz_reph is not None:
            gz_reph.channel = axis

        rephase = rephase and gz_reph is not None

        self.seq = pp.Sequence(system)
        if is_slab:
            if rephase:
                # Concatenated, not summed over one interval: the rephaser
                # starts where the subpulse train ends.
                gz_reph.delay = pp.calc_duration(gz)
                gz = pp.add_gradients(grads=[gz, gz_reph], system=system)
            self.seq.add_block(rf, gz)
        else:
            self.seq.add_block(rf, gz)
            if rephase:
                self.seq.add_block(gz_reph)

        self.center = rf_reference(rf)
        self.spectral_bandwidth_hz = spectral_bandwidth_hz
        self.n_subpulses = n_subpulses
