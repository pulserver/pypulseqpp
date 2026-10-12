"""Excitation selective in frequency, over a spectral band alone or with a slice."""

from __future__ import annotations

__all__ = ["FrequencySelectiveExcitation", "SpspExcitation"]

import numpy as np

import pypulseqpp as pp

from ._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class FrequencySelectiveExcitation(RfModule):
    """SLR excitation of a spectral band without spatial selection.

    The duration is ``time_bw_product / bandwidth_hz`` rounded up to the RF
    raster. The offsets in Hz and in ppm add.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    flip_angle_deg : float
        Flip angle (degrees).
    bandwidth_hz : float
        Spectral passband (Hz).
    freq_offset_ppm : float, default=0.0
        Centre of the band relative to water (ppm).
    freq_offset_hz : float, default=0.0
        Centre of the band (Hz), added to ``freq_offset_ppm``.
    time_bw_product : float, default=4.0
        Time-bandwidth product; a higher value gives a squarer band and a
        longer pulse.
    pulse_type : {'st', 'ex', 'se', 'inv', 'sat'}, default='st'
        SLR design: small-tip, excitation, spin-echo, inversion or saturation.
    use : str, default='excitation'
        Pulseq RF use tag.
    passband_ripple, stopband_ripple : float, default=0.01
        Ripple allowed in each band of the spectral profile.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    flip_angle_deg : float
        Flip angle (degrees).
    thickness_m : float
        Slice or slab thickness (m).
    spectral_bandwidth_hz : float
        Spectral passband (Hz); with ``spectral_time_bw_product`` it sets the
        total duration.
    freq_offset_hz : float, default=0.0
        Centre of the spectral band (Hz).
    is_slab : bool, default=False
        Merge the rephaser into the selection gradient instead of playing it in
        a second block.
    rephase : bool, default=True
        Include the slice rephaser, if the design requires one.
    spatial_time_bw_product : float, default=4.0
        Time-bandwidth product of each spatial subpulse.
    spectral_time_bw_product : float, default=3.0
        Time-bandwidth product of the spectral envelope.
    n_subpulses : int, default=10
        Subpulses in the train.
    axis : {'z', 'x', 'y'}, default='z'
        Selection axis.
    use : str, default='excitation'
        Pulseq RF use tag.


    Attributes
    ----------
    rf : RfEvent
        The pulse.
    duration_s : float
        Pulse length (s), set by the passband.
    bandwidth_hz : float
        Requested passband (Hz).

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

    The envelope and the band it tips. Magnetization outside the passband
    stays longitudinal:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp.plot import plot_rf
       from pypulseqpp.sequences import FrequencySelectiveExcitation

       module = FrequencySelectiveExcitation(pp.Opts(), 90.0, bandwidth_hz=200.0)
       module.seq.paper_plot()
       plot_rf(module, plot_now=False)
       plt.show()
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
            delay=system.rf_dead_time,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf)

        self.center = rf_reference(rf)
        self.duration_s = duration_s
        self.bandwidth_hz = bandwidth_hz


class SpspExcitation(RfModule):
    """Spectral-spatial excitation of SLR subpulses on an alternating gradient.

    The spectral bandwidth and time-bandwidth product set the total duration,
    and the subpulse count sets the spacing of the spectral repetitions. Under
    alternating gradient lobes a slice offset is a phase that follows the
    gradient's area, not a frequency offset: move the slice with
    :meth:`shift`.

    Attributes
    ----------
    rf : RfEvent
        The pulse.
    gz : GradEvent
        The alternating selection gradient.
    gz_reph : TrapEvent
        The rephaser; present only when the design requires one,
        ``rephase`` is true and ``is_slab`` is false.

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

    The subpulse train under its alternating gradient, and the transverse
    magnetization over position and frequency together. The spectral
    passband repeats at the subpulse rate, which ``n_subpulses`` sets:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp.plot import plot_rf
       from pypulseqpp.sequences import SpspExcitation

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
       module = SpspExcitation(
           system,
           30.0,
           thickness_m=10e-3,
           spectral_bandwidth_hz=300.0,
           n_subpulses=12,
       )
       module.seq.paper_plot()
       plot_rf(module, plane="zf", extent=15.0, span=2500.0, samples=81, plot_now=False)
       plt.show()
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

        # The selection gradient's area at each RF sample, from the pulse's
        # centre, taken before a slab's rephaser is merged into the lobes.
        samples = rf.delay + np.asarray(rf.t)
        corners = gz.delay + np.asarray(gz.tt)
        grid = np.union1d(corners, samples)
        amplitude = np.interp(
            grid, corners, np.asarray(gz.waveform), left=0.0, right=0.0
        )
        area = np.concatenate(
            [[0.0], np.cumsum(np.diff(grid) * (amplitude[1:] + amplitude[:-1]) / 2)]
        )
        self._area = np.interp(samples, grid, area) - np.interp(
            rf.delay + rf.center, grid, area
        )
        self._signal = np.array(rf.signal, dtype=complex)

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

    def shift(self, position: float, phase: float = 0.0):
        """Move the selected slice to ``position`` (m) and return the pulse.

        The designed waveform is modulated by ``2π position (K(t) - K(t_c))``,
        with ``K`` the selection gradient's area since the pulse began and
        ``t_c`` its centre, and ``phase`` becomes the phase at the centre.
        Under a constant gradient this is ``freq_offset = G position`` with the
        centre's phase compensated. The module's ``rf`` is changed in place.

        Parameters
        ----------
        position : float
            Slice position along the selection axis (m).
        phase : float, default=0.0
            Phase at the pulse centre (rad).

        Returns
        -------
        RfEvent
            The module's ``rf``, modulated in place.
        """
        self.rf.signal = self._signal * np.exp(2j * np.pi * position * self._area)
        self.rf.phase_offset = phase
        return self.rf
