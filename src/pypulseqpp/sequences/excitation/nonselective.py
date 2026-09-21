"""Non-selective excitation, refocusing and inversion."""

from __future__ import annotations

__all__ = ["NonSelectiveExcitation", "NonSelectiveRefocusing"]

import numpy as np

import pypulseqpp as pp

from ._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class NonSelectiveExcitation(RfModule):
    """Rectangular RF excitation without a selection gradient.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    flip_angle_deg : float, default=10.0
        Nominal flip angle (degrees).
    duration_s : float, default=0.001
        Pulse duration (s); spectral width scales inversely with duration.
    use : str, default='excitation'
        Pulseq use tag. Undefined pulses are treated as excitations by
        k-space analysis.
    freq_offset_hz, phase_offset_rad : float, default=0.0
        Transmit offsets designed into the pulse. A scan loop moves
        ``module.rf.freq_offset`` and ``module.rf.phase_offset`` per shot
        instead of rebuilding the module.

    Attributes
    ----------
    rf : RfEvent
        The pulse.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> excitation = design.NonSelectiveExcitation(pp.Opts(), flip_angle_deg=10.0)
    >>> round(excitation.center * 1e6)
    500

    Without a gradient the only selectivity is in frequency, and the profile
    is the transform of a rectangle:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp.plot import plot_rf
       from pypulseqpp.sequences import NonSelectiveExcitation

       module = NonSelectiveExcitation(pp.Opts(), 90.0, 0.5e-3)
       module.seq.paper_plot()
       plot_rf(module, plot_now=False)
       plt.show()
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float = 10.0,
        duration_s: float = 1e-3,
        *,
        use: str = "excitation",
        freq_offset_hz: float = 0.0,
        phase_offset_rad: float = 0.0,
    ) -> None:
        rf = pp.make_block_pulse(
            flip_angle=np.deg2rad(flip_angle_deg),
            duration=duration_s,
            freq_offset=freq_offset_hz,
            phase_offset=phase_offset_rad,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf)

        self.center = rf_reference(rf)


class NonSelectiveRefocusing(RfModule):
    """Rectangular refocusing pulse between two identical crushers.

    The default phase is pi/2 radians for CPMG with a zero-phase excitation.
    Set spoiling_cycles=0 to omit the crushers.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    flip_angle_deg : float, default=180.0
        Nominal flip angle (degrees).
    duration_s : float, default=0.001
        Pulse duration (s).
    spoiling_cycles : float, default=4.0
        Cycles of dephasing each crusher winds across ``voxel_size_m``. Zero
        omits the crushers, for an echo train that crushes elsewhere.
    voxel_size_m : float, default=0.001
        Length the dephasing is counted over (m).
    axis : {'z', 'x', 'y'}, default='z'
        Crusher axis.
    phase_offset_rad : float, default=np.pi / 2
        RF phase. The CPMG quarter turn by default.
    use : str, default='refocusing'
        What the pulse is for; the trajectory core negates accumulated k at a
        refocusing pulse, so this is not cosmetic.

    Attributes
    ----------
    rf_ref : RfEvent
        The refocusing pulse.
    gz_spoil : GradEvent
        The crusher, published once because both sides play the same event.

    Raises
    ------
    ValueError
        If ``spoiling_cycles`` is negative, ``voxel_size_m`` is not positive,
        or ``axis`` is not a gradient channel.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> refocusing = design.NonSelectiveRefocusing(pp.Opts())
    >>> len(refocusing.blocks), refocusing.blocks[0] == refocusing.blocks[2]
    (3, True)

    >>> len(design.NonSelectiveRefocusing(pp.Opts(), spoiling_cycles=0.0).blocks)
    1

    The pulse and both crushers, integrated as one window:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp.plot import plot_rf
       from pypulseqpp.sequences import NonSelectiveRefocusing

       module = NonSelectiveRefocusing(pp.Opts(), duration_s=0.5e-3)
       module.seq.paper_plot()
       plot_rf(module, whole=True, plot_now=False)
       plt.show()
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float = 180.0,
        duration_s: float = 1e-3,
        *,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        axis: str = "z",
        phase_offset_rad: float = np.pi / 2,
        use: str = "refocusing",
    ) -> None:
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf_ref = pp.make_block_pulse(
            flip_angle=np.deg2rad(flip_angle_deg),
            duration=duration_s,
            phase_offset=phase_offset_rad,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)

        if spoiling_cycles:
            gz_spoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, axis, system=system
            )
            self.seq.add_block(gz_spoil)

        lead = self.seq.duration()[0]
        self.seq.add_block(rf_ref)
        if spoiling_cycles:
            self.seq.add_block(gz_spoil)

        self.center = lead + rf_reference(rf_ref)
