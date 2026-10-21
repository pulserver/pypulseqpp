"""Inversion-recovery preparation."""

from __future__ import annotations

__all__ = ["InversionPreparation"]

import pypulseqpp as pp

from ..excitation._base import RfModule, adiabatic_fitted, held_sweep, rf_reference

_AXES = ("x", "y", "z")


class InversionPreparation(RfModule):
    """Non-selective adiabatic inversion, followed by an optional crusher.

    The module contains no inversion-time delay; the caller adds it. A pulse
    whose peak exceeds :data:`~pypulseqpp.MAX_B1_DERATE` of
    ``system.max_b1`` is lengthened at the same adiabaticity and swept
    bandwidth, up to ``ADIABATIC_MAX_STRETCH`` (2) times ``duration_s``, and
    past it held there with its adiabaticity lowered, to no less than
    ``ADIABATICITY_FLOOR`` (0.75) of ``adiabaticity``.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    duration_s : float, default=0.01
        Inversion pulse duration (s) while the pulse fits the system's peak
        B1; a shorter pulse needs a faster sweep, which may violate the
        adiabatic condition.
    spoiling_cycles : float, default=4.0
        Cycles of dephasing of the crusher across ``voxel_size_m``. Zero omits
        the crusher.
    voxel_size_m : float, default=0.001
        Length the dephasing is counted over (m); the smallest voxel
        dimension.
    axis : {'z', 'x', 'y'}, default='z'
        Gradient channel of the crusher.
    pulse_type : str, default='hypsec'
        Adiabatic sweep family.
    bandwidth_hz : float, default=40000.0
        Frequency width of the sweep (Hz); unused by ``'hypsec'``, which
        sweeps ``mu * beta`` (:func:`~pypulseqpp.make_adiabatic_pulse`).
    adiabaticity : float, default=4
        Sweep-rate margin over the adiabatic condition while the pulse fits
        the system's peak B1.
    labels : sequence of str, default=None
        Labels the inversion block sets to 0, to be changed per shot by the
        caller.

    Attributes
    ----------
    rf_prep : RfEvent
        The inversion pulse.
    pulse_duration : float
        Duration the pulse was designed at (s): ``duration_s``, or longer
        under the system's peak B1.
    adiabaticity : float
        Adiabaticity the pulse was designed at: the requested one, or lower
        under the system's peak B1.
    gz_spoil : GradEvent
        The crusher, when ``spoiling_cycles`` is nonzero.
    prep_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``, in order. Absent when ``labels`` is
        empty, and a bare event rather than a list when there is one.

    Raises
    ------
    ValueError
        If ``voxel_size_m`` is not positive, ``spoiling_cycles`` is negative,
        ``axis`` is not a gradient channel, or the pulse cannot fit the
        system's peak B1.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> prep = design.InversionPreparation(pp.Opts(), duration_s=8e-3)
    >>> len(prep.blocks)
    2

    >>> len(design.InversionPreparation(pp.Opts(), spoiling_cycles=0).blocks)
    1

    Under an 11.2 uT limit the 13.2 uT pulse is lengthened, at the same
    adiabaticity, to peak below 0.9 of it:

    >>> prep = design.InversionPreparation(pp.Opts(max_b1=11.2))
    >>> round(prep.pulse_duration * 1e3, 2), prep.adiabaticity
    (17.26, 4)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp.plot import plot_rf
       from pypulseqpp.sequences import InversionPreparation

       module = InversionPreparation(pp.Opts(), duration_s=8e-3)
       module.seq.paper_plot()
       plot_rf(module, plot_now=False)
       plt.show()
    """

    def init_module(
        self,
        system: pp.Opts,
        duration_s: float = 10e-3,
        *,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        axis: str = "z",
        pulse_type: str = "hypsec",
        bandwidth_hz: float = 40e3,
        adiabaticity: float = 4,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        def make(duration: float, margin: float):
            return pp.make_adiabatic_pulse(
                pulse_type=pulse_type,
                duration=duration,
                bandwidth=bandwidth_hz,
                adiabaticity=margin,
                use="inversion",
                system=system,
                **held_sweep(pulse_type, duration / duration_s),
            )

        self.pulse_duration, self.adiabaticity, rf_prep = adiabatic_fitted(
            system, duration_s, adiabaticity, make
        )
        # A slot per label, so the loop has somewhere to put one. An iteration
        # can leave it out or say something different with it, but it cannot
        # add a slot the module never built.
        prep_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep, *prep_labels)
        if spoiling_cycles:
            gz_spoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, axis, system=system
            )
            self.seq.add_block(gz_spoil)

        self.center = rf_reference(rf_prep)
