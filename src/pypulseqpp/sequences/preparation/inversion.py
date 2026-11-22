"""Inversion-recovery preparation."""

from __future__ import annotations

__all__ = ["InversionPreparation"]

import pypulseqpp as pp

from ..excitation._base import RfModule, adiabatic_fitted, held_sweep, rf_reference

_AXES = ("x", "y", "z")


class InversionPreparation(RfModule):
    """Adiabatic inversion, non-selective or of a slab, followed by an optional crusher.

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
    thickness_m : float, default=0.0
        Thickness of the inverted slab (m), selected by a gradient under the
        pulse; zero inverts the whole transmit volume.
    selection_axis : {'z', 'x', 'y'}, default='z'
        Logical axis the slab is selected along.
    offset_m : float, default=0.0
        Offset of the slab centre from the field-of-view centre along
        ``selection_axis`` (m).

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
    g_select : GradEvent
        The slab-selection gradient, when ``thickness_m`` is nonzero.
    prep_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``, in order. Absent when ``labels`` is
        empty, and a bare event rather than a list when there is one.

    Raises
    ------
    ValueError
        If ``voxel_size_m`` is not positive, ``spoiling_cycles`` or
        ``thickness_m`` is negative, ``axis`` or ``selection_axis`` is not a
        gradient channel, or the pulse cannot fit the system's peak B1.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> prep = design.InversionPreparation(pp.Opts(), duration_s=8e-3)
    >>> len(prep.blocks)
    2

    >>> len(design.InversionPreparation(pp.Opts(), spoiling_cycles=0).blocks)
    1

    A 10 cm slab along x, 5 cm off centre:

    >>> slab = design.InversionPreparation(
    ...     pp.Opts(), thickness_m=0.1, selection_axis="x", offset_m=0.05
    ... )
    >>> slab.g_select.channel
    'x'

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
        thickness_m: float = 0.0,
        selection_axis: str = "z",
        offset_m: float = 0.0,
    ) -> None:
        if thickness_m < 0:
            raise ValueError(f"thickness_m must be >= 0, got {thickness_m}")
        if selection_axis not in _AXES:
            raise ValueError(
                f"selection_axis must be one of {_AXES}, got {selection_axis!r}"
            )
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
                slice_thickness=thickness_m,
                return_gz=thickness_m > 0,
                **held_sweep(pulse_type, duration / duration_s),
            )

        self.pulse_duration, self.adiabaticity, designed = adiabatic_fitted(
            system, duration_s, adiabaticity, make
        )
        selection = []
        if thickness_m > 0:
            rf_prep, g_select, _ = designed
            g_select.channel = selection_axis
            rf_prep.freq_offset = g_select.amplitude * offset_m
            selection = [g_select]
        else:
            rf_prep = designed
        # A slot per label, so the loop has somewhere to put one. An iteration
        # can leave it out or say something different with it, but it cannot
        # add a slot the module never built.
        prep_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep, *selection, *prep_labels)
        if spoiling_cycles:
            gz_spoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, axis, system=system
            )
            self.seq.add_block(gz_spoil)

        self.center = rf_reference(rf_prep)
