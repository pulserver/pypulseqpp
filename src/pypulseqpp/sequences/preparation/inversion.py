"""Inversion-recovery preparation."""

from __future__ import annotations

__all__ = [
    "InversionPreparation",
    "InversionSlots",
    "inversion_slots",
    "shortest_inversion_slot",
]

import dataclasses
import math

import pypulseqpp as pp

from ..excitation._base import RfModule, adiabatic_fitted, held_sweep, rf_reference

_AXES = ("x", "y", "z")


class InversionPreparation(RfModule):
    """Adiabatic inversion, non-selective or of one slab, followed by an optional crusher.

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
        Gradient channel of the crusher and of the slab selection.
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
    slab_thickness_m : float, default=None
        Thickness of the slab the pulse inverts (m), selected on the
        ``axis`` channel under a gradient that holds the swept bandwidth
        across it; ``None`` inverts everything the coil reaches. The caller
        moves the slab by the pulse's frequency offset, the amplitude of
        ``gz_prep`` times the slab centre.

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
    gz_prep : TrapEvent or GradEvent
        The slab-selection gradient, when ``slab_thickness_m`` is set.
    gz_spoil : GradEvent
        The crusher, when ``spoiling_cycles`` is nonzero.
    prep_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``, in order. Absent when ``labels`` is
        empty, and a bare event rather than a list when there is one.

    Raises
    ------
    ValueError
        If ``voxel_size_m`` or ``slab_thickness_m`` is not positive,
        ``spoiling_cycles`` is negative,
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

    An 8 mm slab, swept by a GOIA-WURST pulse under its modulated gradient:

    >>> prep = design.InversionPreparation(
    ...     pp.Opts(), slab_thickness_m=8e-3, pulse_type="goia_wurst", bandwidth_hz=4e3
    ... )
    >>> prep.gz_prep.channel
    'z'

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
        slab_thickness_m: float | None = None,
    ) -> None:
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if slab_thickness_m is not None and slab_thickness_m <= 0:
            raise ValueError("slab_thickness_m must be positive")
        selective = slab_thickness_m is not None
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
                return_gz=selective,
                slice_thickness=slab_thickness_m or 0.0,
                **held_sweep(pulse_type, duration / duration_s),
            )

        self.pulse_duration, self.adiabaticity, designed = adiabatic_fitted(
            system, duration_s, adiabaticity, make
        )
        rf_prep = designed[0] if selective else designed
        # A slot per label, so the loop has somewhere to put one. An iteration
        # can leave it out or say something different with it, but it cannot
        # add a slot the module never built.
        prep_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        if selective:
            # The slab gradient plays on the crusher's channel; the
            # rephaser is left out, as no transverse magnetisation is kept.
            gz_prep = designed[1]
            gz_prep.channel = axis
            self.seq.add_block(rf_prep, gz_prep, *prep_labels)
        else:
            self.seq.add_block(rf_prep, *prep_labels)
        if spoiling_cycles:
            gz_spoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, axis, system=system
            )
            self.seq.add_block(gz_spoil)

        self.center = rf_reference(rf_prep)


@dataclasses.dataclass(frozen=True)
class InversionSlots:
    """Where interleaved inversions sit among the shots of a multislice scan.

    Every slot is ``slot`` long: a shot, its closing delay ``closing``, the
    inversion of the slice excited ``lag`` slots later, and ``tail``. The
    inversion centre precedes that excitation centre by ``ti``.
    """

    slot: float
    closing: float
    tail: float
    lag: int
    ti: float


def inversion_slots(
    ti: float,
    slot: float,
    shot: float,
    excitation: float,
    inversion: InversionPreparation,
    raster: float,
) -> InversionSlots:
    """Place an inversion in each slot of a multislice scan, ``ti`` before a later slice's excitation.

    A slot holds one shot and then one inversion, of the slice whose shot
    comes ``lag`` slots later, so the inversions of later slices sit between
    the shots of earlier ones (interleaved inversion recovery).

    Parameters
    ----------
    ti : float
        Inversion time (s), from the inversion centre to the excitation
        centre.
    slot : float
        Time from one shot to the next (s).
    shot : float
        Duration of a shot without its closing delay (s).
    excitation : float
        Time from the start of a shot to the centre of its excitation (s).
    inversion : InversionPreparation
        The inversion each slot plays.
    raster : float
        Block duration raster (s).

    Returns
    -------
    InversionSlots
        The slot layout.

    Raises
    ------
    ValueError
        If the inversion would overlap a shot; the message names the nearest
        inversion times that fit.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> from pypulseqpp import sequences
    >>> prep = sequences.InversionPreparation(pp.Opts(), slab_thickness_m=8e-3)
    >>> plan = sequences.inversion_slots(0.52, 0.1, 0.05, 0.01, prep, 1e-5)
    >>> plan.lag, round(plan.ti, 4)
    (6, 0.52)
    """
    before = excitation - ti - inversion.center
    lag = -math.floor(before / slot)
    start = pp.round_to_raster(before + lag * slot, raster)
    tail = pp.round_to_raster(slot - start - inversion.duration, raster)
    if start - shot < raster - 1e-12 or tail < -1e-12:
        # The inversion times nearest ti whose inversion starts up to 0.1 ms
        # inside either end of the free part of a slot, so they fit as printed.
        inset = min(1e-4, max(slot - inversion.duration - shot - raster, 0.0) / 2)
        nearest = sorted(
            excitation - inversion.center - free + n * slot
            for free in (shot + raster + inset, slot - inversion.duration - inset)
            for n in (lag - 1, lag, lag + 1)
        )
        below = max((t for t in nearest if t < ti), default=None)
        above = min(t for t in nearest if t > ti)
        raise ValueError(
            f"an inversion {ti * 1e3:.1f} ms before an excitation overlaps a "
            f"shot at {slot * 1e3:.1f} ms per slice; the nearest inversion "
            f"times that fit are "
            + (f"{below * 1e3:.1f} ms and " if below and below > 0 else "")
            + f"{above * 1e3:.1f} ms, or the TR can change"
        )
    achieved = lag * slot + excitation - start - inversion.center
    return InversionSlots(slot, start - shot, max(tail, 0.0), lag, achieved)


def shortest_inversion_slot(
    ti: float,
    minimum: float,
    shot: float,
    excitation: float,
    inversion: InversionPreparation,
    raster: float,
) -> float:
    """Return the shortest slot, at least ``minimum``, that places an interleaved inversion ``ti`` before an excitation.

    Parameters
    ----------
    ti : float
        Inversion time (s), from the inversion centre to the excitation
        centre.
    minimum : float
        Shortest slot allowed (s), on the block raster.
    shot, excitation, inversion, raster
        As for :func:`inversion_slots`.

    Returns
    -------
    float
        Slot duration (s), on the block raster, that :func:`inversion_slots`
        accepts.

    Raises
    ------
    ValueError
        If ``ti`` is shorter than the inversion and the shot leave room for.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> from pypulseqpp import sequences
    >>> prep = sequences.InversionPreparation(pp.Opts(), slab_thickness_m=8e-3)
    >>> slot = sequences.shortest_inversion_slot(0.8, 0.07, 0.05, 0.01, prep, 1e-5)
    >>> slot >= 0.07, sequences.inversion_slots(0.8, slot, 0.05, 0.01, prep, 1e-5).lag
    (True, 12)
    """
    before = excitation - ti - inversion.center
    earliest = shot + raster - before
    latest = -before - inversion.duration
    if latest < 0:
        raise ValueError(
            f"an inversion time of {ti * 1e3:.1f} ms is shorter than the "
            f"{(excitation - inversion.center + inversion.duration) * 1e3:.1f} ms "
            "the inversion and the start of the shot take"
        )
    best = math.inf
    for lag in range(1, math.ceil(earliest / minimum) + 2):
        low = max(minimum, earliest / lag)
        low = math.ceil(low / raster - 1e-9) * raster
        if lag == 1 or low <= latest / (lag - 1) + 1e-12:
            best = min(best, low)
    for step in range(100):
        slot = pp.round_to_raster(best + step * raster, raster)
        try:
            inversion_slots(ti, slot, shot, excitation, inversion, raster)
        except ValueError:
            continue
        return slot
    raise ValueError(
        f"no slot near {best * 1e3:.3f} ms fits an inversion time of {ti * 1e3:.1f} ms"
    )
