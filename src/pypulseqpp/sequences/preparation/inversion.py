"""Inversion-recovery preparation."""

from __future__ import annotations

__all__ = ["InversionPreparation"]

import pypulseqpp as pp

from ..excitation._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class InversionPreparation(RfModule):
    """Non-selective adiabatic inversion, followed by an optional crusher.

    The acquisition loop supplies the inversion-time delay.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    duration_s : float, default=0.01
        Inversion pulse duration (s). Shorter durations increase the required
        sweep rate and may violate the adiabatic condition.
    spoiling_cycles : float, default=4.0
        Dephasing of the crusher, in cycles across ``voxel_size_m``. Zero
        omits the crusher.
    voxel_size_m : float, default=0.001
        Length the dephasing is counted over (m) — the smallest voxel
        dimension, since that is the one that has to be spoiled.
    axis : {'z', 'x', 'y'}, default='z'
        Crusher axis.
    pulse_type : str, default='hypsec'
        Adiabatic sweep family.
    bandwidth_hz : float, default=40000.0
        Frequency width of the sweep (Hz).
    adiabaticity : int, default=4
        Sweep-rate margin over the adiabatic condition.
    labels : sequence of str, default=None
        Counters emitted on the inversion block, which starts each shot. The
        module creates the label events; the acquisition loop assigns their
        per-shot values.

    Attributes
    ----------
    rf_prep : RfEvent
        The inversion pulse.
    gz_spoil : GradEvent
        The crusher, when ``spoiling_cycles`` is nonzero.
    prep_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``, in order. Absent when ``labels`` is
        empty, and a bare event rather than a list when there is one.

    Raises
    ------
    ValueError
        If ``voxel_size_m`` is not positive, ``spoiling_cycles`` is negative,
        or ``axis`` is not a gradient channel.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> prep = design.InversionPreparation(pp.Opts(), duration_s=8e-3)
    >>> len(prep.blocks)
    2

    >>> len(design.InversionPreparation(pp.Opts(), spoiling_cycles=0).blocks)
    1

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
        adiabaticity: int = 4,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf_prep = pp.make_adiabatic_pulse(
            pulse_type=pulse_type,
            duration=duration_s,
            bandwidth=bandwidth_hz,
            adiabaticity=adiabaticity,
            use="inversion",
            system=system,
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
