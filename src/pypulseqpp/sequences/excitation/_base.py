"""What every RF module has in common."""

from __future__ import annotations

__all__ = ["RfModule", "rf_reference"]

from typing import Any

from .._module import SequenceModule


def rf_reference(rf: Any) -> float:
    """Time from the start of an RF pulse's block to the point it is timed against.

    ``rf.center`` is measured from the first sample of the envelope, while a TE
    budget counts from the start of the block -- and the two differ by the
    pulse's own ``delay``, which is where a dead time or a selection gradient's
    ramp puts the envelope.
    """
    return float(rf.delay) + float(rf.center)


class RfModule(SequenceModule):
    """A module whose point is an RF pulse.

    Adds the one view the sequence-level analyses cannot give: what the pulse
    does to the magnetisation. Everything else -- publication, ``blocks``,
    ``duration``, plotting, k-space -- is :class:`~pypulseqpp.sequences.SequenceModule`'s.

    Examples
    --------
    Every module whose point is a pulse inherits :meth:`sim_rf`, so what the
    pulse does to the magnetisation is one call away wherever it comes from:

    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts()
    >>> for module in (
    ...     design.SpatialSelectiveExcitation(system, 15.0, 5e-3),
    ...     design.Inversion(system, 8e-3),
    ...     design.FatSaturation(system),
    ... ):
    ...     mz_z = module.sim_rf()[0]
    ...     print(isinstance(module, design.RfModule), mz_z.ndim)
    True 1
    True 1
    True 1

    """

    def sim_rf(self, pulse=None, **kwargs):
        """Simulate this module's pulse across off-resonance.

        Parameters
        ----------
        pulse : RfEvent, optional
            Which pulse to simulate. The default is the first one the module
            plays, found by type rather than by name -- an excitation calls its
            pulse ``rf`` and a preparation calls it ``rf_prep``, and neither
            spelling should have to be known here.
        **kwargs
            Forwarded to :func:`pypulseqpp.sim_rf` (``rephase_factor``,
            ``df``, ``bandwidth_multiplier``, ``dt``).

        Returns
        -------
        tuple
            MATLAB's six answers: ``mz_z``, ``mz_xy``, the frequency axis,
            ``ref_eff``, ``mx_xy`` and ``my_xy``.

        Examples
        --------
        >>> import pypulseqpp.sequences as design
        >>> import pypulseqpp as pp
        >>> pulse = design.SpatialSelectiveExcitation(pp.Opts(), 15.0, 5e-3)
        >>> mz_z, mz_xy, frequency = pulse.sim_rf()[:3]
        >>> mz_z.shape == frequency.shape
        True
        """
        import pypulseqpp as pp

        if pulse is None:
            pulse = next(
                event
                for block in self.blocks
                for event in block
                if getattr(event, "type", None) == "rf"
            )
        return pp.sim_rf(pulse, **kwargs)

    # Restored when Sequence draws again; the profile plotter it calls is
    # deferred alongside Sequence.plot.
    # def plot_rf(self, pulse=None, **kwargs):
    #     """Draw this module's pulse beside the profile it produces.
    #
    #     What :meth:`sim_rf` computes, as a figure: the ``B1`` envelope, and
    #     the magnetisation response beside it -- against position where the
    #     pulse is played under a gradient, against off-resonance where it is
    #     not, and as a pair of heatmaps when a plane is asked for.
    #
    #     Parameters
    #     ----------
    #     pulse : RfEvent, optional
    #         Which pulse to draw. The default is the first one the module
    #         plays, as for :meth:`sim_rf`.
    #     plane : str, optional
    #         What to simulate over: one of ``"x"``, ``"y"``, ``"z"`` or
    #         ``"f"`` for a profile along one axis, or two of them -- ``"zf"``,
    #         ``"xy"`` -- for a plane, drawn as ``|Mxy|`` and ``Mz`` heatmaps.
    #         The default is the axis the pulse is selective along, or ``"f"``
    #         when it is played under no gradient. **A pulse selective in two
    #         things at once needs a plane**: a spectral-spatial pulse's
    #         passband is a band in position *and* in frequency, and neither
    #         one-dimensional cut shows that.
    #     kind : str, optional
    #         Which response to draw: ``"excitation"``, ``"refocusing"``,
    #         ``"inversion"`` or ``"saturation"``. Read off the pulse's ``use``
    #         by default. Only for a one-dimensional profile; a plane always
    #         draws both components.
    #     extent, span : float or tuple of float, optional
    #         The first and second axes: a half-width about zero, or an explicit
    #         ``(low, high)``, in millimetres for a position and hertz for
    #         off-resonance. Wide enough to hold the pulse's own passband by
    #         default.
    #     samples : int, optional
    #         Points along each axis. A plane is simulated over ``samples**2``
    #         positions and caps at 91 a side, so this is what the figure costs.
    #     dt : float, optional
    #         Integration raster, in seconds, wherever the whole module is
    #         integrated rather than the single pulse.
    #     whole : bool, optional
    #         Integrate everything the module plays -- every pulse, every
    #         crusher, and the free precession between them -- rather than the
    #         one pulse alone. What a preparation module's profile means. A
    #         plane is always integrated this way.
    #     title : str, optional
    #         Figure title.
    #     plot_now : bool, optional
    #         Show the figure before returning.
    #
    #     Returns
    #     -------
    #     matplotlib.figure.Figure
    #
    #     Examples
    #     --------
    #     >>> import pypulseqpp.sequences as design
    #     >>> import pypulseqpp as pp
    #     >>> pulse = design.SpatialSelectiveExcitation(pp.Opts(), 15.0, 5e-3)
    #     >>> figure = pulse.plot_rf(plot_now=False)
    #     >>> [axis.get_title(loc="left") for axis in figure.axes]
    #     ['envelope', 'profile']
    #     """
    #     from ...pypulseq._rf_profile import plot_rf
    #
    #     return plot_rf(self, pulse, **kwargs)
