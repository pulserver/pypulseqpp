"""What the gradients ask of the amplifiers, and whether the scanner allows it.

Two limits bound every gradient a scanner will play: how strong it may be,
and how fast it may change. Neither is a property of one waveform -- three
axes play at once, and what an amplifier sees on its own axis depends on how
the sequence is rotated -- so both are asked of the block, not of the event.

The checks here take the sequence and, optionally, a different `Opts` from
the one it was designed against: a sequence written for one scanner is often
the question "will this run on that one", and answering it should not mean
building the sequence again.

Nothing here expands a waveform. A gradient is stored as a normalised shape
and one amplitude, so the steepest step in a waveform is a property of the
shape -- worked out once however many times it is played -- and what an
instance slews at is that step times its own amplitude. A readout repeated a
hundred thousand times at a hundred thousand amplitudes costs one pass over
its shape and a multiply per block.
"""

from __future__ import annotations

from types import SimpleNamespace

from .. import _ext as _cxx

__all__ = ["check_max_grad", "check_max_slew"]


def _limits(seq, system):
    """Return the limits to judge against: the ones given, or the sequence's."""
    chosen = system if system is not None else seq.system
    if chosen is None:
        raise ValueError(
            "no limits to judge against: build the Sequence with a system= "
            "argument, or pass one here"
        )
    return chosen


def _of(system, name, fallback=0.0):
    value = getattr(system, name, None)
    return fallback if value is None else float(value)


def check_max_grad(seq, system=None) -> tuple[bool, SimpleNamespace]:
    """Return whether every gradient is within the amplitude limit.

    Parameters
    ----------
    seq : Sequence
        The sequence to weigh.
    system : pypulseq.Opts, optional
        The scanner to weigh it against. Defaults to the one the sequence was
        built with, so asking whether a sequence will run somewhere else is
        passing that scanner here.

    Returns
    -------
    is_ok : bool
        True when no axis exceeds the limit.
    report : SimpleNamespace
        ``limit``; the ``per_axis`` and ``vector`` peaks; and ``axes``, the
        peak of each of x, y and z on its own. Each peak carries ``value`` in
        Hz/m, the 1-based ``block`` that plays it, and which ``axis``.

    Notes
    -----
    Two peaks are reported because two questions are being asked. A sequence
    played as written asks one amplifier for the per-axis peak, and that is
    what the limit bounds. A sequence played rotated can put the whole vector
    on one axis, so ``vector`` is what it would ask for then -- reported
    rather than judged, since whether it will be rotated is not something the
    sequence says.

    ``per_axis`` is the worst of ``axes``. The three are what says which
    amplifier is asked for what, which is the question once one of them is
    over its limit, and they cost nothing: the vector magnitude is built from
    them.

    What is weighed is the samples the sequence stores. An interpreter draws
    between them, and where a waveform's samples sit at the centre of each
    raster interval that drawing can pass a little outside the outermost of
    them -- by three parts in ten thousand across the reference sequences.
    Reading it off the stored amplitudes is what makes this a pass over a
    column rather than over every waveform in the scan.
    """
    limits = _limits(seq, system)
    limit = _of(limits, "max_grad")
    found = _cxx.max_gradient(seq._native)

    report = SimpleNamespace(
        limit=limit,
        per_axis=_peak(found["per_axis"]),
        vector=_peak(found["vector"]),
        axes=[_peak(peak) for peak in found["axes"]],
    )
    return (limit <= 0.0 or report.per_axis.value <= limit), report


def check_max_slew(seq, system=None) -> tuple[bool, SimpleNamespace]:
    """Return whether every gradient is within the slew limit, and continuous.

    Parameters
    ----------
    seq : Sequence
        The sequence to weigh.
    system : pypulseq.Opts, optional
        The scanner to weigh it against; the sequence's own by default.

    Returns
    -------
    is_ok : bool
        True when nothing slews too fast, nothing jumps, and the sequence
        leaves its gradients at zero.
    report : SimpleNamespace
        ``limit``; the ``per_axis`` and ``vector`` slew peaks in Hz/m/s;
        ``axes``, the peak of each of x, y and z on its own; the
        ``discontinuities`` found; and ``ends_at_zero``.

    Notes
    -----
    Two things can ask too much of an amplifier and they are not the same. A
    ramp that is too steep asks it within one event; a gradient that starts
    where the last one did not end asks it between two, in no time at all.
    The second is what `discontinuities` reports -- each naming the block, the
    axis, and what the gradient goes ``before`` and ``after`` the jump.

    A sequence that ends with a gradient still on has not ramped down, which
    is the same fault at the end of the scan and is reported as
    ``ends_at_zero``.
    """
    limits = _limits(seq, system)
    limit = _of(limits, "max_slew")
    raster = _of(limits, "grad_raster_time", 10e-6)
    found = _cxx.max_slew(seq._native, max_slew=limit, grad_raster_time=raster)

    report = SimpleNamespace(
        limit=limit,
        per_axis=_peak(found["per_axis"]),
        vector=_peak(found["vector"]),
        axes=[_peak(peak) for peak in found["axes"]],
        discontinuities=[SimpleNamespace(**jump) for jump in found["discontinuities"]],
        ends_at_zero=found["ends_at_zero"],
    )
    is_ok = (
        (limit <= 0.0 or report.per_axis.value <= limit)
        and not report.discontinuities
        and report.ends_at_zero
    )
    return is_ok, report


def _peak(found: dict) -> SimpleNamespace:
    """One peak, with the axis named rather than numbered."""
    axis = found["axis"]
    return SimpleNamespace(
        value=found["value"],
        block=found["block"],
        axis="xyz"[axis] if 0 <= axis < 3 else None,
    )
