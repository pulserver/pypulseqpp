"""Deprecated spellings of renamed sampling routines.

Each wrapper emits a :class:`DeprecationWarning` naming its replacement and
then calls it. The names resolve as ``pypulseqpp.<name>`` but are not in
``__all__`` and are not documented in the API reference.
"""

from __future__ import annotations

import functools
import warnings

from . import _epi, _masks, _ordering

#: Old public name -> (new public name, replacement callable).
RENAMED = {
    "calc_sampled_lines": (
        "make_cartesian_axis_sampling",
        _masks.make_cartesian_axis_sampling,
    ),
    "calc_sampled_pairs": (
        "make_cartesian_plane_sampling",
        _masks.make_cartesian_plane_sampling,
    ),
    "calc_traversal_order": ("make_traversal_order", _ordering.make_traversal_order),
    "calc_epi_order": ("make_epi_shot_offsets", _epi.make_epi_shot_offsets),
}


def _warn(old: str) -> None:
    new = RENAMED[old][0]
    warnings.warn(
        f"pypulseqpp.{old} is deprecated and will be removed in a future "
        f"release; use pypulseqpp.{new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _renamed(old: str):
    """Return a wrapper that warns and forwards every argument unchanged."""
    new, target = RENAMED[old]

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        _warn(old)
        return target(*args, **kwargs)

    wrapper.__name__ = wrapper.__qualname__ = old
    wrapper.__doc__ = f"Call :func:`pypulseqpp.{new}` (deprecated alias)."
    wrapper.__wrapped__ = target
    return wrapper


def calc_sampled_lines(*args, r: int | None = None, **kwargs):
    """Call :func:`pypulseqpp.make_cartesian_axis_sampling` (deprecated alias).

    The former ``r`` argument is now ``acceleration``.
    """
    _warn("calc_sampled_lines")
    if r is not None:
        if "acceleration" in kwargs:
            raise TypeError("give either r or acceleration, not both")
        kwargs["acceleration"] = r
    return _masks.make_cartesian_axis_sampling(*args, **kwargs)


calc_sampled_lines.__wrapped__ = _masks.make_cartesian_axis_sampling
calc_traversal_order = _renamed("calc_traversal_order")
calc_epi_order = _renamed("calc_epi_order")


def calc_sampled_pairs(*args, shuffling: bool | None = None, **kwargs):
    """Call :func:`pypulseqpp.make_cartesian_plane_sampling` (deprecated alias).

    The former ``shuffling=True`` selected Poisson-disc support; it maps to
    ``sampling='poisson'``, and ``shuffling=False`` to ``sampling='lattice'``.
    """
    _warn("calc_sampled_pairs")
    if shuffling is not None:
        if "sampling" in kwargs:
            raise TypeError("give either shuffling or sampling, not both")
        kwargs["sampling"] = "poisson" if shuffling else "lattice"
    return _masks.make_cartesian_plane_sampling(*args, **kwargs)


calc_sampled_pairs.__wrapped__ = _masks.make_cartesian_plane_sampling
