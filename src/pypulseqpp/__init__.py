"""Fast drop-in PyPulseq replacement over a C++ sequence core.

The contract is PyPulseq's own API: a design script written against
`pypulseq` runs here unchanged, with the same functions and the same
signatures. What differs is what each call does underneath. A `make_*`
factory hands back an event whose fields are in slots rather than in a
dictionary, `add_block` registers a whole block in one compiled call, and
reading and writing are C++.

## The namespace

Everything upstream exposes is re-exported, so `import pypulseqpp as pp`
is the only import a script needs. Every callable goes through
`_events.interoperating`, not only the factories: `calc_duration`,
`align`, `split_gradient` and `rotate` all take events, and upstream
implements them with `isinstance` checks and `deepcopy`, neither of which
a compiled event satisfies. The decorator hands them a namespace on the
way in and converts what comes back, so upstream's own helpers work
against events they were never written for.

## What is ours rather than upstream's

`Sequence` is ours outright: upstream's is a different implementation, and
this is the one with the compiled core under it.

Three factories, because upstream 1.5.0 does not have them: `make_rotation`
and `make_rf_shim`, which arrived with Pulseq 1.5.1, and `make_label`,
because upstream's refuses a name outside the list Pulseq defines and a
label here is named rather than numbered. Each is meant to go when
upstream grows its own.
"""

from __future__ import annotations

import functools as _functools
import inspect as _inspect
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

import pypulseq as _pypulseq

from . import _events
from ._events import as_namespace, convert, interoperating
from ._make_label import make_label as _make_label
from ._make_rf_shim import make_rf_shim as _make_rf_shim
from ._make_rotation import make_rotation as _make_rotation
from ._sequence import Sequence as _Sequence

try:
    __version__ = _distribution_version(__name__)
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0.dev0"

#: Upstream names that are not re-exported: the shape codec and the unit
#: helper, which upstream exposes as modules rather than as vocabulary.
_EXCLUDED = {"compress_shape", "convert", "decompress_shape"}

#: Upstream callables that must not be wrapped. Wrapping a class would
#: replace its constructor with a plain function.
_UNWRAPPED = {"SigpyPulseOpts"}

for _name in dir(_pypulseq):
    if _name.startswith("_") or _name in _EXCLUDED:
        continue
    _value = getattr(_pypulseq, _name)
    if _inspect.ismodule(_value):
        globals()[_name] = _value
        continue
    if callable(_value) and not isinstance(_value, type) and _name not in _UNWRAPPED:
        _value = interoperating(_value)
    globals()[_name] = _value
del _name, _value


def _fast_scale_grad(upstream):
    """`scale_grad` that stays in the slotted form for the common case.

    The hot one: a phase-encode loop scales the same prewinder once per
    shot, and the generic decorator would convert the event to a namespace
    and back on every call. A slotted gradient with no system to re-check
    against is scaled in place of that; anything else is upstream's.
    """
    scaled = _events.scaled_gradient

    @_functools.wraps(upstream)
    def scale_grad(grad, scale, system=None):
        if system is None:
            try:
                return scaled(grad, scale)
            except TypeError:
                pass
        return upstream(grad, scale, system=system)

    return scale_grad


scale_grad = _fast_scale_grad(globals()["scale_grad"])

# The factories, which hand back events with their fields in slots.
for _factory in _events.__all__:
    globals()[_factory] = getattr(_events, _factory)
del _factory

# Ours, until upstream has them.
make_label = _make_label
make_rf_shim = _make_rf_shim
make_rotation = _make_rotation

# Ours outright: upstream's Sequence is a different implementation, and
# this is the one with the compiled core under it.
Sequence = _Sequence

#: The events every one of these returns are compiled, not namespaces.
SLOTTED = frozenset(_events.__all__) | {"make_label", "make_rf_shim"}

__all__ = [
    "__version__",
    "SLOTTED",
    "as_namespace",
    "convert",
    *sorted(
        name
        for name in globals()
        if not name.startswith("_") and name not in {"annotations"}
    ),
]
