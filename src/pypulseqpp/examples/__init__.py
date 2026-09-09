"""The shipped sequences: one complete, self-contained script each.

A script is a worked sequence, written the way you would write one yourself.
It composes modules from :mod:`pypulseqpp.sequences` into an encoding plan and
loops over that plan writing blocks. Nothing here is machinery the rest of the
package depends on: read one, copy it, change it.

Every module is callable, and calling it builds its sequence::

    from pypulseqpp import examples

    seq = examples.gre2D_sequence(n_x=128, n_y=128, n_slices=1)
    seq.write("gre2d.seq")

That call is the module's ``main``, and it is the whole of its public surface,
so :func:`help` and :func:`inspect.signature` on the module answer for the
call you are about to make. Each is also a script::

    python -m pypulseqpp.examples.sequence.gre2D_sequence -o gre2d.seq --n-y 64

which is :func:`pypulseqpp.cli.run` reading that same signature.

One flat namespace: every entry is ``pypulseqpp.examples.<name>``, whichever
directory it is written in. Modules are imported on first use, so importing
this package costs nothing.
"""

from __future__ import annotations

import importlib as _importlib
import inspect as _inspect
from types import ModuleType as _ModuleType
from typing import Any

#: The directory the scripts are written in, mapped in beside this file so
#: each stays a standalone file someone can read and copy.
_FAMILY = "sequence"


def _names() -> list[str]:
    """Every script this package ships."""
    return sorted(_importlib.import_module(f"{__name__}.{_FAMILY}").__all__)


__all__ = _names()


class SequenceScript(_ModuleType):
    """A script, callable as the sequence it builds.

    A script *is* its ``main``, so the module carries ``main``'s docstring and
    signature rather than the file's: :func:`help` and
    :func:`inspect.signature` on the module answer for the call you are about
    to make.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.main(*args, **kwargs)


def _as_script(module: _ModuleType) -> _ModuleType:
    """Present ``module`` as the ``main`` it wraps."""
    main = getattr(module, "main", None)
    if main is None:
        return module
    module.__class__ = SequenceScript
    module.__doc__ = main.__doc__
    module.__signature__ = _inspect.signature(main)
    return module


def __getattr__(name: str):
    """Import one script on first use."""
    if name in __all__:
        return _as_script(_importlib.import_module(f"{__name__}.{_FAMILY}.{name}"))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the scripts this package ships."""
    return list(__all__)
