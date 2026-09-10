"""The base class every reusable sequence module is built on."""

from __future__ import annotations

__all__ = ["SequenceModule"]

import sys
import warnings
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any

#: Analyses forwarded to the module's stored sequence.
SEQUENCE_VIEWS = (
    "calculate_kspace",
    "check_timing",
    "test_report",
    "waveforms_and_times",
)


def _unique(events: tuple) -> list:
    """Distinct objects in ``events``, by identity, in first-seen order."""
    seen: dict[int, Any] = {}
    for event in events:
        seen.setdefault(id(event), event)
    return list(seen.values())


class SequenceModule(ABC):
    """A reusable block layout with named, mutable event templates.

    Subclasses assign self.seq and add blocks in init_module. Played events
    are published from constructor locals onto events and the module itself.
    Repeated references are deduplicated by identity: one distinct object
    becomes a scalar event, several become a list in first-seen order.
    Explicit register calls preserve the supplied structure, including
    one-element lists.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to init_module.

    Attributes
    ----------
    events : types.SimpleNamespace
        Published events. Use this namespace when a name conflicts with a
        module attribute; publication warns about conflicts.
    center : float
        Timing reference in seconds from the module start, set by the
        subclass: typically an RF centre or an echo.

    Notes
    -----
    blocks retains the original event objects for replay. Mutating those
    objects does not rewrite the stored sequence used for analysis.
    Only calculate_kspace, check_timing, test_report and waveforms_and_times
    are forwarded to seq.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._seq = None
        self._blocks: list[tuple] = []
        self._played: dict[int, Any] = {}
        self._init_frame = None
        self._named: set[str] = set()
        self._mirrored: set[str] = set()
        self._duration: float | None = None
        self.events = SimpleNamespace()
        self.center = 0.0

        self.init_module(*args, **kwargs)

        self._finalize()

    @abstractmethod
    def init_module(self, *args: Any, **kwargs: Any) -> None:
        """Build the module: the part a subclass writes.

        Assign ``self.seq``, add blocks to it, and set :attr:`center` if the
        module is timed against something other than its own start. Events are
        published automatically; nothing has to be returned.
        """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @property
    def seq(self):
        """Stored sequence for the module's construction-time block layout.

        Assigning a Sequence enables block recording. Later event-template
        changes affect replay through blocks, not this stored sequence.
        """
        return self._seq

    @seq.setter
    def seq(self, sequence) -> None:
        from pypulseqpp import Sequence

        if not isinstance(sequence, Sequence):
            raise TypeError(
                f"a module's seq must be a pypulseqpp.Sequence, not {type(sequence).__name__}"
            )
        sequence.__class__ = _recording_sequence_class()
        sequence._module = self
        self._seq = sequence

    def _find_init_frame(self) -> None:
        """Retain all active init_module frames for this instance.

        Final locals from both subclass and base constructors are needed for
        event publication. Release the retained frames during finalisation.
        """
        targets = {
            getattr(init, "__code__", None)
            for init in (
                getattr(klass, "init_module", None) for klass in type(self).__mro__
            )
            if init is not None
        }
        frames = []
        frame = sys._getframe(2)
        while frame is not None:
            if frame.f_code in targets and frame.f_locals.get("self") is self:
                frames.append(frame)
            frame = frame.f_back
        self._init_frame = frames

    def _finalize(self) -> None:
        name = type(self).__name__
        if self._seq is None:
            raise TypeError(f"{name}.init_module never assigned self.seq")
        if not self._blocks:
            raise TypeError(f"{name}.init_module added no blocks to self.seq")

        # Innermost frame first, so that where a subclass and its base both
        # bind a name, the subclass's meaning is the one published.
        for frame in self._init_frame or ():
            self._publish_locals(frame.f_locals)
        self._init_frame = None  # the constructor and its locals are free to go

        if not vars(self.events):
            warnings.warn(
                f"{name} published no events: nothing it added to self.seq was named by a local "
                f"variable of {name}.init_module. Call self.publish() from wherever the events "
                "were built, or name them with self.register(name=event).",
                stacklevel=3,
            )

    def _publish_locals(self, namespace) -> None:
        for name, value in namespace.items():
            if name.startswith("_") or value is self:
                continue
            items = tuple(value) if isinstance(value, list | tuple) else (value,)
            played = tuple(item for item in items if id(item) in self._played)
            if not played:
                continue
            if name in self._named:
                continue
            self._set_event(name, played)

    def publish(self, **named: Any) -> None:
        """Publish played events from the caller's locals and register keyword aliases.

        Use in construction helpers whose locals are not captured automatically.
        Keyword aliases take precedence over automatic names.
        """
        self._publish_locals(sys._getframe(1).f_locals)
        self.register(**named)

    def register(self, **events: Any) -> None:
        """Publish named events without requiring prior block registration.

        Preserve the supplied container structure and override automatic
        publication for these names.
        """
        for name, event in events.items():
            self._publish(name, event)
            self._named.add(name)

    def _set_event(self, name: str, played: tuple) -> None:
        """Publish ``played`` under ``name``, collapsed to one object if it is one."""
        unique = _unique(played)
        self._publish(name, unique[0] if len(unique) == 1 else unique)

    def _publish(self, name: str, event: Any) -> None:
        """Mirror an event on the module and events, warning on name conflicts."""
        self._warn_if_shadowed(name)
        setattr(self.events, name, event)
        vars(self)[name] = event
        self._mirrored.add(name)

    def _warn_if_shadowed(self, name: str) -> None:
        if name in self._mirrored:
            return  # published before: this is a re-publication, not a clash
        if name in vars(self) or hasattr(type(self), name) or name in SEQUENCE_VIEWS:
            warnings.warn(
                f"{type(self).__name__} publishes an event called {name!r}, which is also a "
                f"module attribute; reach it as .events.{name}",
                stacklevel=4,
            )

    # ------------------------------------------------------------------
    # Reading the result
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """Resolve selected sequence analyses and otherwise-unresolved event names."""
        if name.startswith("_"):
            # Never route dunder or private lookups: copy, pickle and inspect
            # probe for those, and answering would answer for the module.
            raise AttributeError(name)
        if name in SEQUENCE_VIEWS:
            sequence = object.__getattribute__(self, "_seq")
            if sequence is not None:
                return getattr(sequence, name)
        try:
            events = object.__getattribute__(self, "events")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(events, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!r} module has no attribute or event {name!r}"
            ) from None

    def __dir__(self):
        return sorted(
            set(super().__dir__()) | set(vars(self.events)) | set(SEQUENCE_VIEWS)
        )

    @property
    def blocks(self) -> list[tuple]:
        """Return block tuples in play order, retaining the original event objects.

        The list is a copy; event mutations are shared with published templates.
        """
        return list(self._blocks)

    @property
    def duration(self) -> float:
        """Length of the module (s).

        Summed from the blocks, unless the module derived its timing
        analytically and assigned the answer.
        """
        if self._duration is not None:
            return self._duration
        return float(self._seq.duration()[0])

    @duration.setter
    def duration(self, value: float) -> None:
        self._duration = float(value)


# Construct lazily to avoid importing Sequence during package initialisation.
_RECORDING_SEQUENCE = None


def _recording_sequence_class():
    """Return a Sequence subclass that records event identities and block tuples."""
    global _RECORDING_SEQUENCE
    if _RECORDING_SEQUENCE is None:
        from pypulseqpp import Sequence

        class _ModuleSequence(Sequence):
            def add_block(self, *events):
                module = self._module
                if module._init_frame is None:
                    module._find_init_frame()
                module._blocks.append(events)
                for event in events:
                    module._played.setdefault(id(event), event)
                return super().add_block(*events)

        _RECORDING_SEQUENCE = _ModuleSequence
    return _RECORDING_SEQUENCE
