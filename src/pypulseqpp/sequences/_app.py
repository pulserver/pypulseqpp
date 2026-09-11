"""Base class of complete sequences: events designed once, one repetition per call."""

from __future__ import annotations

import inspect
import typing
from abc import ABC, abstractmethod
from typing import Any

import pypulseqpp as pp

__all__ = ["SequenceApp"]

_MAIN_PARAMETERS = """\
plot : bool, optional
    Draw the finished sequence in SeqEyes.
test_report : bool, optional
    Print a report on the finished sequence.
write_seq : bool, optional
    Write the sequence to a .seq file.
seq_filename : str, optional
    Where to write it; ``<NAME>.seq`` when omitted.
system : pypulseqpp.Opts, optional
    System limits, held under the application's ``MAX_GRAD`` and ``MAX_SLEW``."""


class SequenceApp(ABC):
    """A complete sequence: its events and sampling, and the loop that plays them.

    Constructing an application designs it: ``init_sequence`` receives the
    prescription and builds the events and the sampling order. Nothing is
    played until :meth:`design`, which starts a new :attr:`seq`, runs
    :meth:`loop` -- the scan loop, calling :meth:`kernel` once per
    repetition -- and then :meth:`finalize`. Calling the application plays
    one kernel call, so a single repetition or a chunk of the scan is written
    the same way the whole scan is.

    Class attributes are the settings a user does not prescribe. A subclass
    that changes one, and nothing else, is the same sequence under a different
    setting::

        class GentleGre2D(Gre2DApp):
            MAX_SLEW = 120.0

    Attributes
    ----------
    MAX_GRAD, MAX_SLEW : float
        Gradient (mT/m) and slew (T/m/s) ceilings the sequence is held under,
        together with what ``system`` reports. Every concrete application sets
        both; there is no default.
    NAME : str
        Written as the ``Name`` definition and the default file name.
    system : pypulseqpp.Opts
        The limits the sequence was designed under.
    seq : pypulseqpp.Sequence
        What has been played so far.
    """

    MAX_GRAD: float
    MAX_SLEW: float
    NAME: str = "sequence"

    def __init__(self, system: pp.Opts | None = None, **protocol: Any) -> None:
        missing = [name for name in ("MAX_GRAD", "MAX_SLEW") if not hasattr(self, name)]
        if missing:
            raise TypeError(
                f"{type(self).__name__} sets no {' or '.join(missing)}: an application "
                "states the gradient limits it is designed under"
            )
        system = pp.Opts() if system is None else system
        self.system = pp.cap_system(
            system, max_grad=self.MAX_GRAD, max_slew=self.MAX_SLEW
        )
        self.seq = pp.Sequence(self.system)
        self._label_state: dict[str, int] = {}
        self._label_steps: dict[str, int] = {}
        self.init_sequence(**protocol)

    @abstractmethod
    def init_sequence(self, **protocol: Any) -> None:
        """Design the events and the sampling order from the prescription.

        The signature is the application's protocol: its keyword parameters,
        defaults and ``Parameters`` docstring are what :attr:`main`, the
        command line and :meth:`protocol` present.
        """

    @abstractmethod
    def kernel(self, *args: Any, **kwargs: Any) -> None:
        """Play one repetition into :attr:`seq`."""

    @abstractmethod
    def loop(self) -> None:
        """Play the whole scan, calling :meth:`kernel` once per repetition in play order."""

    def finalize(self) -> None:  # noqa: B027 -- an optional hook
        """Complete :attr:`seq` after the loop, e.g. with its definitions."""

    def __call__(self, *args: Any, **kwargs: Any) -> SequenceApp:
        self.kernel(*args, **kwargs)
        return self

    def design(self) -> pp.Sequence:
        """Play the whole scan into a new :attr:`seq`, finalize it and return it."""
        self.seq = pp.Sequence(self.system)
        self._label_state, self._label_steps = {}, {}
        self.loop()
        self.finalize()
        return self.seq

    def labels(self, **values: int) -> list:
        """Return the label events that bring each label to its new value.

        Label state is sticky in Pulseq, so an unchanged value writes nothing.
        A change equal to the label's previous change is an INC, which a scan
        repeats as one event; any other change is a SET. A change of ``ONCE``
        calls :meth:`restart_labels` first, so every label passed with it is
        written again as a SET.
        """
        once = values.get("ONCE")
        if once is not None and int(once) != self._label_state.get("ONCE"):
            self.restart_labels()
        events = []
        for name, value in values.items():
            value = int(value)
            last = self._label_state.get(name)
            if last == value:
                continue
            step = None if last is None else value - last
            if step is not None and step == self._label_steps.get(name):
                events.append(pp.make_label(name, "INC", step))
            else:
                events.append(pp.make_label(name, "SET", value))
            self._label_state[name] = value
            self._label_steps[name] = step
        return events

    def restart_labels(self) -> None:
        """Write every label's next value as a SET, whatever was written before.

        An interpreter repeating the scan plays ``ONCE`` blocks only once, so
        a value set inside them, or counted from one set there, is not there
        on the repeats. :meth:`labels` restarts at every ``ONCE`` it writes;
        call this where a module writes ``ONCE`` itself.
        """
        self._label_state.clear()
        self._label_steps.clear()

    @classmethod
    def protocol(cls) -> dict[str, Any]:
        """Return the prescription ``init_sequence`` accepts, with its defaults."""
        parameters = inspect.signature(cls.init_sequence).parameters
        return {name: p.default for name, p in parameters.items() if name != "self"}

    class _Main:
        """Module-level entry point of a concrete application.

        Builds the application from keyword arguments, designs it, and on
        request reports, plots and writes the sequence. Its parameters are
        ``plot``, ``test_report``, ``write_seq``, ``seq_filename``, ``system``
        and those of ``init_sequence``, which is what the command line reads.
        """

        def __get__(self, instance: Any, owner: type[SequenceApp]):
            if inspect.isabstract(owner):
                return self
            if "_main" not in owner.__dict__:
                owner._main = _make_main(owner)
            return owner._main

    main = _Main()


def _make_main(cls: type[SequenceApp]):
    """Build the module-level entry point of ``cls``, its protocol in the signature."""

    def main(
        plot: bool = False,
        test_report: bool = False,
        write_seq: bool = False,
        seq_filename: str | None = None,
        *,
        system: pp.Opts | None = None,
        **protocol: Any,
    ) -> pp.Sequence:
        seq = cls(system, **protocol).design()
        if test_report:
            print(seq.test_report())
        if plot:
            seq.plot()
        if write_seq:
            from pypulseqpp.cli import write_sequence

            write_sequence(seq, seq_filename or f"{cls.NAME}.seq")
        return seq

    hints = typing.get_type_hints(cls.init_sequence)
    own = list(inspect.signature(main).parameters.values())[:-1]
    prescribed = [
        parameter.replace(
            kind=parameter.KEYWORD_ONLY,
            annotation=hints.get(name, parameter.annotation),
        )
        for name, parameter in inspect.signature(cls.init_sequence).parameters.items()
        if name != "self"
    ]
    main.__signature__ = inspect.Signature(
        [*own, *prescribed], return_annotation=pp.Sequence
    )
    main.__annotations__ = {**typing.get_type_hints(main), **hints}
    main.__module__ = cls.__module__
    main.__qualname__ = "main"

    summary, _, rest = (inspect.getdoc(cls) or "").partition("\n\n")
    main.__doc__ = "\n\n".join(
        part
        for part in (
            summary,
            "Parameters\n----------\n"
            + "\n".join(
                filter(None, (_MAIN_PARAMETERS, _parameters(cls.init_sequence)))
            ),
            "Returns\n-------\npypulseqpp.Sequence\n    The designed sequence.",
            rest,
        )
        if part
    )
    return main


def _parameters(function: Any) -> str:
    """Return the entries of ``function``'s NumPy ``Parameters`` section, dedented."""
    lines = (inspect.getdoc(function) or "").splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.strip() == "Parameters" and set(lines[i + 1].strip()) == {"-"}:
            body = lines[i + 2 :]
            break
    else:
        return ""
    for j, line in enumerate(body[:-1]):
        if line and not line[0].isspace() and set(body[j + 1].strip()) == {"-"}:
            body = body[:j]
            break
    return "\n".join(body).strip()
