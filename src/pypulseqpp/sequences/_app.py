"""Base class of a complete sequence, designed once and played one repetition at a time."""

from __future__ import annotations

import ast
import inspect
import re
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypulseqpp as pp
from pypulseqpp._prescription import accepts_none, documented, scalar

from ._labels import Labels

__all__ = ["ProtocolParameter", "SequenceApp", "parameters"]

_SYSTEM_PARAMETER = """\
system : pypulseqpp.Opts, default=None
    System limits, held under the application's ``MAX_GRAD`` and ``MAX_SLEW``."""

_FUNCTION_RETURNS = """\
pypulseqpp.Sequence or list of pypulseqpp.Sequence
    The designed sequence. An application with prescans returns the list of
    the designed prescans in play order, then the main sequence."""

_MAIN_PARAMETERS = f"""\
plot : bool, default=False
    Draw the finished sequence in SeqEyes.
test_report : bool, default=False
    Print a report on the finished sequence.
write_seq : bool, default=False
    Write the sequence to a .seq file.
seq_filename : str, default=None
    Where to write it; ``<NAME>.seq`` when omitted.
{_SYSTEM_PARAMETER}"""


@dataclass(frozen=True)
class ProtocolParameter:
    """One protocol parameter, as a sequence function or ``init_sequence`` declares and documents it."""

    #: The keyword the sequence function or ``init_sequence`` takes.
    name: str
    #: The scalar the annotation names, bool, int, float or str; None for any
    #: other annotation.
    type: type | None
    #: The default; `inspect.Parameter.empty` for a parameter without one.
    default: Any
    #: Whether the annotation admits None, which leaves the value to the design.
    optional: bool
    #: The first parenthesised group of the description's first sentence, the
    #: form the shipped applications state units in, as in ``Echo time (s).``;
    #: empty where there is none.
    unit: str
    #: The values the documented type lists in braces, as in
    #: ``{'slab', 'nonselective'}``, in the order listed; empty otherwise.
    choices: tuple[Any, ...]
    #: The description in the Parameters section, on one line.
    description: str


def _unit(description: str) -> str:
    first = re.split(r"(?<=\.)\s", description, maxsplit=1)[0]
    group = re.search(r"\(([^()]+)\)", first)
    return group.group(1) if group else ""


def _choices(kind: str) -> tuple[Any, ...]:
    listed = re.match(r"\{(.*?)\}", kind)
    try:
        return tuple(ast.literal_eval(f"[{listed.group(1)}]")) if listed else ()
    except (ValueError, SyntaxError):  # braces around names, not values
        return ()


class SequenceApp(ABC):
    """A complete sequence, designed from a prescription and played one repetition at a time.

    Constructing an application designs it: ``init_sequence`` receives the
    prescription and builds the events and the sampling order. Nothing is
    played until :meth:`design`, which starts a new :attr:`seq`, runs
    :meth:`loop` -- the scan loop, calling :meth:`kernel` once per
    repetition -- and then :meth:`finalize`. Calling the application plays
    one kernel call, so a single repetition or a chunk of the scan is written
    the same way the whole scan is.

    Construction therefore checks a prescription: ``init_sequence`` raises
    when the prescription cannot be designed. It records the values the
    design chooses or adjusts with :meth:`resolve`, and the scan time as
    :attr:`duration`, so that :attr:`resolved` and :meth:`scan_time` are
    available without playing the loop.

    Class attributes are the settings a user does not prescribe. A subclass
    that changes one, and nothing else, is the same sequence under a different
    setting::

        class GentlePulseTrain(HardPulseTrain):
            MAX_SLEW = 100.0

    Attributes
    ----------
    MAX_GRAD, MAX_SLEW : float
        Gradient (mT/m) and slew (T/m/s) ceilings the sequence is held under,
        together with what ``system`` reports. Every concrete application sets
        both; there is no default.
    system : pypulseqpp.Opts
        The limits the sequence was designed under.
    seq : pypulseqpp.Sequence
        What has been played so far.

    Examples
    --------
    A subclass states its ceilings, designs from the prescription in
    ``init_sequence``, plays the scan in ``loop`` and records the definitions a
    reconstruction reads in ``finalize``:

    >>> import pypulseqpp as pp
    >>> import pypulseqpp.sequences as design
    >>> class HardPulseTrain(design.SequenceApp):
    ...     MAX_GRAD = 40.0
    ...     MAX_SLEW = 150.0
    ...     NAME = "hard_pulse_train"
    ...     def init_sequence(self, n_lines: int = 4):
    ...         self.excitation = design.NonSelectiveExcitation(self.system, 10.0, 0.5e-3)
    ...         self.n_lines = n_lines
    ...     def loop(self):
    ...         for line in range(self.n_lines):
    ...             self.kernel(line)
    ...     def kernel(self, line):
    ...         self.seq.add_block(self.excitation.rf, *self.labels(LIN=line))
    ...     def finalize(self):
    ...         self.seq.set_definition("Name", self.NAME)
    >>> seq = HardPulseTrain(n_lines=4).design()
    >>> seq.num_blocks, seq.get_definition("Name")
    (4, 'hard_pulse_train')
    """

    MAX_GRAD: float
    MAX_SLEW: float
    #: Written as the ``Name`` definition and the default file name.
    NAME: str = "sequence"
    #: Time the whole chain of prescans and main sequence plays, in seconds,
    #: when ``init_sequence`` computes it; None otherwise.
    duration: float | None = None

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
        self._labels = Labels()
        self._requested = dict(protocol)
        self._resolved: dict[str, Any] = {}
        self.init_sequence(**protocol)

    @abstractmethod
    def init_sequence(self, **protocol: Any) -> None:
        """Design the events and the sampling order from the prescription.

        The signature is the application's protocol: its keyword parameters,
        defaults and ``Parameters`` docstring are what :attr:`main`, the
        command line and :meth:`protocol` present.

        Parameters
        ----------
        **protocol : object
            The prescription, which the subclass names in its own signature.
        """

    @abstractmethod
    def kernel(self, *args: Any, **kwargs: Any) -> None:
        """Add the blocks of one repetition to :attr:`seq`.

        Parameters
        ----------
        *args, **kwargs : object
            What this repetition is, which the subclass names in its own
            signature; :meth:`loop` passes one repetition's worth per call.
        """

    @abstractmethod
    def loop(self) -> None:
        """Run the whole scan, calling :meth:`kernel` once per repetition in play order."""

    def finalize(self) -> None:  # noqa: B027 -- an optional hook
        """Complete :attr:`seq` after the loop, for example with its definitions."""

    def __call__(self, *args: Any, **kwargs: Any) -> SequenceApp:
        self.kernel(*args, **kwargs)
        return self

    def prescans(self) -> dict[str, Callable[[], None]]:
        """Return the prescans played before this sequence, as the loops that play them.

        :meth:`write` writes each as its own file ahead of the main one, in
        order, and each file names the next as its ``NextSequence``: an
        interpreter plays the chain as one scan while every file stays one
        repeating unit. A prescan loop writes its own definitions. None by
        default.

        Returns
        -------
        dict of {str: callable}
            One loop per prescan, by the name its file is given, in play
            order.
        """
        return {}

    def design(self, prescan: str | None = None) -> pp.Sequence:
        """Build the whole scan, or the named prescan, into a new :attr:`seq`.

        Parameters
        ----------
        prescan : str, optional
            The prescan to build, named as :meth:`prescans` lists it. The
            default builds the main sequence: :meth:`loop`, then
            :meth:`finalize`.

        Returns
        -------
        pypulseqpp.Sequence
            The sequence that was built, which is also :attr:`seq`.

        Raises
        ------
        KeyError
            If ``prescan`` is not one of the names :meth:`prescans` lists.
        """
        self.seq = pp.Sequence(self.system)
        self.restart_labels()
        if prescan is None:
            self.loop()
            self.finalize()
        else:
            self.prescans()[prescan]()
        return self.seq

    def write(self, path: str | Path, *, offline: bool = True) -> list[str]:
        """Design and write the chain of prescans and the main sequence.

        The first file of the chain is written at ``path`` and the others
        beside it as ``<stem>_<prescan>.seq`` and ``<stem>_main.seq``; without
        prescans the main sequence alone is written at ``path``. ``offline``
        selects signed text, and False the binary form, which
        :func:`pypulseqpp.io.write` takes the other way round.

        Parameters
        ----------
        path : str or pathlib.Path
            Where the first file of the chain is written.
        offline : bool, default=True
            Write the text form. False writes the binary form.

        Returns
        -------
        list of str
            The written paths, in play order.
        """
        from pypulseqpp.io import write

        path = Path(path)
        names = [*self.prescans(), None]
        paths = [path] + [
            path.with_name(f"{path.stem}_{name or 'main'}.seq") for name in names[1:]
        ]
        for i, name in enumerate(names):
            seq = self.design(name)
            if i + 1 < len(names):
                seq.set_definition(key="NextSequence", value=paths[i + 1].name)
            write(seq, str(paths[i]), binary=not offline)
        return [str(p) for p in paths]

    def labels(self, **values: int) -> list:
        """Return the label events that set each label to its new value.

        Label state is sticky in Pulseq, so an unchanged value writes nothing.
        A change equal to the label's previous change is an INC, which a scan
        repeats as one event; any other change is a SET. A change of ``ONCE``
        calls :meth:`restart_labels` first, so every label passed with it is
        written again as a SET.

        Parameters
        ----------
        **values : int
            The new value of each label, by its Pulseq name.

        Returns
        -------
        list
            The label events to add to the block, empty when nothing changed.
        """
        return self._labels(**values)

    def restart_labels(self) -> None:
        """Write every label's next value as a SET, regardless of what was written before.

        An interpreter repeating the scan plays ``ONCE`` blocks only once, so
        a value set inside them, or counted from one set there, is not there
        on the repeats. :meth:`labels` restarts at every ``ONCE`` it writes;
        call this where a module writes ``ONCE`` itself.
        """
        self._labels.restart()

    @classmethod
    def protocol(cls) -> dict[str, Any]:
        """Return the prescription ``init_sequence`` accepts, with its defaults.

        Returns
        -------
        dict
            The default of each prescribed parameter, by its name;
            `inspect.Parameter.empty` where there is none.
        """
        parameters = inspect.signature(cls.init_sequence).parameters
        return {name: p.default for name, p in parameters.items() if name != "self"}

    @classmethod
    def parameters(cls) -> dict[str, ProtocolParameter]:
        """Return each parameter of ``init_sequence`` with its type, default, unit and description.

        Returns
        -------
        dict of {str: ProtocolParameter}
            One entry per parameter a prescription names, in signature order.

        Examples
        --------
        >>> from pypulseqpp import sequences
        >>> class Pause(sequences.SequenceApp):
        ...     MAX_GRAD, MAX_SLEW = 40.0, 150.0
        ...     def init_sequence(self, tr: float | None = None):
        ...         '''Design a pause.
        ...
        ...         Parameters
        ...         ----------
        ...         tr : float | None, default=None
        ...             Repetition time (s). ``None`` is as short as possible.
        ...         '''
        ...     def loop(self): ...
        ...     def kernel(self): ...
        >>> tr = Pause.parameters()["tr"]
        >>> tr.type, tr.default, tr.optional, tr.unit
        (<class 'float'>, None, True, 's')
        >>> tr.description
        'Repetition time (s). ``None`` is as short as possible.'
        """
        return parameters(cls.init_sequence)

    def resolve(self, **values: Any) -> None:
        """Record the value a prescribed parameter took in the design.

        Called from ``init_sequence`` for a parameter whose value the design
        chooses, such as the shortest echo time for ``te=None``, or adjusts,
        such as a receiver bandwidth whose dwell time is rounded to the ADC
        raster. :attr:`resolved` reports the recorded value in place of the
        requested one.

        Parameters
        ----------
        **values : object
            The value each parameter took, by name, in the unit it is
            prescribed in.

        Raises
        ------
        TypeError
            If a name is not a parameter of ``init_sequence``.

        Examples
        --------
        >>> from pypulseqpp import sequences
        >>> class Pause(sequences.SequenceApp):
        ...     MAX_GRAD, MAX_SLEW = 40.0, 150.0
        ...     def init_sequence(self, tr: float | None = None):
        ...         self.tr = 10e-3 if tr is None else tr
        ...         self.resolve(tr=self.tr)
        ...     def loop(self): ...
        ...     def kernel(self): ...
        >>> Pause().resolved, Pause(tr=20e-3).resolved
        ({'tr': 0.01}, {'tr': 0.02})
        """
        unknown = sorted(set(values) - set(_prescribed(type(self).init_sequence)))
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.init_sequence has no parameter "
                f"{', '.join(unknown)}"
            )
        self._resolved.update(values)

    @property
    def resolved(self) -> dict[str, Any]:
        """The prescription as designed, by parameter name, in each parameter's prescribed unit.

        A parameter takes the value recorded with :meth:`resolve`, otherwise
        the requested value, otherwise its default.
        """
        values = {**self._requested, **self._resolved}
        return {
            name: values.get(name, parameter.default)
            for name, parameter in _prescribed(type(self).init_sequence).items()
        }

    def scan_time(self) -> float:
        """Return the time the whole chain of prescans and main sequence plays, in seconds.

        This is :attr:`duration` when ``init_sequence`` sets it, and nothing
        is designed; a stated duration can include waits for a trigger, which
        the blocks of a file do not time. Otherwise every file of the chain is
        designed, as :meth:`write` designs them, which leaves the main
        sequence in :attr:`seq`.

        Returns
        -------
        float
            The summed duration of the files of the chain, in seconds.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> from pypulseqpp import sequences
        >>> class Pause(sequences.SequenceApp):
        ...     MAX_GRAD, MAX_SLEW = 40.0, 150.0
        ...     def init_sequence(self, n: int = 4):
        ...         self.n = n
        ...         self.duration = n * 10e-3
        ...     def loop(self):
        ...         for _ in range(self.n):
        ...             self.kernel()
        ...     def kernel(self):
        ...         self.seq.add_block(pp.make_delay(10e-3))
        >>> app = Pause(n=4)
        >>> app.scan_time()
        0.04
        >>> round(app.design().duration()[0], 9)
        0.04
        """
        if self.duration is not None:
            return float(self.duration)
        names = [*self.prescans(), None]
        return float(sum(self.design(name).duration()[0] for name in names))

    @classmethod
    def function(cls) -> Callable[..., pp.Sequence | list[pp.Sequence]]:
        """Return the application as a sequence function, ``function(system, **protocol)``.

        The function constructs the application from its arguments and returns
        :meth:`design`. An application with prescans returns the list of its
        designed prescans, in the order :meth:`prescans` lists them, followed
        by the main sequence; :func:`pypulseqpp.sequences.write` writes that
        list as the chain of linked files :meth:`write` writes. The signature
        is ``system=None`` followed by the keyword parameters of
        ``init_sequence``, and the documentation is the class's with the
        ``Parameters`` of ``init_sequence``, so that
        :func:`pypulseqpp.sequences.parameters` and :func:`pypulseqpp.cli.run`
        read the protocol from the function as they read it from the class.

        Returns
        -------
        callable
            The sequence function.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> from pypulseqpp import sequences
        >>> class Delays(sequences.SequenceApp):
        ...     MAX_GRAD, MAX_SLEW = 40.0, 150.0
        ...     def init_sequence(self, n: int = 2):
        ...         self.n = n
        ...     def loop(self):
        ...         for _ in range(self.n):
        ...             self.kernel()
        ...     def kernel(self):
        ...         self.seq.add_block(pp.make_delay(1e-3))
        ...     def finalize(self):
        ...         self.seq.set_definition("Name", "delays")
        >>> delays = Delays.function()
        >>> delays(n=3).num_blocks
        3

        An application with prescans returns its chain:

        >>> class PrimedDelays(Delays):
        ...     def prescans(self):
        ...         def reference():
        ...             self.seq.add_block(pp.make_delay(2e-3))
        ...             self.seq.set_definition("Name", "reference")
        ...         return {"reference": reference}
        >>> primed = PrimedDelays.function()
        >>> [seq.get_definition("Name") for seq in primed(n=3)]
        ['reference', 'delays']
        """

        def sequence(system: pp.Opts | None = None, **protocol: Any):
            app = cls(system, **protocol)
            chain = [app.design(name) for name in app.prescans()]
            return [*chain, app.design()] if chain else app.design()

        hints = typing.get_type_hints(cls.init_sequence)
        parameters = [
            inspect.Parameter(
                "system",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=pp.Opts | None,
            ),
            *(
                parameter.replace(
                    kind=parameter.KEYWORD_ONLY,
                    annotation=hints.get(name, parameter.annotation),
                )
                for name, parameter in _prescribed(cls.init_sequence).items()
            ),
        ]
        returns = pp.Sequence | list[pp.Sequence]
        sequence.__signature__ = inspect.Signature(
            parameters, return_annotation=returns
        )
        sequence.__annotations__ = {
            **{parameter.name: parameter.annotation for parameter in parameters},
            "return": returns,
        }
        sequence.__doc__ = _documentation(cls, _SYSTEM_PARAMETER, _FUNCTION_RETURNS)
        return sequence

    class _Main:
        """Module-level entry point of a concrete sequence implementation.

        Builds the application from keyword arguments, designs it, and on
        request reports, plots and writes the sequence. Its parameters are
        ``plot``, ``test_report``, ``write_seq``, ``seq_filename``, ``system``
        and those of ``init_sequence``; the command line derives its flags
        from this signature.
        """

        def __get__(self, instance: Any, owner: type[SequenceApp]):
            if inspect.isabstract(owner):
                return self
            if "_main" not in owner.__dict__:
                owner._main = _make_main(owner)
            return owner._main

    main = _Main()


def _prescribed(function: Callable[..., Any]) -> dict[str, inspect.Parameter]:
    """Return the parameters a protocol names: those after the first, which is ``system`` or ``self``."""
    return {
        name: p
        for name, p in list(inspect.signature(function).parameters.items())[1:]
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


def parameters(function: Callable[..., Any]) -> dict[str, ProtocolParameter]:
    """Return each protocol parameter of a sequence function with its type, default, unit and description.

    The protocol is every parameter after the first, which is ``system``. Its
    type, default and whether it admits None are read from the annotation and
    the default; its unit, choices and description from the NumPy
    ``Parameters`` section of the docstring.

    Parameters
    ----------
    function : callable
        A sequence function ``function(system, **protocol)``.

    Returns
    -------
    dict[str, ProtocolParameter]
        One entry per protocol parameter, in signature order.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> def gre(system, *, te: float | None = None, n: int = 64):
    ...     '''Gradient echo.
    ...
    ...     Parameters
    ...     ----------
    ...     te : float | None, default=None
    ...         Echo time (s). ``None`` is as short as the readout admits.
    ...     n : int, default=64
    ...         Matrix size.
    ...     '''
    >>> protocol = sequences.parameters(gre)
    >>> list(protocol), protocol["te"].unit, protocol["te"].optional
    (['te', 'n'], 's', True)
    """
    hints = typing.get_type_hints(function)
    documentation = documented(inspect.getdoc(function))
    entries = {}
    for name, parameter in _prescribed(function).items():
        annotation = hints.get(name, parameter.annotation)
        kind, description = documentation.get(name, ("", ""))
        entries[name] = ProtocolParameter(
            name=name,
            type=scalar(annotation),
            default=parameter.default,
            optional=accepts_none(annotation),
            unit=_unit(description),
            choices=_choices(kind),
            description=description,
        )
    return entries


def _make_main(cls: type[SequenceApp]):
    """Build the module-level entry point of ``cls``, with its protocol in the signature."""

    def main(
        plot: bool = False,
        test_report: bool = False,
        write_seq: bool = False,
        seq_filename: str | None = None,
        *,
        system: pp.Opts | None = None,
        **protocol: Any,
    ) -> pp.Sequence:
        app = cls(system, **protocol)
        seq = app.design()
        if test_report:
            print(seq.test_report())
        if plot:
            seq.plot()
        if write_seq:
            app.write(seq_filename or f"{cls.NAME}.seq")
            seq = app.seq
        return seq

    def write_to(
        path: str,
        *,
        offline: bool = True,
        system: pp.Opts | None = None,
        test_report: bool = False,
        **protocol: Any,
    ) -> list[str]:
        app = cls(system, **protocol)
        paths = app.write(path, offline=offline)
        if test_report:
            print(app.seq.test_report())
        return paths

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
    main.write_to = write_to

    main.__doc__ = _documentation(
        cls, _MAIN_PARAMETERS, "pypulseqpp.Sequence\n    The designed sequence."
    )
    return main


def _documentation(cls: type[SequenceApp], parameters: str, returns: str) -> str:
    """Return the class's documentation, with ``parameters`` ahead of ``init_sequence``'s and ``returns`` as the Returns."""
    summary, description, sections = _split_sections(inspect.getdoc(cls) or "")
    entries = "\n".join(
        filter(None, (parameters, _section(cls.init_sequence, "Parameters")))
    )
    return "\n\n".join(
        part
        for part in (
            summary,
            description,
            _numpy_section("Parameters", entries),
            _numpy_section("Returns", returns),
            _numpy_section("Raises", _section(cls.init_sequence, "Raises")),
            *(_numpy_section(name, body) for name, body in sections),
        )
        if part
    )


def _numpy_section(name: str, body: str) -> str:
    """``body`` under a NumPy section heading, or the empty string if it is empty."""
    return f"{name}\n{'-' * len(name)}\n{body}" if body.strip() else ""


def _split_sections(doc: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Split a NumPy docstring into its summary, its description and its sections.

    The sections are returned in the order they appear, each as its heading and
    its body. Text before the first heading that is not the summary is the
    extended description, which belongs between the summary and ``Parameters``
    rather than after ``Returns``.
    """
    lines = doc.splitlines()
    headings = [
        (i, line.strip())
        for i, line in enumerate(lines[:-1])
        if line.strip() and line.strip() == line and set(lines[i + 1].strip()) == {"-"}
    ]
    bounds = [i for i, _ in headings] + [len(lines)]
    sections = [
        (name, "\n".join(lines[start + 2 : stop]).strip("\n"))
        for (start, name), stop in zip(headings, bounds[1:], strict=True)
    ]
    head = "\n".join(lines[: bounds[0]]).strip()
    summary, _, description = head.partition("\n\n")
    return summary, description.strip(), sections


def _section(function: Any, name: str) -> str:
    """Return the body of ``function``'s NumPy ``name`` section, dedented."""
    for heading, body in _split_sections(inspect.getdoc(function) or "")[2]:
        if heading == name:
            return body
    return ""
