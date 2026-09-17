"""Reference pages and figures for the shipped complete sequences.

Every complete sequence in ``pypulseqpp.sequences`` gets one page, written into
``docs/generated/sequences`` before Sphinx reads its sources. A page carries the
sequence's classification, its prescription rendered by ``autofunction`` from
the docstring, and figures of a configuration designed while the page is built,
so neither the reference text nor the figures can drift from the code.

:data:`SEQUENCES` is the single place a sequence's classification, its
documentation configuration and its figures are stated. The catalogue tables
under ``docs/sequences.md`` are written from the same table, with each
description read from the application's own summary line.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

PAGE_WIDTH = 7.4  # inches, the width of the documentation column

#: The families the catalogue groups by, in the order it presents them.
FAMILIES = (
    "Gradient echo",
    "Spin echo",
    "Fast spin echo",
    "MPRAGE",
    "Balanced SSFP",
    "Echo-planar imaging",
    "Zero echo time",
)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    return plt


def _echo_bounds(seq) -> tuple[np.ndarray, np.ndarray]:
    """Start and end time of each acquisition window, in seconds.

    Windows are separated in the sample times by a gap wider than one dwell.
    """
    times = seq.adc_times()[0]
    dwell = np.median(np.diff(times))
    breaks = np.flatnonzero(np.diff(times) > 1.5 * dwell)
    starts = np.concatenate(([times[0]], times[breaks + 1]))
    ends = np.concatenate((times[breaks], [times[-1]]))
    return starts, ends


def repetition(index: int = 1) -> Callable[[Any], dict]:
    """One repetition, with the others drawn underneath it."""
    return lambda seq: {"tr": index}


def repetitions(count: int, *, skip: int = 0) -> Callable[[Any], dict]:
    """``count`` repetition times, after the first ``skip`` of them.

    ``skip`` selects a repetition that samples representatively where the first
    one does not, as in a partition-by-partition acquisition whose outermost
    partitions hold few views.
    """

    def window(seq):
        tr = seq.get_definition("TR")[0]
        return {"time_range": (skip * tr, (skip + count) * tr)}

    return window


def echoes(
    count: int, *, from_start: bool = False, skip: int = 0, after: int = 0
) -> Callable[[Any], dict]:
    """``count`` acquisition windows, with the events around them.

    ``skip`` discards the acquisitions of that many repetition times first, for
    a sequence whose opening repetitions sample less than a later one. ``after``
    discards that many acquisition windows, which selects the imaging echoes of
    a train that opens with navigator acquisitions.
    """

    def window(seq):
        starts, ends = _echo_bounds(seq)
        opening = 0.0
        if skip:
            opening = skip * seq.get_definition("TR")[0]
            keep = starts >= opening
            starts, ends = starts[keep], ends[keep]
        starts, ends = starts[after:], ends[after:]
        margin = 0.5 * (ends[0] - starts[0])
        first = opening if from_start else max(opening, starts[0] - 3 * margin)
        return {"time_range": (first, ends[count - 1] + margin)}

    return window


@dataclass(frozen=True)
class Figure:
    """One figure of a sequence, and the caption that introduces it."""

    #: Suffix of the file name, after the module name.
    name: str
    #: ``"diagram"`` for a sequence diagram, ``"kspace"`` for a trajectory.
    kind: str
    #: The caption, one or two sentences.
    caption: str
    #: Keyword arguments for the drawing call, given the designed sequence.
    view: Callable[[Any], dict] = lambda seq: {}


@dataclass(frozen=True)
class SequenceDoc:
    """The classification, documentation configuration and figures of one sequence."""

    module: str
    title: str
    family: str
    dimensionality: str
    sampling: str
    readout: str
    config: dict[str, Any]
    figures: tuple[Figure, ...]
    #: A paragraph placed under the classification, for a property of the
    #: implementation that the summary line alone would leave ambiguous.
    notes: str = ""


#: Shared figure captions, so that a family's pages describe the same view the
#: same way.
_ONE_REPETITION = (
    "One repetition. The remaining repetitions are drawn underneath in grey, "
    "so an event that changes between them appears as a band."
)
_SEVERAL_REPETITIONS = "Consecutive repetitions, showing how the encoding steps."
_FIRST_ECHOES = "The opening of one echo train, at the raster the events are played on."
_TRAJECTORY = "Sampling locations in k-space, coloured by shot."


def samples(**options: Any) -> Callable[[Any], dict]:
    """Sampling locations, with the path between them drawn only when asked.

    A multi-shot acquisition that rotates one interleaf is clearer without the
    path, which otherwise crosses the whole of k-space between shots.
    """
    return lambda seq: {"show_trajectory": False, **options}

#: One entry per shipped sequence. The configuration is chosen so that the
#: figures show the structure of the sequence at documentation size; it is not
#: a protocol recommendation.
SEQUENCES: tuple[SequenceDoc, ...] = (
    SequenceDoc(
        module="gre2D_sequence",
        title="2D Cartesian gradient echo",
        family="Gradient echo",
        dimensionality="2D, multi-slice",
        sampling="Cartesian",
        readout="One phase-encode line per repetition, RF-spoiled",
        config=dict(
            n_x=64, n_y=16, n_slices=1, te=3e-3, tr=8e-3, n_dummy=0,
            readout_oversampling=1.0, readout_bandwidth_hz=200e3,
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("train", "diagram", _SEVERAL_REPETITIONS, repetitions(4)),
        ),
    ),
    SequenceDoc(
        module="gre3D_sequence",
        title="3D Cartesian gradient echo",
        family="Gradient echo",
        dimensionality="3D, slab-selective",
        sampling="Cartesian",
        readout="One (line, partition) view per repetition, RF-spoiled",
        config=dict(
            n_x=64, n_y=16, n_z=8, te=None, tr=None, n_dummy=0,
            readout_oversampling=1.0, readout_bandwidth_hz=200e3,
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("train", "diagram", _SEVERAL_REPETITIONS, repetitions(4)),
        ),
    ),
    SequenceDoc(
        module="gre_multiecho2D_sequence",
        title="2D Cartesian multi-echo gradient echo",
        family="Gradient echo",
        dimensionality="2D, multi-slice",
        sampling="Cartesian",
        readout="One phase-encode line read at several echo times per repetition",
        config=dict(
            n_x=64, n_y=16, n_slices=1, n_echoes=4, te=None, tr=12e-3, n_dummy=0,
            readout_oversampling=1.0, readout_bandwidth_hz=200e3,
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("echoes", "diagram", _FIRST_ECHOES, echoes(2)),
        ),
    ),
    SequenceDoc(
        module="gre_multiecho3D_sequence",
        title="3D Cartesian multi-echo gradient echo",
        family="Gradient echo",
        dimensionality="3D, slab-selective",
        sampling="Cartesian",
        readout="One (line, partition) view read at several echo times per repetition",
        config=dict(
            n_x=64, n_y=16, n_z=8, n_echoes=4, tr=None, n_dummy=0,
            readout_oversampling=1.0, readout_bandwidth_hz=200e3,
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("echoes", "diagram", _FIRST_ECHOES, echoes(2)),
        ),
    ),
    SequenceDoc(
        module="gre_radial2D_sequence",
        title="2D radial gradient echo",
        family="Gradient echo",
        dimensionality="2D, multi-slice",
        sampling="Radial",
        readout="One full spoke per repetition, RF-spoiled",
        config=dict(
            n=64, ry=8, n_slices=1, te=None, tr=8e-3, n_dummy=0, readout_oversampling=1.0
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
    SequenceDoc(
        module="gre_stack_of_stars3D_sequence",
        title="3D stack-of-stars gradient echo",
        family="Gradient echo",
        dimensionality="3D, slab-selective",
        sampling="Radial in-plane, Cartesian partitions",
        readout="One spoke at one partition per repetition, RF-spoiled",
        config=dict(n=64, ry=8, n_z=4, tr=None, n_dummy=0, readout_oversampling=1.0),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
    SequenceDoc(
        module="gre_spiral2D_sequence",
        title="2D spiral gradient echo",
        family="Gradient echo",
        dimensionality="2D, multi-slice",
        sampling="Spiral",
        readout="One interleaf per repetition, RF-spoiled",
        config=dict(n=64, n_shots=8, n_slices=1, tr=9e-3, n_dummy=0),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY),
        ),
    ),
    SequenceDoc(
        module="gre_stack_of_spirals3D_sequence",
        title="3D stack-of-spirals gradient echo",
        family="Gradient echo",
        dimensionality="3D, slab-selective",
        sampling="Spiral in-plane, Cartesian partitions",
        readout="One interleaf at one partition per repetition, RF-spoiled",
        config=dict(n=64, n_z=4, n_shots=8, tr=None, n_dummy=0),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY),
        ),
    ),
    SequenceDoc(
        module="gre_propeller2D_sequence",
        title="2D PROPELLER gradient echo",
        family="Gradient echo",
        dimensionality="2D, multi-slice",
        sampling="PROPELLER blades",
        readout="One line of one blade per repetition, RF-spoiled",
        config=dict(
            n=64, blade_width=8, n_slices=1, te=4e-3, tr=10e-3, n_dummy=0
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
    SequenceDoc(
        module="gre_stack_of_blades3D_sequence",
        title="3D stack-of-blades gradient echo",
        family="Gradient echo",
        dimensionality="3D, slab-selective",
        sampling="PROPELLER in-plane, Cartesian partitions",
        readout="One blade line at one partition per repetition, RF-spoiled",
        config=dict(n=64, n_z=4, blade_width=8, tr=None, n_dummy=0),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
    SequenceDoc(
        module="se2D_sequence",
        title="2D Cartesian spin echo",
        family="Spin echo",
        dimensionality="2D, multi-slice",
        sampling="Cartesian",
        readout="One phase-encode line per excitation, read at a refocused echo",
        config=dict(
            n_x=64, n_y=16, n_slices=1, te=14e-3, tr=22e-3, n_dummy=0,
            readout_oversampling=1.0,
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("train", "diagram", _SEVERAL_REPETITIONS, repetitions(3)),
        ),
    ),
    SequenceDoc(
        module="se3D_sequence",
        title="3D Cartesian spin echo",
        family="Spin echo",
        dimensionality="3D, slab-selective",
        sampling="Cartesian",
        readout="One (line, partition) view per excitation, read at a refocused echo",
        config=dict(
            n_x=64, n_y=16, n_z=8, te=None, tr=16e-3, n_dummy=0,
            readout_oversampling=1.0,
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("train", "diagram", _SEVERAL_REPETITIONS, repetitions(3)),
        ),
    ),
    SequenceDoc(
        module="se_radial2D_sequence",
        title="2D radial spin echo",
        family="Spin echo",
        dimensionality="2D, multi-slice",
        sampling="Radial",
        readout="One full spoke per excitation, read at a refocused echo",
        config=dict(
            n=64, ry=8, n_slices=1, te=None, tr=20e-3, n_dummy=0, readout_oversampling=1.0
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
    SequenceDoc(
        module="se_stack_of_stars3D_sequence",
        title="3D stack-of-stars spin echo",
        family="Spin echo",
        dimensionality="3D, slab-selective",
        sampling="Radial in-plane, Cartesian partitions",
        readout="One spoke at one partition per excitation",
        config=dict(
            n=64, ry=8, n_z=4, te=None, tr=17e-3, n_dummy=0, readout_oversampling=1.0
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
    SequenceDoc(
        module="se_spiral2D_sequence",
        title="2D spiral spin echo",
        family="Spin echo",
        dimensionality="2D, multi-slice",
        sampling="Spiral",
        readout="One interleaf per excitation, read at a refocused echo",
        config=dict(n=64, n_shots=8, n_slices=1, te=None, tr=22e-3, n_dummy=0),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY),
        ),
    ),
    SequenceDoc(
        module="se_stack_of_spirals3D_sequence",
        title="3D stack-of-spirals spin echo",
        family="Spin echo",
        dimensionality="3D, slab-selective",
        sampling="Spiral in-plane, Cartesian partitions",
        readout="One interleaf at one partition per excitation",
        config=dict(n=64, n_z=4, n_shots=8, te=None, tr=19e-3, n_dummy=0),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY),
        ),
    ),
    SequenceDoc(
        module="se_propeller2D_sequence",
        title="2D PROPELLER spin echo",
        family="Spin echo",
        dimensionality="2D, multi-slice",
        sampling="PROPELLER blades",
        readout="One line of one blade per excitation",
        config=dict(n=64, blade_width=8, n_slices=1, te=14e-3, tr=22e-3, n_dummy=0),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
    SequenceDoc(
        module="se_epi_propeller2D_sequence",
        title="2D PROPELLER spin echo with echo-planar blades",
        family="Spin echo",
        dimensionality="2D, multi-slice",
        sampling="PROPELLER blades, echo-planar within a blade",
        readout="One whole blade per excitation, read as an echo-planar train",
        config=dict(
            n_x=64, blade_width=8, n_blades=4, n_slices=1, te=30e-3, tr=45e-3,
            n_dummy=0,
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
    SequenceDoc(
        module="se_stack_of_blades3D_sequence",
        title="3D stack-of-blades spin echo",
        family="Spin echo",
        dimensionality="3D, slab-selective",
        sampling="PROPELLER in-plane, Cartesian partitions",
        readout="One blade line at one partition per excitation",
        config=dict(n=64, n_z=4, blade_width=8, te=None, tr=16e-3, n_dummy=0),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
    SequenceDoc(
        module="fse3D_sequence",
        title="3D fast spin echo",
        family="Fast spin echo",
        dimensionality="3D, slab-selective",
        sampling="Cartesian",
        readout="One CPMG train per excitation, one (line, partition) view per echo",
        config=dict(
            n_x=64, n_y=16, n_z=8, etl=8, te=None, tr=100e-3, n_dummy=0,
            readout_oversampling=1.0,
        ),
        figures=(
            Figure(
                "train",
                "diagram",
                "One echo train: the excitation, then one refocusing pulse and one "
                "acquisition per echo.",
                repetitions(1),
            ),
            Figure("echoes", "diagram", _FIRST_ECHOES, echoes(3, from_start=True)),
        ),
    ),
    SequenceDoc(
        module="mprage3D_sequence",
        title="3D Cartesian MPRAGE",
        family="MPRAGE",
        dimensionality="3D, slab-selective",
        sampling="Cartesian",
        readout=(
            "One inversion per shot, followed by the phase-encode lines of one "
            "partition as a spoiled gradient-echo train"
        ),
        notes=(
            "Each shot applies one inversion and then acquires every sampled "
            "in-plane view of a single partition, so the partition encode is "
            "constant within a shot and the number of shots equals the number "
            "of sampled partitions. The inversion time is measured from the "
            "centre of the inversion pulse to the centre of the first "
            "excitation of the train, and the repetition time is the interval "
            "between successive inversions."
        ),
        config=dict(
            n_x=64, n_y=16, n_z=8, ti=15e-3, tr=110e-3, n_dummy=0,
            readout_oversampling=1.0,
        ),
        figures=(
            Figure(
                "shot",
                "diagram",
                "One shot: the inversion pulse and its crusher, the inversion "
                "delay, the gradient-echo train, and the recovery that completes "
                "the inversion-to-inversion interval.",
                repetitions(1, skip=4),
            ),
            Figure(
                "echoes",
                "diagram",
                "The first repetitions of the train, at the raster the events are "
                "played on.",
                echoes(3, skip=4),
            ),
        ),
    ),
    SequenceDoc(
        module="mprage_stack_of_stars3D_sequence",
        title="3D stack-of-stars MPRAGE",
        family="MPRAGE",
        dimensionality="3D, slab-selective",
        sampling="Radial in-plane, Cartesian partitions",
        readout=(
            "One inversion per shot, followed by the radial spokes of one "
            "partition as a spoiled gradient-echo train"
        ),
        notes=(
            "Each shot applies one inversion and then acquires every sampled "
            "in-plane view of a single partition, so the partition encode is "
            "constant within a shot and the number of shots equals the number "
            "of sampled partitions. The inversion time is measured from the "
            "centre of the inversion pulse to the centre of the first "
            "excitation of the train, and the repetition time is the interval "
            "between successive inversions."
        ),
        config=dict(
            n=64, n_z=4, ry=8, ti=15e-3, tr=110e-3, n_dummy=0,
            readout_oversampling=1.0,
        ),
        figures=(
            Figure(
                "shot",
                "diagram",
                "One shot: the inversion, the inversion delay, the train of spokes, "
                "and the recovery.",
                repetitions(1, skip=4),
            ),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
    SequenceDoc(
        module="mprage_stack_of_spirals3D_sequence",
        title="3D stack-of-spirals MPRAGE",
        family="MPRAGE",
        dimensionality="3D, slab-selective",
        sampling="Spiral in-plane, Cartesian partitions",
        readout=(
            "One inversion per shot, followed by the spiral interleaves of one "
            "partition as a spoiled gradient-echo train"
        ),
        notes=(
            "Each shot applies one inversion and then acquires every sampled "
            "in-plane view of a single partition, so the partition encode is "
            "constant within a shot and the number of shots equals the number "
            "of sampled partitions. The inversion time is measured from the "
            "centre of the inversion pulse to the centre of the first "
            "excitation of the train, and the repetition time is the interval "
            "between successive inversions."
        ),
        config=dict(n=64, n_z=4, n_shots=8, ti=15e-3, tr=110e-3, n_dummy=0),
        figures=(
            Figure(
                "shot",
                "diagram",
                "One shot: the inversion, the inversion delay, the train of "
                "interleaves, and the recovery.",
                repetitions(1, skip=4),
            ),
            Figure("kspace", "kspace", _TRAJECTORY),
        ),
    ),
    SequenceDoc(
        module="bssfp2D_sequence",
        title="2D balanced SSFP",
        family="Balanced SSFP",
        dimensionality="2D, multi-slice",
        sampling="Cartesian",
        readout="One phase-encode line per repetition, every gradient axis balanced",
        config=dict(
            n_x=64, n_y=16, n_slices=1, n_phases=1, tr=None, n_dummy=4,
            readout_oversampling=1.0, readout_bandwidth_hz=80e3,
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure(
                "train",
                "diagram",
                "Consecutive repetitions. Every gradient axis returns to zero "
                "moment within each repetition.",
                repetitions(4, skip=3),
            ),
        ),
    ),
    SequenceDoc(
        module="bssfp3D_sequence",
        title="3D balanced SSFP",
        family="Balanced SSFP",
        dimensionality="3D, slab-selective or non-selective",
        sampling="Cartesian",
        readout="One (line, partition) view per repetition, every gradient axis "
        "balanced",
        config=dict(
            n_x=64, n_y=16, n_z=8, tr=None, readout_oversampling=1.0,
            readout_bandwidth_hz=80e3,
        ),
        figures=(
            Figure("repetition", "diagram", _ONE_REPETITION, repetition()),
            Figure(
                "train",
                "diagram",
                "Consecutive repetitions. Every gradient axis returns to zero "
                "moment within each repetition.",
                repetitions(4, skip=3),
            ),
        ),
    ),
    SequenceDoc(
        module="epi2D_sequence",
        title="2D echo-planar imaging",
        family="Echo-planar imaging",
        dimensionality="2D, multi-slice",
        sampling="Cartesian",
        readout="One blipped echo train per excitation",
        config=dict(n_x=32, n_y=32, n_slices=1, tr=None, n_dummy=0),
        figures=(
            Figure(
                "train",
                "diagram",
                "One echo train: the excitation, the prewinders, and the "
                "alternating readout lobes with the phase-encode blips between "
                "them.",
                repetition(),
            ),
            Figure("echoes", "diagram", _FIRST_ECHOES, echoes(4, after=3)),
        ),
    ),
    SequenceDoc(
        module="epi3D_sequence",
        title="3D echo-planar imaging",
        family="Echo-planar imaging",
        dimensionality="3D, slab-selective",
        sampling="Cartesian, skipped-CAIPI when undersampled",
        readout="One blipped echo train per (shot, shell)",
        config=dict(n_x=32, n_y=32, n_z=4, tr=None, n_dummy=0),
        figures=(
            Figure(
                "train",
                "diagram",
                "One echo train, with the partition encode applied before it.",
                repetition(),
            ),
            Figure("echoes", "diagram", _FIRST_ECHOES, echoes(4, after=3)),
        ),
    ),
    SequenceDoc(
        module="zte3D_sequence",
        title="3D zero echo time",
        family="Zero echo time",
        dimensionality="3D, non-selective",
        sampling="Radial half-spokes on a sphere",
        readout="A hard pulse transmitted with the readout gradient already at "
        "amplitude, one half-spoke per view",
        config=dict(n_x=32, n_dummy=0),
        figures=(
            Figure(
                "views",
                "diagram",
                "Several consecutive views. The readout gradient stays on across "
                "the hard pulses, and its amplitude steps from one view to the "
                "next.",
                echoes(4, from_start=True),
            ),
            Figure("kspace", "kspace", _TRAJECTORY, samples()),
        ),
    ),
)


def _app(doc: SequenceDoc) -> type:
    """The ``SequenceApp`` subclass ``doc`` documents."""
    module = importlib.import_module(f"pypulseqpp.sequences.sequence.{doc.module}")
    return next(
        value
        for name, value in vars(module).items()
        if name.endswith("App") and isinstance(value, type)
    )


def summary(doc: SequenceDoc) -> str:
    """The application's own summary line, as the catalogue reports it."""
    return (inspect.getdoc(_app(doc)) or "").split("\n\n")[0].replace("\n", " ")


def design(doc: SequenceDoc):
    """Design the documentation configuration of ``doc``."""
    from pypulseqpp import sequences

    return getattr(sequences, doc.module)(**doc.config)


def _call(doc: SequenceDoc) -> str:
    """The configuration as the call that produces it."""
    arguments = ",\n    ".join(f"{k}={v!r}" for k, v in doc.config.items())
    return f"seq = sequences.{doc.module}(\n    {arguments},\n)"


def draw(doc: SequenceDoc, seq, figure: Figure):
    """Draw one figure of ``seq``."""
    import pypulseqpp as pp

    if figure.kind == "kspace":
        return pp.plot.plot_kspace(
            seq, color_by="shot", plot_now=False, **figure.view(seq)
        )
    return seq.paper_plot(**figure.view(seq)).diagram.plot.get_figure()


def _related(doc: SequenceDoc) -> tuple[list[SequenceDoc], list[SequenceDoc]]:
    """The other sequences of ``doc``'s family, and those sampling the same way."""
    family = [
        other
        for other in SEQUENCES
        if other.family == doc.family and other.module != doc.module
    ]
    sampling = [
        other
        for other in SEQUENCES
        if other.sampling == doc.sampling and other.family != doc.family
    ]
    return family, sampling


def _links(docs: list[SequenceDoc]) -> str:
    """``docs`` as a comma-separated list of cross-references."""
    return ", ".join(f":doc:`{doc.module} <{doc.module}>`" for doc in docs)


def page(doc: SequenceDoc, figures: list[str]) -> str:
    """The reStructuredText of one sequence's reference page."""
    rule = "=" * len(doc.title)
    lines = [
        rule,
        doc.title,
        rule,
        "",
        f":Family: {doc.family}",
        f":Dimensionality: {doc.dimensionality}",
        f":Sampling: {doc.sampling}",
        f":Readout: {doc.readout}",
        "",
        f"{summary(doc)}",
        "",
    ]
    if doc.notes:
        lines += [doc.notes, ""]
    lines += [
        "Prescription",
        "------------",
        "",
        ".. currentmodule:: pypulseqpp.sequences",
        "",
        f".. autofunction:: {doc.module}",
        "",
        "Representative configuration for documentation",
        "----------------------------------------------",
        "",
        "The configuration below is chosen to make the structure of the sequence",
        "legible at the size of this page: the matrix is small, delays that would",
        "otherwise dominate the diagram are short, and the figures are drawn from",
        "the sequence this call designs. It is not a protocol recommendation.",
        "",
        ".. code-block:: python",
        "",
        "    from pypulseqpp import sequences",
        "",
    ]
    lines += [f"    {line}" for line in _call(doc).splitlines()]
    lines.append("")
    for figure, path in zip(doc.figures, figures, strict=True):
        lines += [
            f".. figure:: /{path}",
            "   :width: 100%",
            "",
            f"   {figure.caption}",
            "",
        ]
    family, sampling = _related(doc)
    lines += ["Related sequences", "-----------------", ""]
    if family:
        lines += [
            f"Other {doc.family.lower()} sequences: " + _links(family) + ".",
            "",
        ]
    if sampling:
        lines += [
            "The same sampling in another family: " + _links(sampling) + ".",
            "",
        ]
    lines += [
        "Every shipped sequence is listed in the :doc:`catalogue </sequences>`.",
        "",
    ]
    return "\n".join(lines)


def catalogue_table(family: str) -> str:
    """The catalogue's table for one family, as reStructuredText."""
    rows = [doc for doc in SEQUENCES if doc.family == family]
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 26 16 24 34",
        "",
        "   * - Sequence",
        "     - Dimensionality",
        "     - Sampling",
        "     - Description",
    ]
    for doc in rows:
        lines += [
            f"   * - :doc:`{doc.module} </generated/sequences/{doc.module}>`",
            f"     - {doc.dimensionality}",
            f"     - {doc.sampling}",
            f"     - {summary(doc)}",
        ]
    return "\n".join(lines) + "\n"


def render(into: str | Path) -> None:
    """Write every sequence's page and figures into ``into``.

    ``into`` is the documentation source directory. Pages are written under
    ``generated/sequences`` and the catalogue's tables under
    ``generated/sequences/tables``, both of which the sources include by path.
    """
    plt = _pyplot()
    root = Path(into)
    pages = root / "generated" / "sequences"
    tables = pages / "tables"
    pages.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    for family in FAMILIES:
        stem = family.lower().replace(" ", "-")
        (tables / f"{stem}.rst").write_text(catalogue_table(family), encoding="utf-8")

    for doc in SEQUENCES:
        seq = design(doc)
        written = []
        for figure in doc.figures:
            drawn = draw(doc, seq, figure)
            drawn.set_size_inches(PAGE_WIDTH, drawn.get_size_inches()[1])
            name = f"generated/sequences/{doc.module}-{figure.name}.png"
            drawn.savefig(root / name, bbox_inches="tight")
            plt.close(drawn)
            written.append(name)
        (pages / f"{doc.module}.rst").write_text(page(doc, written), encoding="utf-8")

    (pages / "index.rst").write_text(_toctree(), encoding="utf-8")


def _toctree() -> str:
    """A hidden toctree over every sequence page, for the catalogue to include."""
    lines = [".. toctree::", "   :hidden:", ""]
    lines += [f"   /generated/sequences/{doc.module}" for doc in SEQUENCES]
    return "\n".join(lines) + "\n"
