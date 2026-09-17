"""Reference pages for the shipped complete sequences.

Every complete sequence in ``pypulseqpp.sequences`` gets one page, written into
``docs/generated/sequences`` before Sphinx reads its sources. A page carries the
sequence's classification, its prescription rendered by ``autofunction`` from
the docstring, and the gallery pages that build it, so the reference text cannot
drift from the code. The configurations and the figures are in the gallery,
under ``docs/examples/built-in-sequences``.

:data:`SEQUENCES` is the single place a sequence's classification is stated. The
catalogue tables under ``docs/sequences.md`` are written from the same table,
with each description read from the application's own summary line.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class SequenceDoc:
    """The classification of one shipped sequence."""

    module: str
    title: str
    family: str
    dimensionality: str
    sampling: str
    readout: str
    #: A paragraph placed under the classification, for a property of the
    #: implementation that the summary line alone would leave ambiguous.
    notes: str = ""


SEQUENCES: tuple[SequenceDoc, ...] = (
    SequenceDoc(
        module="gre2D_sequence",
        title="2D Cartesian gradient echo",
        family="Gradient echo",
        dimensionality="2D, multi-slice",
        sampling="Cartesian",
        readout="One phase-encode line per repetition, RF-spoiled",
    ),
    SequenceDoc(
        module="gre3D_sequence",
        title="3D Cartesian gradient echo",
        family="Gradient echo",
        dimensionality="3D, slab-selective",
        sampling="Cartesian",
        readout="One (line, partition) view per repetition, RF-spoiled",
    ),
    SequenceDoc(
        module="gre_multiecho2D_sequence",
        title="2D Cartesian multi-echo gradient echo",
        family="Gradient echo",
        dimensionality="2D, multi-slice",
        sampling="Cartesian",
        readout="One phase-encode line read at several echo times per repetition",
    ),
    SequenceDoc(
        module="gre_multiecho3D_sequence",
        title="3D Cartesian multi-echo gradient echo",
        family="Gradient echo",
        dimensionality="3D, slab-selective",
        sampling="Cartesian",
        readout="One (line, partition) view read at several echo times per repetition",
    ),
    SequenceDoc(
        module="gre_radial2D_sequence",
        title="2D radial gradient echo",
        family="Gradient echo",
        dimensionality="2D, multi-slice",
        sampling="Radial",
        readout="One full spoke per repetition, RF-spoiled",
    ),
    SequenceDoc(
        module="gre_stack_of_stars3D_sequence",
        title="3D stack-of-stars gradient echo",
        family="Gradient echo",
        dimensionality="3D, slab-selective",
        sampling="Radial in-plane, Cartesian partitions",
        readout="One spoke at one partition per repetition, RF-spoiled",
    ),
    SequenceDoc(
        module="gre_spiral2D_sequence",
        title="2D spiral gradient echo",
        family="Gradient echo",
        dimensionality="2D, multi-slice",
        sampling="Spiral",
        readout="One interleaf per repetition, RF-spoiled",
    ),
    SequenceDoc(
        module="gre_stack_of_spirals3D_sequence",
        title="3D stack-of-spirals gradient echo",
        family="Gradient echo",
        dimensionality="3D, slab-selective",
        sampling="Spiral in-plane, Cartesian partitions",
        readout="One interleaf at one partition per repetition, RF-spoiled",
    ),
    SequenceDoc(
        module="gre_propeller2D_sequence",
        title="2D PROPELLER gradient echo",
        family="Gradient echo",
        dimensionality="2D, multi-slice",
        sampling="PROPELLER blades",
        readout="One line of one blade per repetition, RF-spoiled",
    ),
    SequenceDoc(
        module="gre_stack_of_blades3D_sequence",
        title="3D stack-of-blades gradient echo",
        family="Gradient echo",
        dimensionality="3D, slab-selective",
        sampling="PROPELLER in-plane, Cartesian partitions",
        readout="One blade line at one partition per repetition, RF-spoiled",
    ),
    SequenceDoc(
        module="se2D_sequence",
        title="2D Cartesian spin echo",
        family="Spin echo",
        dimensionality="2D, multi-slice",
        sampling="Cartesian",
        readout="One phase-encode line per excitation, read at a refocused echo",
    ),
    SequenceDoc(
        module="se3D_sequence",
        title="3D Cartesian spin echo",
        family="Spin echo",
        dimensionality="3D, slab-selective",
        sampling="Cartesian",
        readout="One (line, partition) view per excitation, read at a refocused echo",
    ),
    SequenceDoc(
        module="se_radial2D_sequence",
        title="2D radial spin echo",
        family="Spin echo",
        dimensionality="2D, multi-slice",
        sampling="Radial",
        readout="One full spoke per excitation, read at a refocused echo",
    ),
    SequenceDoc(
        module="se_stack_of_stars3D_sequence",
        title="3D stack-of-stars spin echo",
        family="Spin echo",
        dimensionality="3D, slab-selective",
        sampling="Radial in-plane, Cartesian partitions",
        readout="One spoke at one partition per excitation",
    ),
    SequenceDoc(
        module="se_spiral2D_sequence",
        title="2D spiral spin echo",
        family="Spin echo",
        dimensionality="2D, multi-slice",
        sampling="Spiral",
        readout="One interleaf per excitation, read at a refocused echo",
    ),
    SequenceDoc(
        module="se_stack_of_spirals3D_sequence",
        title="3D stack-of-spirals spin echo",
        family="Spin echo",
        dimensionality="3D, slab-selective",
        sampling="Spiral in-plane, Cartesian partitions",
        readout="One interleaf at one partition per excitation",
    ),
    SequenceDoc(
        module="se_propeller2D_sequence",
        title="2D PROPELLER spin echo",
        family="Spin echo",
        dimensionality="2D, multi-slice",
        sampling="PROPELLER blades",
        readout="One line of one blade per excitation",
    ),
    SequenceDoc(
        module="se_epi_propeller2D_sequence",
        title="2D PROPELLER spin echo with echo-planar blades",
        family="Spin echo",
        dimensionality="2D, multi-slice",
        sampling="PROPELLER blades, echo-planar within a blade",
        readout="One whole blade per excitation, read as an echo-planar train",
    ),
    SequenceDoc(
        module="se_stack_of_blades3D_sequence",
        title="3D stack-of-blades spin echo",
        family="Spin echo",
        dimensionality="3D, slab-selective",
        sampling="PROPELLER in-plane, Cartesian partitions",
        readout="One blade line at one partition per excitation",
    ),
    SequenceDoc(
        module="fse3D_sequence",
        title="3D fast spin echo",
        family="Fast spin echo",
        dimensionality="3D, slab-selective",
        sampling="Cartesian",
        readout="One CPMG train per excitation, one (line, partition) view per echo",
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
            "Each shot applies one inversion and then acquires every sampled in- "
            "plane view of a single partition, so the partition encode is "
            "constant within a shot and the number of shots equals the number of "
            "sampled partitions. The inversion time is measured from the centre "
            "of the inversion pulse to the centre of the first excitation of the "
            "train, and the repetition time is the interval between successive "
            "inversions."
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
            "Each shot applies one inversion and then acquires every sampled in- "
            "plane view of a single partition, so the partition encode is "
            "constant within a shot and the number of shots equals the number of "
            "sampled partitions. The inversion time is measured from the centre "
            "of the inversion pulse to the centre of the first excitation of the "
            "train, and the repetition time is the interval between successive "
            "inversions."
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
            "Each shot applies one inversion and then acquires every sampled in- "
            "plane view of a single partition, so the partition encode is "
            "constant within a shot and the number of shots equals the number of "
            "sampled partitions. The inversion time is measured from the centre "
            "of the inversion pulse to the centre of the first excitation of the "
            "train, and the repetition time is the interval between successive "
            "inversions."
        ),
    ),
    SequenceDoc(
        module="bssfp2D_sequence",
        title="2D balanced SSFP",
        family="Balanced SSFP",
        dimensionality="2D, multi-slice",
        sampling="Cartesian",
        readout="One phase-encode line per repetition, every gradient axis balanced",
    ),
    SequenceDoc(
        module="bssfp3D_sequence",
        title="3D balanced SSFP",
        family="Balanced SSFP",
        dimensionality="3D, slab-selective or non-selective",
        sampling="Cartesian",
        readout=(
            "One (line, partition) view per repetition, every gradient axis "
            "balanced"
        ),
    ),
    SequenceDoc(
        module="epi2D_sequence",
        title="2D echo-planar imaging",
        family="Echo-planar imaging",
        dimensionality="2D, multi-slice",
        sampling="Cartesian",
        readout="One blipped echo train per excitation",
    ),
    SequenceDoc(
        module="epi3D_sequence",
        title="3D echo-planar imaging",
        family="Echo-planar imaging",
        dimensionality="3D, slab-selective",
        sampling="Cartesian, skipped-CAIPI when undersampled",
        readout="One blipped echo train per (shot, shell)",
    ),
    SequenceDoc(
        module="zte3D_sequence",
        title="3D zero echo time",
        family="Zero echo time",
        dimensionality="3D, non-selective",
        sampling="Radial half-spokes on a sphere",
        readout=(
            "A hard pulse transmitted with the readout gradient already at "
            "amplitude, one half-spoke per view"
        ),
    ),
)



def _app(doc: SequenceDoc) -> type:
    """Return the ``SequenceApp`` subclass ``doc`` documents."""
    module = importlib.import_module(f"pypulseqpp.sequences.sequence.{doc.module}")
    return next(
        value
        for name, value in vars(module).items()
        if name.endswith("App") and isinstance(value, type)
    )


def summary(doc: SequenceDoc) -> str:
    """Return the application's own summary line, as the catalogue reports it."""
    return (inspect.getdoc(_app(doc)) or "").split("\n\n")[0].replace("\n", " ")


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


def page(doc: SequenceDoc) -> str:
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
        summary(doc),
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
        ".. minigallery:: pypulseqpp.sequences." + doc.module,
        "   :add-heading: Designed and drawn",
        "",
    ]
    family, sampling = _related(doc)
    lines += ["See also", "--------", ""]
    if family:
        lines += [f"Other {doc.family.lower()} sequences: " + _links(family) + ".", ""]
    if sampling:
        lines += ["The same sampling in another family: " + _links(sampling) + ".", ""]
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
    """Write every sequence's reference page into ``into``.

    ``into`` is the documentation source directory. Pages are written under
    ``generated/sequences`` and the catalogue's tables under
    ``generated/sequences/tables``, both of which the sources include by path.
    """
    root = Path(into)
    pages = root / "generated" / "sequences"
    tables = pages / "tables"
    pages.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    for family in FAMILIES:
        stem = family.lower().replace(" ", "-")
        (tables / f"{stem}.rst").write_text(catalogue_table(family), encoding="utf-8")

    for doc in SEQUENCES:
        (pages / f"{doc.module}.rst").write_text(page(doc), encoding="utf-8")

    (pages / "pages.rst").write_text(_holder(), encoding="utf-8")


def _holder() -> str:
    """The page that owns the sequence pages' toctree, outside the navigation.

    The catalogue links every sequence from its family tables, so the toctree
    exists to give the pages a parent rather than to be navigated: an orphan
    holder keeps twenty-eight entries out of the sidebar without leaving them
    outside every toctree.
    """
    lines = [
        ":orphan:",
        "",
        "Sequence reference pages",
        "========================",
        "",
        "The reference page of every shipped sequence, reached from the tables",
        "in the :doc:`catalogue </sequences>`.",
        "",
        ".. toctree::",
        "   :hidden:",
        "",
    ]
    lines += [f"   /generated/sequences/{doc.module}" for doc in SEQUENCES]
    return "\n".join(lines) + "\n"
