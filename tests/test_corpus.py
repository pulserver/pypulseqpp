"""Pulseq reader coverage using external sequence files and format revisions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pypulseqpp import _ext

CORPUS = Path(__file__).parent / "seq"

#: One sequence written at every revision the format has had. What the 1.5.0
#: file says is what the others have to be read as, so these are the fixtures
#: for reading an older file rather than files to be round-tripped.
AS_1_5 = CORPUS / "simple_mprage150.seq"
OLDER = sorted(
    path.name for path in CORPUS.glob("simple_mprage1[234]*.seq") if path != AS_1_5
)

#: Of those, the ones whose shapes are stored the way 1.5 stores them. Before
#: 1.4 an extended trapezoid carried no time shape of its own, so the same
#: sequence is held in fewer entries -- six rather than eight here.
SHAPES_AS_1_5 = [name for name in OLDER if name >= "simple_mprage14"]

#: And the ones whose blocks last as long. 1.2.0 is a different sequence in
#: that one respect, not a conversion that went wrong: the reference reader
#: gets the same duration from it.
DURATION_AS_1_5 = [name for name in OLDER if name >= "simple_mprage13"]

#: Every file the reader is expected to take as it stands.
CURRENT = sorted(
    path.name
    for path in CORPUS.glob("*.seq")
    if not path.name.startswith("simple_mprage1") or path == AS_1_5
)


def rows(text: bytes, section: str):
    """The data lines of one section of a written file."""
    out, inside = [], False
    for line in text.decode().splitlines():
        if line.startswith("["):
            inside = line == section
        elif inside and line and not line.startswith("#"):
            out.append(line)
    return out


def normalise(text: bytes) -> str:
    """Remove comments, signatures, TotalDuration and writer revision before comparison."""
    out = text.decode().replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\n\[SIGNATURE\][\s\S]*$", "", out)
    out = re.sub(r"^#.*\n", "", out, flags=re.MULTILINE)
    out = re.sub(r"^revision \d+\n", "", out, flags=re.MULTILINE)
    out = re.sub(r"^TotalDuration .*\n", "", out, flags=re.MULTILINE)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.rstrip() + "\n"


#: Files a rewrite here cannot reproduce byte for byte, and why.
#:
#: A soft delay's offset is written `%g` by the toolbox that defines the
#: format and `%.0f` by the other one, so a whole number of microseconds
#: comes out as `1e+06` there and `1000000` here. These files were written
#: the second way. What is held for them instead is that writing is
#: idempotent, which is the test below.
WRITTEN_WITH_OTHER_CONVENTIONS = {"seq6.seq"}


@pytest.fixture(params=CURRENT, ids=lambda name: name[:-4])
def corpus_file(request):
    """Each current-revision file of the corpus in turn."""
    return CORPUS / request.param


def test_the_corpus_is_here(corpus_file):
    """A guard: an empty glob would make every test below vacuous."""
    assert len(CURRENT) >= 30
    assert corpus_file.stat().st_size > 0


def test_a_file_is_written_back_as_it_was_read(corpus_file):
    """The reader and the writer are each other's inverse on real files."""
    if corpus_file.name in WRITTEN_WITH_OTHER_CONVENTIONS:
        pytest.skip("written with the other toolbox's conventions; see below")
    contents = corpus_file.read_bytes()

    written = _ext.write_text(_ext.read(contents), True)

    assert normalise(written) == normalise(contents)


def test_writing_a_file_twice_writes_the_same_bytes(corpus_file):
    """Reading back what we wrote changes nothing, whoever wrote the source.

    This is what still holds for a file written with another toolbox's
    formatting: our conventions may differ from the source's, but applying
    them twice has to be the same as applying them once.
    """
    once = _ext.write_text(_ext.read(corpus_file.read_bytes()), True)

    twice = _ext.write_text(_ext.read(once), True)

    assert twice == once


def test_a_file_survives_the_binary_form(corpus_file):
    """Everything but the shapes, which the binary form holds in float32."""
    text = _ext.write_text(_ext.read(corpus_file.read_bytes()), True)

    through_binary = _ext.write_text(
        _ext.read(_ext.write_binary(_ext.read(corpus_file.read_bytes()))), True
    )

    def comparable(rendered):
        return re.sub(r"\n\[SHAPES\][\s\S]*$", "", normalise(rendered))

    assert comparable(through_binary) == comparable(text)


def test_reading_a_file_splits_it_into_definitions_and_instances(corpus_file):
    """A sequence off disk is forked, whoever wrote the file."""
    sequence = _ext.read(corpus_file.read_bytes())

    definitions = sequence.instance_definitions()
    assert len(definitions) == sequence.num_blocks()
    if sequence.num_blocks():
        assert 1 <= definitions.min() <= definitions.max()
        assert definitions.max() <= sequence.num_block_definitions()


def test_a_signature_the_file_carries_verifies(corpus_file):
    """Every file that was signed was signed over the bytes it still has."""
    contents = corpus_file.read_bytes()
    if b"[SIGNATURE]" not in contents:
        pytest.skip("this file carries no signature")

    _ext.read(contents, True)


# --------------------------------------------------------------------------
# Reading a file older than 1.5.0.
#
# Every revision moved columns, and three fields arrived only in 1.5: an RF
# pulse's centre, and an arbitrary gradient's first and last sample. Before
# 1.4 a block's duration is not stored either -- the table holds an index into
# `[DELAYS]` and the duration is whatever the block plays. None of that can be
# defaulted, so all of it is derived, and the 1.5.0 file is the answer.
# --------------------------------------------------------------------------


def rendered(path):
    return normalise(_ext.write_text(_ext.read(path.read_bytes()), True))


@pytest.mark.parametrize("name", OLDER)
def test_an_older_file_holds_the_events_the_1_5_file_holds(name):
    older = _ext.read((CORPUS / name).read_bytes())
    current = _ext.read(AS_1_5.read_bytes())

    for count in ("num_blocks", "num_rf", "num_gradients", "num_adc"):
        assert getattr(older, count)() == getattr(current, count)(), count


@pytest.mark.parametrize("name", SHAPES_AS_1_5)
def test_a_1_4_file_holds_the_shapes_the_1_5_file_holds(name):
    assert (
        _ext.read((CORPUS / name).read_bytes()).num_shapes()
        == _ext.read(AS_1_5.read_bytes()).num_shapes()
    )


@pytest.mark.parametrize("name", DURATION_AS_1_5)
def test_an_older_file_lasts_as_long_as_the_1_5_file(name):
    """Before 1.4 the duration is derived from what each block plays."""
    assert _ext.read((CORPUS / name).read_bytes()).duration() == pytest.approx(
        _ext.read(AS_1_5.read_bytes()).duration()
    )


@pytest.mark.parametrize("name", SHAPES_AS_1_5)
def test_a_1_4_file_reads_as_the_1_5_file_but_for_what_it_cannot_carry(name):
    """Two things it cannot carry, and nothing else differs.

    A 1.4 file has no `use` column, so what each pulse is *for* is unknown
    and stays undefined rather than being guessed from its flip angle. And
    one gradient's `last` is derived by the extrapolation the factory would
    have made, which is close to but not the value the design knew: the
    reference toolbox derives the same number from the same file, so this is
    the format's limit rather than a difference between readers.
    """
    differing = [
        (before, after)
        for before, after in zip(
            rendered(AS_1_5).splitlines(),
            rendered(CORPUS / name).splitlines(),
            strict=False,
        )
        if before != after
    ]

    unknown_use = [
        pair
        for pair in differing
        if pair[0][:-1] == pair[1][:-1] and pair[1][-1] == "u"
    ]
    derived_edge = [pair for pair in differing if pair not in unknown_use]

    assert len(unknown_use) == _ext.read(AS_1_5.read_bytes()).num_rf()
    assert len(derived_edge) == 1


@pytest.mark.parametrize("name", OLDER)
def test_an_older_file_says_1_5_once_it_has_been_read(name):
    """It has been converted, so it is a 1.5 sequence and says so."""
    converted = _ext.write_text(_ext.read((CORPUS / name).read_bytes()), True)

    assert rows(converted, "[VERSION]") == ["major 1", "minor 5", "revision 1"]


@pytest.mark.parametrize("name", OLDER)
def test_a_converted_file_writes_the_same_bytes_twice(name):
    once = _ext.write_text(_ext.read((CORPUS / name).read_bytes()), True)

    assert _ext.write_text(_ext.read(once), True) == once


@pytest.mark.parametrize("name", OLDER)
def test_a_converted_file_carries_the_rasters_it_was_read_with(name):
    """A file that declared none of them says what it was taken to mean."""
    converted = _ext.read((CORPUS / name).read_bytes()).definitions()

    for raster in (
        "GradientRasterTime",
        "RadiofrequencyRasterTime",
        "AdcRasterTime",
        "BlockDurationRaster",
    ):
        assert raster in converted


def test_a_file_older_than_the_format_is_named_rather_than_misread():
    """1.2.0 is where the format is defined from."""
    contents = AS_1_5.read_bytes().replace(b"minor 5", b"minor 1", 1)

    with pytest.raises(RuntimeError, match=r"1\.2\.0 is the oldest"):
        _ext.read(contents)


def test_the_corpus_spans_the_revisions_it_is_here_for():
    """A guard on the corpus itself, so a lost file is noticed."""
    declared = set()
    for path in CORPUS.glob("*.seq"):
        text = path.read_text(errors="replace")
        version = re.search(
            r"\[VERSION\]\s*\nmajor (\d+)\s*\nminor (\d+)\s*\nrevision (\d+)", text
        )
        assert version, path.name
        declared.add(".".join(version.groups()))

    assert {"1.2.0", "1.3.0", "1.3.1", "1.4.0", "1.4.1", "1.4.2"} <= declared
    assert {"1.5.0", "1.5.1"} <= declared


def test_every_corpus_sequence_is_playable_on_the_rasters_it_declares(corpus_file):
    """A file the toolbox wrote holds no time its own rasters cannot address.

    Dead times and ringdown are a property of the scanner rather than of the
    file, so they are left at zero here and the margins they guard are not
    what is being judged: what is, is that every delay, dwell, ramp and block
    duration in the corpus lands on a tick.
    """
    sequence = _ext.read((CORPUS / corpus_file).read_bytes())
    declared = sequence.definitions()

    def raster(name):
        value = declared[name]
        return float(value[0] if isinstance(value, list) else value)

    problems = _ext.check_timing(
        sequence,
        rf_raster_time=raster("RadiofrequencyRasterTime"),
        grad_raster_time=raster("GradientRasterTime"),
        adc_raster_time=raster("AdcRasterTime"),
        block_duration_raster=raster("BlockDurationRaster"),
    )

    assert problems == []
