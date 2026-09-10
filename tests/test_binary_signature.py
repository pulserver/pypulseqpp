"""Binary MD5 storage, verification and reference-toolbox interoperability."""

import numpy as np
import pytest

import pypulseqpp as pp

toolbox = pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the format; see reference.py",
)


@pytest.fixture
def system():
    return pp.Opts()


@pytest.fixture(params=["sampled", "excited"])
def sequence(request, system):
    """Two sequences, one holding a shape and one not."""
    built = pp.Sequence(system)
    if request.param == "sampled":
        built.add_block(
            pp.make_trapezoid("x", area=1000, duration=1e-3, system=system),
            pp.make_adc(128, duration=1e-3, system=system),
        )
        built.add_block(pp.make_delay(2e-3))
    else:
        pulse, slice_select, _ = pp.make_sinc_pulse(
            np.pi / 2,
            duration=2e-3,
            slice_thickness=5e-3,
            apodization=0.5,
            time_bw_product=4,
            system=system,
            use="excitation",
            return_gz=True,
        )
        built.add_block(pulse, slice_select)
        built.add_block(
            pp.make_trapezoid(
                "z", area=-slice_select.area / 2, duration=1e-3, system=system
            )
        )
        built.add_block(pp.make_delay(1e-3))
    return built


def test_a_written_file_carries_the_signature_it_reports(sequence, system, tmp_path):
    path = tmp_path / "signed.bseq"
    written = sequence.write_binary(str(path))

    reread = pp.Sequence(system)
    reread.read_binary(str(path))

    assert reread.signature_type == "md5"
    assert reread.signature_file == "bin"
    assert reread.signature_value == written


def test_the_toolbox_agrees_about_the_digest(sequence, tmp_path):
    """Ours over our bytes is theirs over the same bytes, or neither means
    anything."""
    path = tmp_path / "signed.bseq"
    written = sequence.write_binary(str(path))

    valid, stored, computed = toolbox.verify_file_signature(path)

    assert valid
    assert stored == computed == written


def test_a_file_the_toolbox_signed_verifies_here(system, tmp_path):
    theirs = toolbox.Sequence()
    theirs.add_block(
        toolbox.make_trapezoid("x", area=1000, duration=1e-3),
        toolbox.make_adc(128, duration=1e-3),
    )
    path = tmp_path / "theirs.bseq"
    theirs.write_binary(str(path))

    ours = pp.Sequence(system)
    ours.read(str(path), verify=True)

    assert ours.signature_value == theirs.signature_value


def test_a_patched_file_is_refused(sequence, system, tmp_path):
    """A byte the parser would not notice is a byte the signature does."""
    path = tmp_path / "signed.bseq"
    sequence.write_binary(str(path))
    patched = bytearray(path.read_bytes())
    patched[len(patched) // 2] ^= 0x01
    path.write_bytes(bytes(patched))

    with pytest.raises(RuntimeError, match="signature"):
        pp.Sequence(system).read(str(path), verify=True)


def test_an_unsigned_file_is_read_without_complaint(sequence, system, tmp_path):
    """Nothing to check is not the same as a check that fails."""
    path = tmp_path / "unsigned.bseq"
    sequence.write_binary(str(path), create_signature=False)

    reread = pp.Sequence(system)
    reread.read(str(path), verify=True)

    assert reread.signature_value is None
    assert len(reread) == len(sequence)
