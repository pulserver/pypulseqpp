import pytest

import pypulseqpp as pp


def _sequence():
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(pp.make_delay(1e-3))
    return sequence


def test_read_skips_an_unknown_extension_specification(tmp_path):
    path = tmp_path / "plain.seq"
    _sequence().write(path, create_signature=False)
    text = (
        path.read_text().rstrip("\n")
        + "\n\n[EXTENSIONS]\nextension VENDOR_NOTES 42\n1 7\n"
    )
    path.write_text(text)
    loaded = pp.Sequence()
    loaded.read(path)
    assert loaded.num_blocks == 1


def test_verify_refuses_a_text_file_without_a_signature(tmp_path):
    path = tmp_path / "unsigned.seq"
    _sequence().write(path, create_signature=False)
    with pytest.raises(RuntimeError, match="no signature"):
        pp.Sequence().read(path, verify=True)


def test_verify_refuses_a_binary_file_without_a_signature(tmp_path):
    path = tmp_path / "unsigned.bseq"
    _sequence().write_binary(path, create_signature=False)
    with pytest.raises(RuntimeError, match="no signature"):
        pp.Sequence().read(path, verify=True)
