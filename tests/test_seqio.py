import pytest

from ligase.utils.seqio import load_sequences


def test_fasta_multi_record(tmp_path):
    f = tmp_path / "s.fasta"
    f.write_text(">sp|P0|TEST Test protein\nMKTA\nYIAK\n>tr|X|other\nACDE\n")
    assert load_sequences(f) == ["MKTAYIAK", "ACDE"]


def test_txt_comments_and_blanks(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("# my favorites\nMKTA\n\nACDE\n")
    assert load_sequences(f) == ["MKTA", "ACDE"]


def test_csv_sequence_column(tmp_path):
    f = tmp_path / "s.csv"
    f.write_text("id,Sequence\na,MKTA\nb,ACDE\n")
    assert load_sequences(f) == ["MKTA", "ACDE"]


def test_csv_without_sequence_column(tmp_path):
    f = tmp_path / "s.csv"
    f.write_text("id,name\na,MKTA\nbACDE\n")
    with pytest.raises(ValueError, match="sequence"):
        load_sequences(f)


def test_unknown_extension_rejected(tmp_path):
    f = tmp_path / "s.xlsx"
    f.write_text("nope")
    with pytest.raises(ValueError, match="FASTA"):
        load_sequences(f)


def test_file_to_embed_alignment(mock_source, tmp_path):
    """The point of it all: file -> loader -> embed -> (L, D) with L == len(seq)."""
    from ligase.utils.caching import cached

    f = tmp_path / "s.fasta"
    f.write_text(">x\nMKTAYIAK\n")
    out = cached(mock_source, tmp_path / "c").embed(load_sequences(f))
    assert out["MKTAYIAK"].shape == (8, mock_source.dim)
