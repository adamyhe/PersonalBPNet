import numpy as np
import pytest

from personal_bpnet.utils import (
    get_twohot_fasta_sequences,
    reverse_complement_twohot,
    twohot_encode,
)

# IUPAC ambiguity codes and their expected complement code
_COMPLEMENTS = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "M": "K",  # A/C -> G/T
    "R": "Y",  # A/G -> C/T
    "W": "W",  # A/T is self-complementary
    "S": "S",  # C/G is self-complementary
    "Y": "R",  # C/T -> A/G
    "K": "M",  # G/T -> A/C
}


class TestTwohotEncode:
    def test_homozygous_bases_are_one_hot(self):
        encoded = twohot_encode("ACGT")
        assert encoded.shape == (4, 4)
        assert np.array_equal(encoded, np.eye(4))

    def test_heterozygote_sums_to_one(self):
        for code in ("M", "R", "W", "S", "Y", "K"):
            encoded = twohot_encode(code)
            assert np.isclose(encoded.sum(), 1.0)
            assert np.count_nonzero(encoded) == 2

    def test_n_encodes_to_all_zero(self):
        encoded = twohot_encode("N")
        assert np.array_equal(encoded, np.zeros((4, 1)))

    def test_lowercase_is_uppercased(self):
        assert np.array_equal(twohot_encode("acgt"), twohot_encode("ACGT"))

    def test_unrecognized_character_falls_back_to_all_zero(self):
        # e.g. an alignment gap character or an IUPAC code outside the ones this
        # encoding supports.
        encoded = twohot_encode("-")
        assert np.array_equal(encoded, np.zeros((4, 1)))


class TestReverseComplementTwohot:
    def test_does_not_mutate_input(self):
        for code in _COMPLEMENTS:
            encoded = twohot_encode(code)
            original = encoded.copy()
            reverse_complement_twohot(encoded)
            assert np.array_equal(encoded, original)

    def test_matches_expected_iupac_complement(self):
        for code, complement in _COMPLEMENTS.items():
            actual = reverse_complement_twohot(twohot_encode(code))
            expected = twohot_encode(complement)
            assert np.allclose(actual, expected)

    def test_reverses_position_order(self):
        encoded = twohot_encode("AC")
        rc = reverse_complement_twohot(encoded)
        # reverse-complement of "AC" is "GT"
        assert np.allclose(rc, twohot_encode("GT"))

    def test_round_trip_is_identity(self):
        encoded = twohot_encode("ACGTMRWSYK")
        assert np.allclose(
            reverse_complement_twohot(reverse_complement_twohot(encoded)), encoded
        )


class TestGetTwohotFastaSequences:
    def _write_fasta(self, path, seqs):
        with open(path, "w") as f:
            for i, seq in enumerate(seqs):
                f.write(f">seq{i}\n{seq}\n")
        return path

    def test_matches_per_sequence_encoding(self, tmp_path):
        length = 50
        bases = "ACGTMRWSYK"
        seqs = [(bases[i:] + bases[:i]) * 5 for i in range(3)]
        seqs = [seq[:length] for seq in seqs]
        assert len({len(seq) for seq in seqs}) == 1

        fasta = self._write_fasta(tmp_path / "equal.fa", seqs)

        actual = get_twohot_fasta_sequences(str(fasta))
        expected = np.stack([twohot_encode(seq) for seq in seqs], axis=0)

        assert actual.shape == expected.shape
        assert np.array_equal(actual, expected)

    def test_raises_on_unequal_length_records(self, tmp_path):
        fasta = self._write_fasta(tmp_path / "unequal.fa", ["ACGT", "ACGTAA"])

        with pytest.raises(ValueError, match="same length"):
            get_twohot_fasta_sequences(str(fasta))
