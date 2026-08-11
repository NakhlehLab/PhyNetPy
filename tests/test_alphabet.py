"""
Test suite for the Alphabet module (phynetpy.Alphabet).

Validates the behavior of alphabet mappings used throughout PhyNetPy for
encoding biological sequences (DNA, RNA, SNP, and user-defined alphabets).

Covers:
    - Immutability of built-in alphabet mappings (DNA, RNA, etc.)
    - SNP alphabet construction
    - Forward and reverse character-to-index mapping
    - Case-insensitive mapping for user-defined alphabets
    - Error handling for invalid map/reverse_map inputs
    - Alphabet type introspection via get_type()
"""

import pytest
from dataclasses import FrozenInstanceError

from phynetpy.Alphabet import (
    Alphabet,
    AlphabetMapping,
    AlphabetError,
    DNA,
    RNA,
    snp_alphabet,
)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestAlphabetImmutability:
    """Verify that built-in AlphabetMapping objects are frozen/immutable."""

    def test_dna_name_is_immutable(self):
        """Attempting to reassign the name of the DNA mapping should raise."""
        with pytest.raises(FrozenInstanceError):
            DNA.name = "DNA2"

    def test_dna_mapping_is_immutable(self):
        """Attempting to reassign the mapping dict of DNA should raise."""
        with pytest.raises(FrozenInstanceError):
            DNA.mapping = {"A": 1, "C": 2, "G": 3, "T": 4}


# ---------------------------------------------------------------------------
# SNP Alphabet
# ---------------------------------------------------------------------------

class TestSNPAlphabet:
    """Test dynamic SNP alphabet construction."""

    def test_snp_alphabet_name_and_mapping(self):
        """snp_alphabet(2) should produce a mapping with allele counts 0, 1, 2
        plus a gap character '-'."""
        snp = snp_alphabet(2)
        assert snp.name == "SNP"
        assert snp.mapping == {"0": 0, "1": 1, "2": 2, "-": 3}


# ---------------------------------------------------------------------------
# Alphabet Initialisation & Mapping
# ---------------------------------------------------------------------------

class TestAlphabetInit:
    """Test Alphabet wrapper initialisation for standard and custom mappings."""

    def test_standard_dna_alphabet(self):
        """The DNA alphabet should expose the full ambiguity-code mapping and
        build a correct reverse mapping."""
        alpha = Alphabet(DNA)

        assert alpha.alphabet.name == "DNA"
        assert alpha.alphabet.mapping == {
            "-": 0, "A": 1, "C": 2, "M": 3, "G": 4, "R": 5,
            "S": 6, "V": 7, "T": 8, "W": 9, "Y": 10, "H": 11,
            "K": 12, "D": 13, "B": 14, "X": 15,
        }

        # Reverse mapping round-trip
        expected_reverse = AlphabetMapping(
            "DNA_REVERSE",
            {0: "-", 1: "A", 2: "C", 3: "M", 4: "G", 5: "R",
             6: "S", 7: "V", 8: "T", 9: "W", 10: "Y", 11: "H",
             12: "K", 13: "D", 14: "B", 15: "X"},
        )
        assert alpha._reverse_mapping == expected_reverse

    def test_standard_dna_forward_and_reverse_mapping(self):
        """Individual forward/reverse lookups for the four standard bases."""
        alpha = Alphabet(DNA)

        for char, idx in [("A", 1), ("C", 2), ("G", 4), ("T", 8), ("X", 15)]:
            assert alpha.map(char) == idx
            assert alpha.reverse_map(idx) == char

    def test_user_defined_alphabet(self):
        """A custom AlphabetMapping should work exactly like the built-ins."""
        custom = AlphabetMapping("USER", {"A": 1, "C": 2, "G": 3, "T": 4})
        alpha = Alphabet(custom)

        assert alpha.alphabet.name == "USER"
        assert alpha.alphabet.mapping == {"A": 1, "C": 2, "G": 3, "T": 4}

        expected_reverse = AlphabetMapping(
            "USER_REVERSE", {1: "A", 2: "C", 3: "G", 4: "T"}
        )
        assert alpha._reverse_mapping == expected_reverse

    def test_user_defined_forward_and_reverse_mapping(self):
        """Forward and reverse mapping round-trips for a user-defined alphabet."""
        custom = AlphabetMapping("USER", {"A": 1, "C": 2, "G": 3, "T": 4})
        alpha = Alphabet(custom)

        for char, idx in [("A", 1), ("C", 2), ("G", 3), ("T", 4)]:
            assert alpha.map(char) == idx
            assert alpha.reverse_map(idx) == char

    def test_case_insensitive_mapping(self):
        """Lowercase input should be accepted and map to the same index."""
        custom = AlphabetMapping("USER", {"A": 1, "C": 2, "G": 3, "T": 4})
        alpha = Alphabet(custom)
        assert alpha.map("a") == 1


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestAlphabetErrors:
    """Verify that invalid inputs raise AlphabetError."""

    def test_invalid_character_raises(self):
        """Mapping a character not in the alphabet should raise AlphabetError."""
        alpha = Alphabet(DNA)
        with pytest.raises(AlphabetError):
            alpha.map("Z")

    def test_invalid_reverse_index_raises(self):
        """Reverse-mapping an index not in the alphabet should raise AlphabetError."""
        alpha = Alphabet(DNA)
        with pytest.raises(AlphabetError):
            alpha.reverse_map(-1)


# ---------------------------------------------------------------------------
# Type Introspection
# ---------------------------------------------------------------------------

class TestAlphabetGetType:
    """Test the get_type() convenience accessor."""

    def test_rna_type(self):
        """get_type() should return the name string of the underlying mapping."""
        alpha = Alphabet(RNA)
        assert alpha.get_type() == "RNA"

    def test_user_type(self):
        """Custom alphabets should report their name via get_type()."""
        custom = AlphabetMapping("USER", {"A": 1, "C": 2, "G": 3, "T": 4})
        alpha = Alphabet(custom)
        assert alpha.get_type() == "USER"
