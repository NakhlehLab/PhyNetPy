"""
Use pytest to test the correctness of the Alphabet.py module.
"""

from PhyNetPy.Alphabet import *
import pytest
from dataclasses import FrozenInstanceError

def test_immutability():
    with pytest.raises(FrozenInstanceError):
        DNA.name = "DNA2"
    with pytest.raises(FrozenInstanceError):
        DNA.mapping = {"A": 1, "C": 2, "G": 3, "T": 4}

def test_snp_alphabet():
    snp_alphabet_value = snp_alphabet(2)
    assert snp_alphabet_value.name == "SNP"
    assert snp_alphabet_value.mapping == {"0": 0, "1": 1, "2": 2, "-": 3}

def test_alphabet_init():
    # Test standard alphabet
    my_alphabet = Alphabet(DNA)
    assert my_alphabet.alphabet.name == "DNA"
    assert my_alphabet.alphabet.mapping == {"-": 0, "A": 1, "C": 2, "M": 3, "G": 4, "R": 5, "S": 6, "V": 7, "T": 8, "W": 9, "Y": 10, "H": 11, "K": 12, "D": 13, "B": 14, "X": 15}
    assert my_alphabet._reverse_mapping == AlphabetMapping("DNA_REVERSE", {0: "-", 1: "A", 2: "C", 3: "M", 4: "G", 5: "R", 6: "S", 7: "V", 8: "T", 9: "W", 10: "Y", 11: "H", 12: "K", 13: "D", 14: "B", 15: "X"})
    assert my_alphabet.map("A") == 1
    assert my_alphabet.map("C") == 2
    assert my_alphabet.map("G") == 4
    assert my_alphabet.map("T") == 8
    assert my_alphabet.map("X") == 15
    assert my_alphabet.reverse_map(1) == "A"
    assert my_alphabet.reverse_map(2) == "C"
    assert my_alphabet.reverse_map(4) == "G"
    # Test user-defined alphabet
    my_alphabet = Alphabet(AlphabetMapping("USER", {"A": 1, "C": 2, "G": 3, "T": 4}))
    assert my_alphabet.alphabet.name == "USER"
    assert my_alphabet.alphabet.mapping == {"A": 1, "C": 2, "G": 3, "T": 4}
    assert my_alphabet._reverse_mapping == AlphabetMapping("USER_REVERSE", {1: "A", 2: "C", 3: "G", 4: "T"})
    assert my_alphabet.map("A") == 1
    assert my_alphabet.map('a') == 1
    assert my_alphabet.map("C") == 2
    assert my_alphabet.map("G") == 3
    assert my_alphabet.map("T") == 4
    assert my_alphabet.reverse_map(1) == "A"
    assert my_alphabet.reverse_map(2) == "C"
    assert my_alphabet.reverse_map(3) == "G"
    assert my_alphabet.reverse_map(4) == "T"

def test_bogus_map_input():
    my_alphabet = Alphabet(DNA)
    with pytest.raises(AlphabetError):
        my_alphabet.map("Z")
    with pytest.raises(AlphabetError):
        my_alphabet.reverse_map(-1)

def test_get_type():
    my_alphabet = Alphabet(RNA)
    assert my_alphabet.get_type() == "RNA"
    my_alphabet = Alphabet(AlphabetMapping("USER", {"A": 1, "C": 2, "G": 3, "T": 4}))
    assert my_alphabet.get_type() == "USER"
    
if __name__ == "__main__":
    test_get_type()