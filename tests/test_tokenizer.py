"""Tests for tokenizer.py - Run with: pytest tests/test_tokenizer.py -v"""

import pytest
import sys
sys.path.insert(0, 'src')

from tokenizer import Tokenizer, pad_sequence


class TestTokenizer:
    """Test the Tokenizer class."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer with our calculator vocabulary."""
        vocab = {
            "[PAD]": 0, "[START]": 1, "[END]": 2, "[UNK]": 3,
            "zero": 4, "one": 5, "two": 6, "three": 7, "four": 8,
            "five": 9, "six": 10, "seven": 11, "eight": 12, "nine": 13,
            "ten": 14, "eleven": 15, "twelve": 16, "thirteen": 17,
            "fourteen": 18, "fifteen": 19, "sixteen": 20, "seventeen": 21,
            "eighteen": 22, "nineteen": 23,
            "twenty": 24, "thirty": 25, "forty": 26, "fifty": 27,
            "sixty": 28, "seventy": 29, "eighty": 30, "ninety": 31,
            "plus": 32, "minus": 33, "times": 34, "equals": 35,
        }
        return Tokenizer(vocab)

    # === Test encode() ===

    def test_encode_simple(self, tokenizer):
        """Test encoding a simple expression."""
        ids = tokenizer.encode("two plus three")
        assert ids == [1, 6, 32, 7, 2]  # [START] two plus three [END]

    def test_encode_without_special_tokens(self, tokenizer):
        """Test encoding without special tokens."""
        ids = tokenizer.encode("two plus three", add_special_tokens=False)
        assert ids == [6, 32, 7]  # two plus three

    def test_encode_unknown_word(self, tokenizer):
        """Test that unknown words get [UNK] token."""
        ids = tokenizer.encode("two plus banana")
        assert ids == [1, 6, 32, 3, 2]  # [START] two plus [UNK] [END]

    def test_encode_all_operations(self, tokenizer):
        """Test encoding all supported operations."""
        assert 32 in tokenizer.encode("one plus one")    # plus
        assert 33 in tokenizer.encode("one minus one")   # minus
        assert 34 in tokenizer.encode("one times one")   # times
        assert 35 in tokenizer.encode("one equals one")  # equals

    # === Test decode() ===

    def test_decode_simple(self, tokenizer):
        """Test decoding token IDs back to text."""
        text = tokenizer.decode([1, 6, 32, 7, 2])
        assert text == "two plus three"

    def test_decode_with_special_tokens(self, tokenizer):
        """Test decoding keeps special tokens when asked."""
        text = tokenizer.decode([1, 6, 32, 7, 2], skip_special_tokens=False)
        assert "[START]" in text
        assert "[END]" in text

    def test_encode_decode_roundtrip(self, tokenizer):
        """Test that encode -> decode returns original text."""
        original = "five times seven"
        ids = tokenizer.encode(original)
        decoded = tokenizer.decode(ids)
        assert decoded == original

    # === Test normalize() ===

    def test_normalize_symbols(self, tokenizer):
        """Test that symbols are converted to words."""
        assert tokenizer.normalize("2 + 3") == "2 plus 3"
        assert tokenizer.normalize("5 - 2") == "5 minus 2"
        assert tokenizer.normalize("4 * 3") == "4 times 3"
        assert tokenizer.normalize("1 = 1") == "1 equals 1"

    def test_normalize_compound_numbers(self, tokenizer):
        """Test that compound numbers are split."""
        assert tokenizer.normalize("twentythree") == "twenty three"
        assert tokenizer.normalize("fortyfive") == "forty five"
        assert tokenizer.normalize("ninetynine") == "ninety nine"

    def test_normalize_case_insensitive(self, tokenizer):
        """Test that input is lowercased."""
        assert tokenizer.normalize("TWO PLUS THREE") == "two plus three"
        assert tokenizer.normalize("Two Plus Three") == "two plus three"

    def test_normalize_extra_whitespace(self, tokenizer):
        """Test that extra whitespace is normalized."""
        assert tokenizer.normalize("two  plus   three") == "two plus three"
        assert tokenizer.normalize("  two plus three  ") == "two plus three"

    # === Test special token properties ===

    def test_special_token_ids(self, tokenizer):
        """Test that special token IDs are accessible."""
        assert tokenizer.pad_token_id == 0
        assert tokenizer.start_token_id == 1
        assert tokenizer.end_token_id == 2


class TestPadSequence:
    """Test the pad_sequence utility function."""

    def test_pad_short_sequence(self):
        """Test padding a sequence shorter than max_len."""
        result = pad_sequence([1, 2, 3], max_len=5, pad_token_id=0)
        assert result == [1, 2, 3, 0, 0]

    def test_truncate_long_sequence(self):
        """Test truncating a sequence longer than max_len."""
        result = pad_sequence([1, 2, 3, 4, 5, 6], max_len=4, pad_token_id=0)
        assert result == [1, 2, 3, 4]

    def test_exact_length_unchanged(self):
        """Test that exact-length sequences are unchanged."""
        result = pad_sequence([1, 2, 3], max_len=3, pad_token_id=0)
        assert result == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
