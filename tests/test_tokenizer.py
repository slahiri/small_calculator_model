"""Tests for the Tokenizer class."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tokenizer import Tokenizer


def test_vocab_size():
    """Test that vocabulary size is correct."""
    tokenizer = Tokenizer()
    assert tokenizer.vocab_size == 36, f"Expected vocab_size=36, got {tokenizer.vocab_size}"
    print("✓ vocab_size = 36")


def test_special_tokens():
    """Test special token IDs."""
    tokenizer = Tokenizer()
    assert tokenizer.pad_token_id == 0, f"Expected pad_token_id=0, got {tokenizer.pad_token_id}"
    assert tokenizer.start_token_id == 1, f"Expected start_token_id=1, got {tokenizer.start_token_id}"
    assert tokenizer.end_token_id == 2, f"Expected end_token_id=2, got {tokenizer.end_token_id}"
    print("✓ Special tokens correct: [PAD]=0, [START]=1, [END]=2")


def test_encode_basic():
    """Test basic encoding."""
    tokenizer = Tokenizer()

    # Test simple expressions
    assert tokenizer.encode("two plus three") == [5, 31, 6]
    assert tokenizer.encode("seven minus four") == [10, 32, 7]
    assert tokenizer.encode("twenty divided by five") == [23, 34, 35, 8]
    assert tokenizer.encode("six times eight") == [9, 33, 11]

    print("✓ Basic encoding works")


def test_encode_with_special_tokens():
    """Test encoding with special tokens."""
    tokenizer = Tokenizer()

    result = tokenizer.encode("two plus three", add_special_tokens=True)
    assert result == [1, 5, 31, 6, 2], f"Expected [1, 5, 31, 6, 2], got {result}"

    print("✓ Encoding with special tokens works")


def test_decode_basic():
    """Test basic decoding."""
    tokenizer = Tokenizer()

    assert tokenizer.decode([5, 31, 6]) == "two plus three"
    assert tokenizer.decode([8]) == "five"
    assert tokenizer.decode([25, 5]) == "forty two"  # 42 as two tokens

    print("✓ Basic decoding works")


def test_decode_with_special_tokens():
    """Test decoding with special tokens."""
    tokenizer = Tokenizer()

    # Skip special tokens by default
    result = tokenizer.decode([1, 5, 31, 6, 2], skip_special_tokens=True)
    assert result == "two plus three", f"Expected 'two plus three', got '{result}'"

    # Include special tokens
    result = tokenizer.decode([1, 5, 31, 6, 2], skip_special_tokens=False)
    assert result == "[START] two plus three [END]", f"Expected '[START] two plus three [END]', got '{result}'"

    print("✓ Decoding with special tokens works")


def test_num_to_words():
    """Test number to words conversion."""
    tokenizer = Tokenizer()

    # Single digit and teens
    assert tokenizer.num_to_words(0) == "zero"
    assert tokenizer.num_to_words(5) == "five"
    assert tokenizer.num_to_words(13) == "thirteen"
    assert tokenizer.num_to_words(19) == "nineteen"

    # Tens
    assert tokenizer.num_to_words(20) == "twenty"
    assert tokenizer.num_to_words(30) == "thirty"
    assert tokenizer.num_to_words(90) == "ninety"

    # Compound numbers
    assert tokenizer.num_to_words(21) == "twenty one"
    assert tokenizer.num_to_words(42) == "forty two"
    assert tokenizer.num_to_words(99) == "ninety nine"
    assert tokenizer.num_to_words(55) == "fifty five"

    print("✓ num_to_words works for 0-99")


def test_words_to_num():
    """Test words to number conversion."""
    tokenizer = Tokenizer()

    # Single digit and teens
    assert tokenizer.words_to_num("zero") == 0
    assert tokenizer.words_to_num("five") == 5
    assert tokenizer.words_to_num("thirteen") == 13
    assert tokenizer.words_to_num("nineteen") == 19

    # Tens
    assert tokenizer.words_to_num("twenty") == 20
    assert tokenizer.words_to_num("thirty") == 30
    assert tokenizer.words_to_num("ninety") == 90

    # Compound numbers
    assert tokenizer.words_to_num("twenty one") == 21
    assert tokenizer.words_to_num("forty two") == 42
    assert tokenizer.words_to_num("ninety nine") == 99
    assert tokenizer.words_to_num("fifty five") == 55

    print("✓ words_to_num works for 0-99")


def test_roundtrip():
    """Test that num_to_words and words_to_num are inverses."""
    tokenizer = Tokenizer()

    for n in range(100):
        words = tokenizer.num_to_words(n)
        result = tokenizer.words_to_num(words)
        assert result == n, f"Roundtrip failed for {n}: {words} -> {result}"

    print("✓ Roundtrip conversion works for all 0-99")


def test_encode_decode_roundtrip():
    """Test that encode and decode are inverses."""
    tokenizer = Tokenizer()

    test_texts = [
        "two plus three",
        "seven minus four",
        "six times eight",
        "twenty divided by five",
        "zero",
        "ninety nine",
        "forty two",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Roundtrip failed: '{text}' -> {encoded} -> '{decoded}'"

    print("✓ Encode/decode roundtrip works")


def test_case_insensitivity():
    """Test that encoding is case-insensitive."""
    tokenizer = Tokenizer()

    assert tokenizer.encode("Two Plus Three") == tokenizer.encode("two plus three")
    assert tokenizer.encode("SEVEN MINUS FOUR") == tokenizer.encode("seven minus four")

    print("✓ Encoding is case-insensitive")


def test_all_tokens_have_ids():
    """Test that all expected tokens are in vocabulary."""
    tokenizer = Tokenizer()

    # Check all ones
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    for word in ones:
        assert word in tokenizer.get_vocab(), f"Missing: {word}"

    # Check all tens
    tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    for word in tens:
        assert word in tokenizer.get_vocab(), f"Missing: {word}"

    # Check operations
    ops = ["plus", "minus", "times", "divided", "by"]
    for word in ops:
        assert word in tokenizer.get_vocab(), f"Missing: {word}"

    # Check special tokens
    special = ["[PAD]", "[START]", "[END]"]
    for word in special:
        assert word in tokenizer.get_vocab(), f"Missing: {word}"

    print("✓ All expected tokens are in vocabulary")


def run_all_tests():
    """Run all tokenizer tests."""
    print("=" * 50)
    print("Running Tokenizer Tests")
    print("=" * 50)

    test_vocab_size()
    test_special_tokens()
    test_encode_basic()
    test_encode_with_special_tokens()
    test_decode_basic()
    test_decode_with_special_tokens()
    test_num_to_words()
    test_words_to_num()
    test_roundtrip()
    test_encode_decode_roundtrip()
    test_case_insensitivity()
    test_all_tokens_have_ids()

    print("=" * 50)
    print("All tokenizer tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
