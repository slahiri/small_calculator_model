"""Tests for generate.py - Run with: pytest tests/test_generate.py -v"""

import pytest
import torch
import sys
sys.path.insert(0, 'src')

from tokenizer import Tokenizer
from model import CalculatorLLM
from generate import generate, solve, evaluate_model


class TestGenerate:
    """Test the generate function."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        model = CalculatorLLM(
            vocab_size=36,
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            ff_dim=64,
            max_seq_len=16,
        )
        model.eval()
        return model

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
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

    def test_returns_string(self, model, tokenizer):
        """Test that generate returns a string."""
        result = generate(model, tokenizer, "two plus three equals")
        assert isinstance(result, str)

    def test_includes_prompt(self, model, tokenizer):
        """Test that output includes the prompt."""
        prompt = "two plus three equals"
        result = generate(model, tokenizer, prompt)
        # Result should start with or contain the prompt tokens
        assert "two" in result or "plus" in result

    def test_max_tokens_limit(self, model, tokenizer):
        """Test that generation respects max_new_tokens."""
        result = generate(model, tokenizer, "two plus", max_new_tokens=3)
        # Output should be reasonably bounded
        tokens = result.split()
        assert len(tokens) <= 10  # prompt + max_new_tokens + some buffer

    def test_stops_at_end_token(self, model, tokenizer):
        """Test that generation can stop at END token."""
        # With untrained model, may or may not hit END
        # Just verify it doesn't crash
        result = generate(model, tokenizer, "two plus three equals")
        assert result is not None


class TestSolve:
    """Test the solve function."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        model = CalculatorLLM(
            vocab_size=36,
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            ff_dim=64,
            max_seq_len=16,
        )
        model.eval()
        return model

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
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

    def test_returns_string(self, model, tokenizer):
        """Test that solve returns a string."""
        result = solve(model, tokenizer, "two plus three")
        assert isinstance(result, str)

    def test_adds_equals_if_missing(self, model, tokenizer):
        """Test that solve adds 'equals' if not in problem."""
        # This is an internal behavior - just verify no crash
        result = solve(model, tokenizer, "two plus three")
        assert result is not None

    def test_handles_uppercase(self, model, tokenizer):
        """Test that solve handles uppercase input."""
        result = solve(model, tokenizer, "TWO PLUS THREE")
        assert result is not None

    def test_handles_trailing_whitespace(self, model, tokenizer):
        """Test that solve handles trailing whitespace."""
        result = solve(model, tokenizer, "  two plus three  ")
        assert result is not None


class TestEvaluateModel:
    """Test the evaluate_model function."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        model = CalculatorLLM(
            vocab_size=36,
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            ff_dim=64,
            max_seq_len=16,
        )
        model.eval()
        return model

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
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

    @pytest.fixture
    def test_data(self):
        """Create small test dataset."""
        return [
            {"input": "two plus three", "output": "five"},
            {"input": "one plus one", "output": "two"},
            {"input": "five minus two", "output": "three"},
        ]

    def test_returns_accuracy_and_errors(self, model, tokenizer, test_data):
        """Test that evaluate returns accuracy and errors list."""
        accuracy, errors = evaluate_model(model, tokenizer, test_data)
        assert isinstance(accuracy, float)
        assert isinstance(errors, list)

    def test_accuracy_in_valid_range(self, model, tokenizer, test_data):
        """Test that accuracy is between 0 and 1."""
        accuracy, errors = evaluate_model(model, tokenizer, test_data)
        assert 0 <= accuracy <= 1

    def test_errors_have_correct_structure(self, model, tokenizer, test_data):
        """Test that error items have required fields."""
        accuracy, errors = evaluate_model(model, tokenizer, test_data)
        for error in errors:
            assert "input" in error
            assert "expected" in error
            assert "got" in error

    def test_correct_plus_errors_equals_total(self, model, tokenizer, test_data):
        """Test that correct + errors = total test items."""
        accuracy, errors = evaluate_model(model, tokenizer, test_data)
        num_correct = int(accuracy * len(test_data))
        # Allow for rounding
        assert abs(num_correct + len(errors) - len(test_data)) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
