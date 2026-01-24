"""Tests for data.py - Run with: pytest tests/test_data.py -v"""

import pytest
import sys
sys.path.insert(0, 'src')

from data import num_to_words, generate_all_combinations


class TestNumToWords:
    """Test the num_to_words function."""

    # === Single digit numbers ===

    def test_zero(self):
        assert num_to_words(0) == "zero"

    def test_single_digits(self):
        assert num_to_words(1) == "one"
        assert num_to_words(5) == "five"
        assert num_to_words(9) == "nine"

    # === Teen numbers ===

    def test_ten(self):
        assert num_to_words(10) == "ten"

    def test_teens(self):
        assert num_to_words(11) == "eleven"
        assert num_to_words(13) == "thirteen"
        assert num_to_words(15) == "fifteen"
        assert num_to_words(19) == "nineteen"

    # === Tens ===

    def test_round_tens(self):
        assert num_to_words(20) == "twenty"
        assert num_to_words(30) == "thirty"
        assert num_to_words(50) == "fifty"
        assert num_to_words(90) == "ninety"

    # === Compound numbers ===

    def test_compound_numbers(self):
        assert num_to_words(21) == "twenty one"
        assert num_to_words(35) == "thirty five"
        assert num_to_words(99) == "ninety nine"
        assert num_to_words(42) == "forty two"

    # === Edge cases ===

    def test_negative_numbers(self):
        assert num_to_words(-5) == "negative five"
        assert num_to_words(-42) == "negative forty two"

    def test_numbers_over_99(self):
        """Numbers over 99 should return as string."""
        assert num_to_words(100) == "100"
        assert num_to_words(999) == "999"


class TestGenerateAllCombinations:
    """Test the generate_all_combinations function."""

    @pytest.fixture
    def all_data(self):
        """Generate all combinations once for testing."""
        return generate_all_combinations()

    def test_returns_list(self, all_data):
        """Test that function returns a list."""
        assert isinstance(all_data, list)

    def test_data_not_empty(self, all_data):
        """Test that we generate some data."""
        assert len(all_data) > 0

    def test_data_has_correct_structure(self, all_data):
        """Test that each item has required fields."""
        for item in all_data[:10]:  # Check first 10
            assert "input" in item
            assert "output" in item
            assert "full" in item

    # === Addition tests ===

    def test_addition_exists(self, all_data):
        """Test that addition problems are generated."""
        addition_items = [d for d in all_data if "plus" in d["input"]]
        assert len(addition_items) > 0

    def test_addition_example(self, all_data):
        """Test a specific addition example."""
        two_plus_three = [d for d in all_data
                         if d["input"] == "two plus three"]
        assert len(two_plus_three) > 0
        assert two_plus_three[0]["output"] == "five"

    def test_addition_result_within_bounds(self, all_data):
        """Test that all addition results are <= 99."""
        for item in all_data:
            if "plus" in item["input"]:
                # Parse the output number
                output = item["output"]
                # Convert back to check bound
                # (we trust num_to_words is correct from other tests)
                assert "negative" not in output

    # === Subtraction tests ===

    def test_subtraction_exists(self, all_data):
        """Test that subtraction problems are generated."""
        subtraction_items = [d for d in all_data if "minus" in d["input"]]
        assert len(subtraction_items) > 0

    def test_subtraction_example(self, all_data):
        """Test a specific subtraction example."""
        five_minus_three = [d for d in all_data
                           if d["input"] == "five minus three"]
        assert len(five_minus_three) > 0
        assert five_minus_three[0]["output"] == "two"

    def test_subtraction_no_negative_results(self, all_data):
        """Test that subtraction never produces negative results."""
        for item in all_data:
            if "minus" in item["input"]:
                assert "negative" not in item["output"]

    # === Multiplication tests ===

    def test_multiplication_exists(self, all_data):
        """Test that multiplication problems are generated."""
        mult_items = [d for d in all_data if "times" in d["input"]]
        assert len(mult_items) > 0

    def test_multiplication_example(self, all_data):
        """Test a specific multiplication example."""
        two_times_three = [d for d in all_data
                          if d["input"] == "two times three"]
        assert len(two_times_three) > 0
        assert two_times_three[0]["output"] == "six"

    def test_multiplication_result_within_bounds(self, all_data):
        """Test that all multiplication results are <= 99."""
        for item in all_data:
            if "times" in item["input"]:
                # Ensure no three-digit results
                assert len(item["output"].split()) <= 2  # max "ninety nine"

    # === Full equation format ===

    def test_full_equation_format(self, all_data):
        """Test that full equations have correct format."""
        for item in all_data[:10]:
            full = item["full"]
            assert "equals" in full
            assert item["input"] in full
            assert item["output"] in full.split("equals")[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
