"""
Data Generation for Calculator LLM

Generates training and test data for English math problems.

Our calculator handles three operations:
- Addition: a + b where result <= 99
- Subtraction: a - b where result >= 0
- Multiplication: a * b where result <= 99

All problems and answers use English words (e.g., "two plus three equals five")
"""

import json
import random
from pathlib import Path


def num_to_words(n: int) -> str:
    """
    Convert a number (0-99) to English words.

    This handles the English number system:
    - 0-19: unique words (zero, one, ..., nineteen)
    - 20-99: compound words (twenty, twenty one, ..., ninety nine)

    Args:
        n: Integer to convert (typically 0-99)

    Returns:
        English word representation

    Examples:
        >>> num_to_words(5)
        'five'
        >>> num_to_words(42)
        'forty two'
        >>> num_to_words(-3)
        'negative three'
    """
    # Handle negative numbers recursively
    if n < 0:
        return "negative " + num_to_words(-n)

    # Words for 0-19 (each is unique in English)
    words_0_19 = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    ]

    # Simple lookup for 0-19
    if n < 20:
        return words_0_19[n]

    # Words for tens place (20, 30, ..., 90)
    tens_words = [
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety",
    ]

    # Compound numbers: "twenty" or "twenty one", etc.
    if n < 100:
        tens = tens_words[n // 10]  # e.g., 42 // 10 = 4 -> "forty"
        ones = n % 10               # e.g., 42 % 10 = 2
        return tens if ones == 0 else f"{tens} {num_to_words(ones)}"

    # Numbers >= 100 return as string (not supported in our vocab)
    return str(n)


def generate_all_combinations() -> list[dict]:
    """
    Generate all valid math problem combinations.

    We constrain our problems so that:
    - All inputs are 0-99
    - All outputs are 0-99 (no negative or 100+ results)

    This creates a bounded problem space that our small model can learn.

    Returns:
        List of dicts with 'input', 'output', and 'full' keys

    Example entry:
        {
            "input": "two plus three",
            "output": "five",
            "full": "two plus three equals five"
        }
    """
    data = []

    # === Addition: a + b where result <= 99 ===
    # For each a, b can only go up to (99 - a) to keep result <= 99
    for a in range(100):
        for b in range(100 - a):  # b goes from 0 to (99-a)
            result = a + b
            data.append({
                "input": f"{num_to_words(a)} plus {num_to_words(b)}",
                "output": num_to_words(result),
                "full": f"{num_to_words(a)} plus {num_to_words(b)} equals {num_to_words(result)}",
            })

    # === Subtraction: a - b where result >= 0 ===
    # For each a, b can only go up to a to keep result >= 0
    for a in range(100):
        for b in range(a + 1):  # b goes from 0 to a
            result = a - b
            data.append({
                "input": f"{num_to_words(a)} minus {num_to_words(b)}",
                "output": num_to_words(result),
                "full": f"{num_to_words(a)} minus {num_to_words(b)} equals {num_to_words(result)}",
            })

    # === Multiplication: a * b where result <= 99 ===
    # Only include pairs where product doesn't exceed 99
    for a in range(100):
        for b in range(100):
            result = a * b
            if result <= 99:  # Skip if result would be >= 100
                data.append({
                    "input": f"{num_to_words(a)} times {num_to_words(b)}",
                    "output": num_to_words(result),
                    "full": f"{num_to_words(a)} times {num_to_words(b)} equals {num_to_words(result)}",
                })

    return data


def generate_train_test_split(
    train_multiplier: int = 10,
    test_size: int = 500,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Generate training and test data.

    Training includes ALL valid combinations (repeated for more exposure).
    Test is a random sample (may overlap with training - that's OK for this demo).

    Why repeat training data?
    - More epochs over the same data helps the model learn patterns
    - Shuffling repeated data provides variety in batches

    Args:
        train_multiplier: How many times to duplicate training data
        test_size: Number of test examples to generate
        seed: Random seed for reproducibility

    Returns:
        (train_data, test_data) tuple
    """
    random.seed(seed)

    # Generate all unique combinations for training
    all_data = generate_all_combinations()

    # Training: all combinations repeated train_multiplier times
    # This gives the model more exposure to all problems
    train_data = all_data * train_multiplier
    random.shuffle(train_data)  # Shuffle to mix different problem types

    # Test: random sample from all combinations
    test_data = random.sample(all_data, min(test_size, len(all_data)))

    return train_data, test_data


def save_data(
    output_dir: str | Path,
    train_multiplier: int = 10,
    test_size: int = 500,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Generate and save training/test data to JSON files.

    Creates:
    - training_data.json: Compact format (for size)
    - test_data.json: Pretty-printed (for readability)

    Args:
        output_dir: Directory to save data files
        train_multiplier: How many times to duplicate training data
        test_size: Number of test examples
        seed: Random seed for reproducibility

    Returns:
        (num_train, num_test) tuple with counts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the data
    train_data, test_data = generate_train_test_split(
        train_multiplier=train_multiplier,
        test_size=test_size,
        seed=seed,
    )

    # Save training data (compact format for smaller file size)
    with open(output_dir / "training_data.json", "w") as f:
        json.dump(train_data, f)

    # Save test data (pretty format so humans can read it easily)
    with open(output_dir / "test_data.json", "w") as f:
        json.dump(test_data, f, indent=2)

    return len(train_data), len(test_data)


if __name__ == "__main__":
    # Generate data when run directly: python data.py
    num_train, num_test = save_data("data")
    print(f"Generated {num_train:,} training examples")
    print(f"Generated {num_test:,} test examples")
