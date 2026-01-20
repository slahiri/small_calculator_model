"""
Data Generation for Calculator LLM

Generates training and test data for English math problems.
"""

import json
import random
from pathlib import Path


def num_to_words(n: int) -> str:
    """Convert a number (0-99) to English words."""
    if n < 0:
        return "negative " + num_to_words(-n)

    words_0_19 = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    ]

    if n < 20:
        return words_0_19[n]

    tens_words = [
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety",
    ]

    if n < 100:
        tens = tens_words[n // 10]
        ones = n % 10
        return tens if ones == 0 else f"{tens} {num_to_words(ones)}"

    return str(n)


def generate_all_combinations() -> list[dict]:
    """Generate all valid math problem combinations."""
    data = []

    # Addition: a + b where result <= 99
    for a in range(100):
        for b in range(100 - a):
            result = a + b
            data.append({
                "input": f"{num_to_words(a)} plus {num_to_words(b)}",
                "output": num_to_words(result),
                "full": f"{num_to_words(a)} plus {num_to_words(b)} equals {num_to_words(result)}",
            })

    # Subtraction: a - b where result >= 0
    for a in range(100):
        for b in range(a + 1):
            result = a - b
            data.append({
                "input": f"{num_to_words(a)} minus {num_to_words(b)}",
                "output": num_to_words(result),
                "full": f"{num_to_words(a)} minus {num_to_words(b)} equals {num_to_words(result)}",
            })

    # Multiplication: a * b where result <= 99
    for a in range(100):
        for b in range(100):
            result = a * b
            if result <= 99:
                data.append({
                    "input": f"{num_to_words(a)} times {num_to_words(b)}",
                    "output": num_to_words(result),
                    "full": f"{num_to_words(a)} times {num_to_words(b)} equals {num_to_words(result)}",
                })

    return data


def generate_train_test_split(
    test_ratio: float = 0.1,
    train_multiplier: int = 10,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Generate training and test data with no overlap.

    Args:
        test_ratio: Fraction of unique combinations to hold out for testing
        train_multiplier: How many times to duplicate training data
        seed: Random seed for reproducibility

    Returns:
        (train_data, test_data) tuple
    """
    random.seed(seed)

    # Generate all unique combinations
    all_data = generate_all_combinations()
    random.shuffle(all_data)

    # Split
    split_idx = int(len(all_data) * (1 - test_ratio))
    train_unique = all_data[:split_idx]
    test_data = all_data[split_idx:]

    # Duplicate training data
    train_data = train_unique * train_multiplier
    random.shuffle(train_data)

    return train_data, test_data


def save_data(
    output_dir: str | Path,
    test_ratio: float = 0.1,
    train_multiplier: int = 10,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Generate and save training/test data to JSON files.

    Returns:
        (num_train, num_test) tuple
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = generate_train_test_split(
        test_ratio=test_ratio,
        train_multiplier=train_multiplier,
        seed=seed,
    )

    # Save training data (compact format for size)
    with open(output_dir / "training_data.json", "w") as f:
        json.dump(train_data, f)

    # Save test data (pretty format for readability)
    with open(output_dir / "test_data.json", "w") as f:
        json.dump(test_data, f, indent=2)

    return len(train_data), len(test_data)


if __name__ == "__main__":
    # Generate data when run directly
    num_train, num_test = save_data("data")
    print(f"Generated {num_train:,} training examples")
    print(f"Generated {num_test:,} test examples")
