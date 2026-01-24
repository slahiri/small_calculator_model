"""
Tokenizer for Calculator LLM

Converts English math expressions to token IDs and back.
This is the first step in our LLM pipeline - converting human-readable
text into numbers that the neural network can process.
"""

import json
import re
from pathlib import Path


class Tokenizer:
    """
    Tokenizer for English math expressions.

    The tokenizer handles three main tasks:
    1. Normalize text (lowercase, replace symbols with words)
    2. Encode text to token IDs (words → numbers)
    3. Decode token IDs back to text (numbers → words)

    Our vocabulary has 36 tokens:
    - 4 special tokens: [PAD], [START], [END], [UNK]
    - 28 number words: zero through nineteen, plus tens (twenty, thirty, etc.)
    - 4 operation words: plus, minus, times, equals
    """

    def __init__(self, vocab: dict[str, int]):
        """
        Initialize tokenizer with a vocabulary mapping.

        Args:
            vocab: Dictionary mapping words to token IDs
                   e.g., {"[PAD]": 0, "[START]": 1, "zero": 4, "plus": 32, ...}
        """
        # Word to ID mapping (for encoding)
        self.vocab = vocab
        # ID to word mapping (for decoding) - reverse the vocab dict
        self.id_to_word = {v: k for k, v in vocab.items()}

        # Define special tokens - these control sequence boundaries
        self.pad_token = "[PAD]"      # Fills unused positions in fixed-length sequences
        self.start_token = "[START]"  # Marks the beginning of a sequence
        self.end_token = "[END]"      # Marks the end of a sequence
        self.unk_token = "[UNK]"      # Represents unknown/out-of-vocabulary words

    @classmethod
    def from_file(cls, vocab_path: str | Path) -> "Tokenizer":
        """
        Load tokenizer from vocab.json file.

        Args:
            vocab_path: Path to JSON file containing the vocabulary

        Returns:
            Initialized Tokenizer instance
        """
        with open(vocab_path) as f:
            vocab = json.load(f)
        return cls(vocab)

    def normalize(self, text: str) -> str:
        """
        Normalize input text for tokenization.

        This ensures consistent input by:
        1. Converting to lowercase
        2. Replacing math symbols with word equivalents
        3. Splitting compound numbers (e.g., "twentyfive" → "twenty five")
        4. Normalizing whitespace

        Args:
            text: Raw input text (e.g., "2 + 3" or "Two Plus Three")

        Returns:
            Normalized text (e.g., "2 plus 3" or "two plus three")
        """
        # Step 1: Lowercase and strip whitespace
        text = text.lower().strip()

        # Step 2: Replace math symbols with words
        # This allows users to type "2+3" or "2 plus 3"
        text = text.replace("+", " plus ")
        text = text.replace("-", " minus ")
        text = text.replace("*", " times ")
        text = text.replace("=", " equals ")

        # Step 3: Split compound numbers (e.g., "twentyfive" -> "twenty five")
        # This handles cases where users don't put a space in compound numbers
        tens = [
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]
        ones = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        for ten in tens:
            for one in ones:
                text = text.replace(f"{ten}{one}", f"{ten} {one}")

        # Step 4: Normalize whitespace - collapse multiple spaces into one
        return " ".join(text.split())

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Convert text to token IDs.

        This is the main encoding function that transforms human-readable
        text into a sequence of integers the model can process.

        Args:
            text: Input text (e.g., "two plus three")
            add_special_tokens: If True, wrap sequence with [START] and [END]

        Returns:
            List of token IDs (e.g., [1, 6, 32, 7, 2])

        Example:
            >>> tokenizer.encode("two plus three")
            [1, 6, 32, 7, 2]  # [START] two plus three [END]
        """
        # First normalize the text
        text = self.normalize(text)

        # Start with [START] token if adding special tokens
        ids = [self.vocab[self.start_token]] if add_special_tokens else []

        # Convert each word to its token ID
        for word in text.split():
            # Use [UNK] token for words not in vocabulary
            ids.append(self.vocab.get(word, self.vocab[self.unk_token]))

        # Add [END] token if adding special tokens
        if add_special_tokens:
            ids.append(self.vocab[self.end_token])

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.

        This reverses the encoding process, turning model output
        back into human-readable text.

        Args:
            ids: List of token IDs (e.g., [1, 6, 32, 7, 2])
            skip_special_tokens: If True, remove [PAD], [START], [END], [UNK]

        Returns:
            Decoded text (e.g., "two plus three")

        Example:
            >>> tokenizer.decode([1, 6, 32, 7, 2])
            "two plus three"
        """
        # Define which tokens to skip
        special = {self.pad_token, self.start_token, self.end_token, self.unk_token}

        # Convert each ID to its word, optionally filtering special tokens
        words = [
            self.id_to_word.get(id, self.unk_token)
            for id in ids
            if not (
                skip_special_tokens
                and self.id_to_word.get(id, self.unk_token) in special
            )
        ]
        return " ".join(words)

    @property
    def pad_token_id(self) -> int:
        """Get the ID of the [PAD] token (used for padding sequences)."""
        return self.vocab[self.pad_token]

    @property
    def start_token_id(self) -> int:
        """Get the ID of the [START] token (marks sequence beginning)."""
        return self.vocab[self.start_token]

    @property
    def end_token_id(self) -> int:
        """Get the ID of the [END] token (marks sequence end)."""
        return self.vocab[self.end_token]

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)


def pad_sequence(ids: list[int], max_len: int, pad_token_id: int) -> list[int]:
    """
    Pad or truncate a sequence to a fixed length.

    Neural networks require fixed-size inputs. This function ensures
    all sequences have the same length by:
    - Truncating sequences longer than max_len
    - Padding shorter sequences with pad_token_id

    Args:
        ids: List of token IDs
        max_len: Target sequence length
        pad_token_id: ID to use for padding (typically 0)

    Returns:
        Sequence of exactly max_len tokens

    Example:
        >>> pad_sequence([1, 6, 32, 7, 2], max_len=8, pad_token_id=0)
        [1, 6, 32, 7, 2, 0, 0, 0]
    """
    if len(ids) > max_len:
        # Truncate: keep only the first max_len tokens
        return ids[:max_len]
    # Pad: add pad tokens to reach max_len
    return ids + [pad_token_id] * (max_len - len(ids))
