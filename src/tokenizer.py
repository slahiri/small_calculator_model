"""
Tokenizer for Calculator LLM

Converts English math expressions to token IDs and back.
"""

import json
import re
from pathlib import Path


class Tokenizer:
    """Tokenizer for English math expressions."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.id_to_word = {v: k for k, v in vocab.items()}
        self.pad_token = "[PAD]"
        self.start_token = "[START]"
        self.end_token = "[END]"
        self.unk_token = "[UNK]"

    @classmethod
    def from_file(cls, vocab_path: str | Path) -> "Tokenizer":
        """Load tokenizer from vocab.json file."""
        with open(vocab_path) as f:
            vocab = json.load(f)
        return cls(vocab)

    def normalize(self, text: str) -> str:
        """Normalize input text for tokenization."""
        text = text.lower().strip()

        # Replace symbols with words
        text = text.replace("+", " plus ")
        text = text.replace("-", " minus ")
        text = text.replace("*", " times ")
        text = text.replace("=", " equals ")

        # Split compound numbers (e.g., "twentyfive" -> "twenty five")
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

        # Normalize whitespace
        return " ".join(text.split())

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert text to token IDs."""
        text = self.normalize(text)
        ids = [self.vocab[self.start_token]] if add_special_tokens else []

        for word in text.split():
            ids.append(self.vocab.get(word, self.vocab[self.unk_token]))

        if add_special_tokens:
            ids.append(self.vocab[self.end_token])

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        special = {self.pad_token, self.start_token, self.end_token, self.unk_token}
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
        return self.vocab[self.pad_token]

    @property
    def start_token_id(self) -> int:
        return self.vocab[self.start_token]

    @property
    def end_token_id(self) -> int:
        return self.vocab[self.end_token]

    def __len__(self) -> int:
        return len(self.vocab)


def pad_sequence(ids: list[int], max_len: int, pad_token_id: int) -> list[int]:
    """Pad or truncate a sequence to max_len."""
    if len(ids) > max_len:
        return ids[:max_len]
    return ids + [pad_token_id] * (max_len - len(ids))
