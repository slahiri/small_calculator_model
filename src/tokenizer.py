"""
Tokenizer for the Calculator LLM.

Handles conversion between text (English math phrases) and token IDs.
Supports numbers 0-99 and basic arithmetic operations.
"""


class Tokenizer:
    """
    Tokenizer for the calculator LLM.

    Handles conversion between text and token IDs.

    Vocabulary:
        [PAD] = 0    - Padding for batch processing
        [START] = 1  - Start of output sequence
        [END] = 2    - End of output sequence
        zero-nineteen = 3-22  - Numbers 0-19
        twenty-ninety = 23-30 - Tens (20, 30, ..., 90)
        plus = 31, minus = 32, times = 33, divided = 34, by = 35
    """

    def __init__(self):
        """Initialize vocabulary mappings."""
        # Special tokens
        self._special_tokens = {
            "[PAD]": 0,
            "[START]": 1,
            "[END]": 2,
        }

        # Numbers 0-19
        self._ones = [
            "zero", "one", "two", "three", "four", "five",
            "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
        ]

        # Tens: 20, 30, 40, ..., 90
        self._tens = [
            "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"
        ]

        # Operations
        self._operations = ["plus", "minus", "times", "divided", "by"]

        # Build token to ID mapping
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

        # Add special tokens (IDs 0-2)
        for token, id_ in self._special_tokens.items():
            self._token_to_id[token] = id_
            self._id_to_token[id_] = token

        # Add ones (IDs 3-22)
        for i, word in enumerate(self._ones):
            id_ = 3 + i
            self._token_to_id[word] = id_
            self._id_to_token[id_] = word

        # Add tens (IDs 23-30)
        for i, word in enumerate(self._tens):
            id_ = 23 + i
            self._token_to_id[word] = id_
            self._id_to_token[id_] = word

        # Add operations (IDs 31-35)
        for i, word in enumerate(self._operations):
            id_ = 31 + i
            self._token_to_id[word] = id_
            self._id_to_token[id_] = word

        # Build word-to-number mappings for conversion
        self._word_to_num: dict[str, int] = {}
        for i, word in enumerate(self._ones):
            self._word_to_num[word] = i
        for i, word in enumerate(self._tens):
            self._word_to_num[word] = (i + 2) * 10  # 20, 30, 40, ...

    @property
    def vocab_size(self) -> int:
        """Return size of vocabulary."""
        return len(self._token_to_id)

    @property
    def pad_token_id(self) -> int:
        """Return ID of [PAD] token."""
        return self._special_tokens["[PAD]"]

    @property
    def start_token_id(self) -> int:
        """Return ID of [START] token."""
        return self._special_tokens["[START]"]

    @property
    def end_token_id(self) -> int:
        """Return ID of [END] token."""
        return self._special_tokens["[END]"]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input string (e.g., "two plus three")
            add_special_tokens: If True, add [START] at beginning and [END] at end

        Returns:
            List of token IDs

        Example:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.encode("two plus three")
            [5, 31, 6]
            >>> tokenizer.encode("two plus three", add_special_tokens=True)
            [1, 5, 31, 6, 2]
        """
        # Normalize and split text
        words = text.lower().strip().split()

        token_ids = []
        for word in words:
            if word in self._token_to_id:
                token_ids.append(self._token_to_id[word])
            else:
                raise ValueError(f"Unknown token: '{word}'")

        if add_special_tokens:
            token_ids = [self.start_token_id] + token_ids + [self.end_token_id]

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: If True, don't include [PAD], [START], [END] in output

        Returns:
            Decoded string

        Example:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.decode([5, 31, 6])
            'two plus three'
            >>> tokenizer.decode([1, 5, 31, 6, 2], skip_special_tokens=True)
            'two plus three'
        """
        special_ids = set(self._special_tokens.values())

        words = []
        for id_ in token_ids:
            if id_ not in self._id_to_token:
                raise ValueError(f"Unknown token ID: {id_}")

            if skip_special_tokens and id_ in special_ids:
                continue

            words.append(self._id_to_token[id_])

        return " ".join(words)

    def num_to_words(self, n: int) -> str:
        """
        Convert integer (0-99) to words.

        Args:
            n: Integer between 0 and 99

        Returns:
            Word representation

        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.num_to_words(5)
            'five'
            >>> tokenizer.num_to_words(42)
            'forty two'
        """
        if not isinstance(n, int) or n < 0 or n > 99:
            raise ValueError(f"Number must be an integer between 0 and 99, got {n}")

        # 0-19: single word
        if n < 20:
            return self._ones[n]

        # 20, 30, 40, ..., 90: single word
        tens_digit = n // 10
        ones_digit = n % 10

        tens_word = self._tens[tens_digit - 2]  # -2 because tens starts at 20

        if ones_digit == 0:
            return tens_word
        else:
            ones_word = self._ones[ones_digit]
            return f"{tens_word} {ones_word}"

    def words_to_num(self, words: str) -> int:
        """
        Convert words back to integer.

        Args:
            words: Word representation of number

        Returns:
            Integer value

        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.words_to_num("five")
            5
            >>> tokenizer.words_to_num("forty two")
            42
        """
        parts = words.lower().strip().split()

        if len(parts) == 1:
            # Single word: either 0-19 or a tens word (20, 30, etc.)
            word = parts[0]
            if word not in self._word_to_num:
                raise ValueError(f"Unknown number word: '{word}'")
            return self._word_to_num[word]

        elif len(parts) == 2:
            # Two words: tens + ones (e.g., "forty two")
            tens_word, ones_word = parts

            if tens_word not in self._word_to_num:
                raise ValueError(f"Unknown tens word: '{tens_word}'")
            if ones_word not in self._word_to_num:
                raise ValueError(f"Unknown ones word: '{ones_word}'")

            tens_value = self._word_to_num[tens_word]
            ones_value = self._word_to_num[ones_word]

            # Validate: tens_word should be 20+ and ones_word should be 1-9
            if tens_value < 20:
                raise ValueError(f"Invalid compound number: '{words}'")
            if ones_value >= 10:
                raise ValueError(f"Invalid compound number: '{words}'")

            return tens_value + ones_value

        else:
            raise ValueError(f"Invalid number format: '{words}'")

    def get_vocab(self) -> dict[str, int]:
        """Return the complete vocabulary mapping."""
        return self._token_to_id.copy()

    def get_all_tokens(self) -> list[str]:
        """Return all tokens in vocabulary order."""
        return [self._id_to_token[i] for i in range(self.vocab_size)]
