# Calculator LLM

A tiny LLM that converts English math phrases to answers. Built from scratch to learn how language models work.

**Tutorial:** [Build Your First LLM](https://sid.sh/learn/build-your-first-llm)

**Demo:** [Hugging Face Space](https://huggingface.co/spaces/slahiri/small_calculator_model)

## What it does

Converts text like:
```
"two plus three" -> "five"
"seven minus four" -> "three"
"six times eight" -> "forty eight"
```

## Project Structure

```
calculator-llm/
├── src/
│   ├── __init__.py
│   ├── tokenizer.py      # Vocabulary + Tokenizer class
│   └── embeddings.py     # Embedding + PositionalEncoding + InputEmbedding
├── tests/
│   ├── __init__.py
│   ├── test_tokenizer.py
│   └── test_embeddings.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
# Run tokenizer tests
python tests/test_tokenizer.py

# Run embedding tests
python tests/test_embeddings.py
```

## Usage

### Tokenizer

```python
from src.tokenizer import Tokenizer

tokenizer = Tokenizer()

# Encode text to token IDs
tokenizer.encode("two plus three")  # [5, 31, 6]

# Decode token IDs to text
tokenizer.decode([8])  # "five"

# Convert numbers to/from words
tokenizer.num_to_words(42)  # "forty two"
tokenizer.words_to_num("forty two")  # 42
```

### Embeddings

```python
from src.embeddings import InputEmbedding
import torch

# Create embedding layer
input_emb = InputEmbedding(vocab_size=36, embed_dim=64, max_seq_len=32)

# Convert token IDs to embeddings
token_ids = torch.tensor([[5, 31, 6]])  # "two plus three"
embeddings = input_emb(token_ids)
print(embeddings.shape)  # torch.Size([1, 3, 64])
```

## Vocabulary

36 tokens total:
- Special: `[PAD]`, `[START]`, `[END]`
- Numbers 0-19: `zero`, `one`, ... `nineteen`
- Tens: `twenty`, `thirty`, ... `ninety`
- Operations: `plus`, `minus`, `times`, `divided`, `by`

## Model Configuration

- Vocabulary size: 36
- Embedding dimension: 64
- Max sequence length: 32
- Total embedding parameters: 4,352
