# Calculator LLM

A tiny decoder-only transformer (~105K parameters) that solves English math problems.

[![Train and Deploy](https://github.com/slahiri/small_calculator_model/actions/workflows/train-and-deploy.yml/badge.svg)](https://github.com/slahiri/small_calculator_model/actions/workflows/train-and-deploy.yml)
[![Demo](https://img.shields.io/badge/demo-huggingface-yellow)](https://huggingface.co/spaces/slahiri/small_calculator_model)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Overview

This is an educational implementation of a GPT-style transformer built from scratch. The model learns to perform arithmetic (addition, subtraction, multiplication) on numbers 0-99 expressed in English words.

```
Input:  "two plus three"     → Output: "five"
Input:  "seven times eight"  → Output: "fifty six"
Input:  "ninety minus forty" → Output: "fifty"
```

**Tutorial**: [sid.sh/learn/build-your-first-llm](https://sid.sh/learn/build-your-first-llm)

## Demo

[huggingface.co/spaces/slahiri/small_calculator_model](https://huggingface.co/spaces/slahiri/small_calculator_model)

## Installation

```bash
pip install torch
```

## Quick Start

**Train:**

```bash
cd src
python train.py --output ../output --epochs 100
```

Training takes ~50 minutes on CPU.

**Inference:**

```bash
python generate.py ../output "two plus three"
# five
```

## Model

| | |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 104,740 |
| Layers | 2 |
| Attention Heads | 4 |
| Embedding Dim | 64 |
| Feed-Forward Dim | 256 |
| Vocabulary | 36 tokens |
| Context Length | 16 tokens |

## Data

~97K training examples covering:
- **Addition**: a + b where 0 ≤ result ≤ 99
- **Subtraction**: a - b where result ≥ 0
- **Multiplication**: a × b where result ≤ 99

10% held out for testing. Expected accuracy: ~99%.

## Structure

```
src/
  model.py        # Transformer architecture
  tokenizer.py    # Text tokenization
  data.py         # Data generation
  train.py        # Training
  generate.py     # Inference
config/
  config.json     # Hyperparameters
  vocab.json      # Vocabulary
app/              # HF Space demo
```

## CI/CD

On push to `main`:
1. Train on GitHub Actions CPU (~50 min)
2. Validate accuracy ≥ 95%
3. Deploy to Hugging Face Spaces

Add `HF_TOKEN` secret with write access.

## Architecture

Standard transformer decoder:
- Token embeddings + sinusoidal positional encoding
- 2× transformer blocks (attention + FFN + layer norm + residuals)
- Causal masking for autoregressive generation

## License

MIT
