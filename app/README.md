---
title: Calculator LLM
emoji: 🧮
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
language:
  - en
tags:
  - transformer
  - educational
  - math
  - from-scratch
datasets:
  - custom
pipeline_tag: text-generation
---

# Calculator LLM

A tiny decoder-only transformer (105K parameters) that performs arithmetic in English.

## Model Description

Calculator LLM is an educational GPT-style model built from scratch. It learns to solve math problems expressed in English words, outputting answers in English.

- **Developed by:** [Siddhartha Lahiri](https://sid.sh)
- **Model type:** Decoder-only Transformer
- **Language:** English
- **License:** MIT
- **Tutorial:** [Build Your First LLM](https://sid.sh/learn/build-your-first-llm)
- **Repository:** [GitHub](https://github.com/slahiri/small_calculator_model)

## Uses

### Direct Use

Text generation for simple arithmetic:

```python
Input:  "two plus three"      →  Output: "five"
Input:  "seven times eight"   →  Output: "fifty six"
Input:  "ninety minus forty"  →  Output: "fifty"
```

### Intended Use

- Educational demonstration of transformer architecture
- Learning how LLMs work from scratch
- Experimentation with small-scale language models

### Limitations

- Limited to numbers 0-99
- Only supports: addition, subtraction, multiplication
- English words only (not digits)
- ~99% accuracy (not 100%)

## Training

### Training Data

~97,000 examples generated programmatically covering all valid combinations:
- Addition: a + b where result ≤ 99
- Subtraction: a - b where result ≥ 0
- Multiplication: a × b where result ≤ 99

10% held out for testing (no overlap).

### Training Procedure

- **Epochs:** 100
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.01)
- **Scheduler:** Cosine annealing
- **Hardware:** CPU (~50 min) or GPU (~2 min)

## Evaluation

| Metric | Value |
|--------|-------|
| Test Accuracy | ~99% |
| Test Set Size | 1,078 |

## Technical Specifications

### Architecture

| | |
|---|---|
| Parameters | 104,740 |
| Layers | 2 |
| Attention Heads | 4 |
| Embedding Dim | 64 |
| FF Dim | 256 |
| Vocabulary | 36 tokens |
| Context Length | 16 tokens |
| Positional Encoding | Sinusoidal |

### Compute

- **Training:** ~50 min on CPU, ~2 min on GPU
- **Inference:** <10ms per query

## Citation

```bibtex
@misc{calculatorllm2024,
  author = {Lahiri, Siddhartha},
  title = {Calculator LLM: A Tiny Transformer for English Math},
  year = {2024},
  url = {https://github.com/slahiri/small_calculator_model}
}
```
