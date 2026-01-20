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
---

# 🧮 Calculator LLM

A tiny transformer model (~105K parameters) that solves English math problems.

## Try It

Enter a math problem in English like:
- "two plus three"
- "seven times eight"
- "twenty minus five"

The model will output the answer in English!

## Examples

| Input | Output |
|-------|--------|
| two plus three | five |
| seven times eight | fifty six |
| twenty minus five | fifteen |
| nine times nine | eighty one |

## Built From Scratch

This model was built following the tutorial at [sid.sh/learn/build-your-first-llm](https://sid.sh/learn/build-your-first-llm)

Same architecture as GPT (attention, feed-forward, transformer blocks), just much smaller!

## Model Details

| Property | Value |
|----------|-------|
| Parameters | ~105K |
| Layers | 2 transformer blocks |
| Embedding | 64 dimensions |
| Attention Heads | 4 |
| Vocabulary | 36 tokens |
| Operations | plus, minus, times |
| Number Range | 0-99 |

## Architecture

This is a decoder-only transformer with:
- Token embeddings + sinusoidal positional encoding
- 2 transformer blocks (multi-head attention + feed-forward)
- Causal masking for autoregressive generation
- Layer normalization and residual connections

## Links

- 📚 [Tutorial: Build Your First LLM](https://sid.sh/learn/build-your-first-llm)
- 💻 [Source Code on GitHub](https://github.com/slahiri/small_calculator_model)
