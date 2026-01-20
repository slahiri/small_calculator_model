# 🧮 Calculator LLM

A tiny transformer model (~105K parameters) that solves English math problems, built from scratch.

[![Train and Deploy](https://github.com/slahiri/small_calculator_model/actions/workflows/train-and-deploy.yml/badge.svg)](https://github.com/slahiri/small_calculator_model/actions/workflows/train-and-deploy.yml)
[![Hugging Face Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/slahiri/small_calculator_model)

## Live Demo

Try it out: [huggingface.co/spaces/slahiri/small_calculator_model](https://huggingface.co/spaces/slahiri/small_calculator_model)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/slahiri/small_calculator_model
cd small_calculator_model

# Install dependencies
pip install -r requirements.txt

# Train the model
cd src
python train.py --output ../output

# Test inference
python generate.py ../output "two plus three"
# Output: two plus three = five
```

## Project Structure

```
small_calculator_model/
├── .github/workflows/
│   └── train-and-deploy.yml    # CI/CD: train on push, deploy to HF
├── src/
│   ├── model.py                # Transformer architecture
│   ├── tokenizer.py            # Text ↔ token ID conversion
│   ├── data.py                 # Training data generation
│   ├── train.py                # Training script
│   └── generate.py             # Inference utilities
├── config/
│   ├── config.json             # Model hyperparameters
│   └── vocab.json              # 36-token vocabulary
├── app/
│   ├── app.py                  # Gradio demo for HF Space
│   ├── requirements.txt        # HF Space dependencies
│   └── README.md               # HF Space metadata
├── notebooks/
│   └── full_calculator_llm.ipynb  # Tutorial notebook
└── requirements.txt            # Training dependencies
```

## Model Architecture

| Property | Value |
|----------|-------|
| Type | Decoder-only Transformer |
| Parameters | ~105K |
| Layers | 2 transformer blocks |
| Embedding Dim | 64 |
| Attention Heads | 4 |
| FF Dim | 256 |
| Vocabulary | 36 tokens |
| Max Sequence | 16 tokens |

## Training

The model trains on ~97K examples covering:
- **Addition**: `a + b` where `a + b ≤ 99`
- **Subtraction**: `a - b` where `a - b ≥ 0`
- **Multiplication**: `a × b` where `a × b ≤ 99`

Test accuracy: **~99%** on held-out test set (no overlap with training).

## CI/CD Pipeline

On push to `main`:
1. **Train**: Run training on GitHub Actions (CPU, ~50 mins)
2. **Validate**: Ensure test accuracy ≥ 95%
3. **Deploy**: Push model to Hugging Face Space

### Setup

Add `HF_TOKEN` to your repository secrets:
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a token with write access
3. Add to GitHub: Settings → Secrets → Actions → `HF_TOKEN`

## Tutorial

This model was built following: [sid.sh/learn/build-your-first-llm](https://sid.sh/learn/build-your-first-llm)

## License

MIT
