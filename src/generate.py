"""
Text Generation for Calculator LLM

Inference and evaluation utilities.
"""

import json
from pathlib import Path

import torch

from model import CalculatorLLM
from tokenizer import Tokenizer


def load_model(
    model_dir: str | Path,
    device: str = "cpu",
) -> tuple[CalculatorLLM, Tokenizer, dict]:
    """
    Load a trained Calculator LLM model.

    Returns:
        (model, tokenizer, config) tuple
    """
    model_dir = Path(model_dir)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    tokenizer = Tokenizer.from_file(model_dir / "vocab.json")

    model = CalculatorLLM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=config.get("dropout", 0.1),
    )

    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    return model, tokenizer, config


def generate(
    model: CalculatorLLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    device: str = "cpu",
) -> str:
    """
    Generate text from a prompt using greedy decoding.

    Args:
        model: The trained model
        tokenizer: The tokenizer
        prompt: Input prompt (e.g., "two plus three equals")
        max_new_tokens: Maximum tokens to generate
        device: Device to run inference on

    Returns:
        Generated text including the prompt
    """
    model.eval()

    # Encode prompt (without end token)
    tokens = tokenizer.encode(prompt, add_special_tokens=True)[:-1]
    input_ids = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token = logits[0, -1, :].argmax().item()

            if next_token == tokenizer.end_token_id:
                break

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]]).to(device)], dim=1
            )

    return tokenizer.decode(input_ids[0].tolist())


def solve(
    model: CalculatorLLM,
    tokenizer: Tokenizer,
    problem: str,
    device: str = "cpu",
) -> str:
    """
    Solve an English math problem.

    Args:
        model: The trained model
        tokenizer: The tokenizer
        problem: Math problem (e.g., "two plus three")
        device: Device to run inference on

    Returns:
        Just the answer (e.g., "five")
    """
    # Normalize and ensure it ends with "equals"
    problem = problem.lower().strip()
    if not problem.endswith("equals"):
        problem = problem + " equals"

    result = generate(model, tokenizer, problem, device=device)

    # Extract just the answer after "equals"
    if "equals" in result:
        return result.split("equals")[-1].strip()
    return result


def evaluate_model(
    model: CalculatorLLM,
    tokenizer: Tokenizer,
    test_data: list[dict],
    device: str = "cpu",
) -> tuple[float, list[dict]]:
    """
    Evaluate model on test data.

    Returns:
        (accuracy, list_of_errors) tuple
    """
    model.eval()
    correct = 0
    errors = []

    for item in test_data:
        prompt = item["input"] + " equals"
        result = generate(model, tokenizer, prompt, device=device)

        if "equals" in result:
            answer = result.split("equals")[-1].strip()
        else:
            answer = result

        expected = item["output"]

        if answer == expected:
            correct += 1
        else:
            errors.append({
                "input": item["input"],
                "expected": expected,
                "got": answer,
            })

    accuracy = correct / len(test_data)
    return accuracy, errors


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python generate.py <model_dir> <problem>")
        print("Example: python generate.py output 'two plus three'")
        sys.exit(1)

    model_dir = sys.argv[1]
    problem = " ".join(sys.argv[2:])

    model, tokenizer, _ = load_model(model_dir)
    answer = solve(model, tokenizer, problem)
    print(f"{problem} = {answer}")
