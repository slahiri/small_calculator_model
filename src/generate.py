"""
Text Generation for Calculator LLM

Inference and evaluation utilities.

Generation process:
1. Encode the prompt (e.g., "two plus three equals")
2. Feed through model to get logits for next token
3. Pick the highest-probability token (greedy decoding)
4. Append to sequence and repeat until [END] or max length

This file also includes evaluation functions to test model accuracy.
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

    This loads all components needed for inference:
    - Model weights (model.pt)
    - Config (config.json)
    - Tokenizer vocabulary (vocab.json)

    Args:
        model_dir: Directory containing model.pt, config.json, vocab.json
        device: Device to load model on ("cpu" or "cuda")

    Returns:
        (model, tokenizer, config) tuple
    """
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(model_dir / "vocab.json")

    # Create model with same architecture as training
    model = CalculatorLLM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=config.get("dropout", 0.1),
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout)

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

    Greedy decoding: Always pick the highest probability token.
    Simple but effective for deterministic tasks like math.

    The process:
    1. Tokenize prompt: "two plus three equals" -> [1, 6, 32, 7, 35]
    2. Feed through model to get logits for position 5
    3. Pick argmax of logits -> token ID for "five"
    4. Append and repeat until [END] or max_new_tokens

    Args:
        model: The trained model
        tokenizer: The tokenizer
        prompt: Input prompt (e.g., "two plus three equals")
        max_new_tokens: Maximum tokens to generate
        device: Device to run inference on

    Returns:
        Generated text including the prompt

    Example:
        >>> generate(model, tokenizer, "two plus three equals")
        "two plus three equals five"
    """
    model.eval()  # Ensure model is in eval mode

    # Encode prompt (without end token since we're continuing generation)
    # encode() adds [START] and [END], so we remove the [END]
    tokens = tokenizer.encode(prompt, add_special_tokens=True)[:-1]
    input_ids = torch.tensor([tokens]).to(device)

    # Generate tokens one at a time
    with torch.no_grad():  # Disable gradient computation for inference
        for _ in range(max_new_tokens):
            # Forward pass: get logits for all positions
            logits = model(input_ids)  # [1, seq_len, vocab_size]

            # Get logits for the last position (next token prediction)
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # Greedy decoding: pick the highest probability token
            next_token = next_token_logits.argmax().item()

            # Stop if we hit the [END] token
            if next_token == tokenizer.end_token_id:
                break

            # Append the new token and continue
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]]).to(device)], dim=1
            )

    # Decode back to text
    return tokenizer.decode(input_ids[0].tolist())


def solve(
    model: CalculatorLLM,
    tokenizer: Tokenizer,
    problem: str,
    device: str = "cpu",
) -> str:
    """
    Solve an English math problem.

    This is a convenience wrapper around generate() that:
    1. Normalizes the input
    2. Ensures it ends with "equals"
    3. Returns just the answer (not the full equation)

    Args:
        model: The trained model
        tokenizer: The tokenizer
        problem: Math problem (e.g., "two plus three" or "2 + 3")
        device: Device to run inference on

    Returns:
        Just the answer (e.g., "five")

    Example:
        >>> solve(model, tokenizer, "two plus three")
        "five"
        >>> solve(model, tokenizer, "7 * 8")
        "fifty six"
    """
    # Normalize and ensure it ends with "equals"
    problem = problem.lower().strip()
    if not problem.endswith("equals"):
        problem = problem + " equals"

    # Generate the full equation
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

    For each test example:
    1. Generate answer from the input
    2. Compare to expected output
    3. Track correct/incorrect

    Args:
        model: The trained model
        tokenizer: The tokenizer
        test_data: List of dicts with 'input' and 'output' keys
        device: Device to run inference on

    Returns:
        (accuracy, list_of_errors) tuple
        - accuracy: float from 0.0 to 1.0
        - errors: list of dicts with 'input', 'expected', 'got' keys

    Example:
        >>> accuracy, errors = evaluate_model(model, tokenizer, test_data)
        >>> print(f"Accuracy: {accuracy:.1%}")
        Accuracy: 98.5%
    """
    model.eval()
    correct = 0
    errors = []

    for item in test_data:
        # Build prompt: "two plus three equals"
        prompt = item["input"] + " equals"

        # Generate answer
        result = generate(model, tokenizer, prompt, device=device)

        # Extract answer from generated text
        if "equals" in result:
            answer = result.split("equals")[-1].strip()
        else:
            answer = result

        expected = item["output"]

        # Check if correct
        if answer == expected:
            correct += 1
        else:
            errors.append({
                "input": item["input"],
                "expected": expected,
                "got": answer,
            })

    # Calculate accuracy
    accuracy = correct / len(test_data)
    return accuracy, errors


if __name__ == "__main__":
    """
    Command-line interface for solving problems.

    Usage:
        python generate.py <model_dir> <problem>

    Example:
        python generate.py output "two plus three"
        python generate.py output "7 * 8"
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: python generate.py <model_dir> <problem>")
        print("Example: python generate.py output 'two plus three'")
        sys.exit(1)

    model_dir = sys.argv[1]
    problem = " ".join(sys.argv[2:])

    # Load model and solve
    model, tokenizer, _ = load_model(model_dir)
    answer = solve(model, tokenizer, problem)
    print(f"{problem} = {answer}")
