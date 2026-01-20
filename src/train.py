"""
Training Script for Calculator LLM

Trains the model and saves artifacts for deployment.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import CalculatorLLM
from tokenizer import Tokenizer, pad_sequence
from data import save_data
from generate import evaluate_model


def load_config(config_path: str | Path) -> dict:
    """Load model configuration from JSON."""
    with open(config_path) as f:
        return json.load(f)


def prepare_dataloader(
    data: list[dict],
    tokenizer: Tokenizer,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Convert data to PyTorch DataLoader."""
    sequences = [
        pad_sequence(
            tokenizer.encode(item["full"]),
            max_seq_len,
            tokenizer.pad_token_id,
        )
        for item in data
    ]
    tensor_data = torch.tensor(sequences)
    dataset = TensorDataset(tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(
    config_path: str = "config/config.json",
    vocab_path: str = "config/vocab.json",
    output_dir: str = "output",
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    min_accuracy: float = 0.95,
) -> float:
    """
    Train the Calculator LLM model.

    Returns:
        Test accuracy (0.0 to 1.0)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and tokenizer
    print("Loading configuration...")
    config = load_config(config_path)
    tokenizer = Tokenizer.from_file(vocab_path)

    # Generate data
    print("Generating training and test data...")
    train_data, test_data = save_data(output_dir)

    # Load the generated data
    with open(output_dir / "training_data.json") as f:
        train_data = json.load(f)
    with open(output_dir / "test_data.json") as f:
        test_data = json.load(f)

    print(f"  Training examples: {len(train_data):,}")
    print(f"  Test examples: {len(test_data):,}")

    # Prepare data loaders
    train_loader = prepare_dataloader(
        train_data, tokenizer, config["max_seq_len"], batch_size
    )

    # Create model
    print("Creating model...")
    model = CalculatorLLM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0,  # No dropout for this small model
    )
    print(f"  Parameters: {model.count_parameters():,}")

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            x = batch[0]
            inputs = x[:, :-1]
            targets = x[:, 1:]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(
                outputs.reshape(-1, config["vocab_size"]),
                targets.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    accuracy, errors = evaluate_model(model, tokenizer, test_data)
    print(f"  Test Accuracy: {accuracy:.1%}")

    if errors:
        print(f"\n  Sample errors (showing first 5):")
        for e in errors[:5]:
            print(f"    {e['input']} = {e['got']} (expected: {e['expected']})")

    # Check minimum accuracy
    if accuracy < min_accuracy:
        print(f"\n ERROR: Accuracy {accuracy:.1%} is below minimum {min_accuracy:.1%}")
        sys.exit(1)

    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"  Saved to {output_dir / 'model.pt'}")

    # Copy config files to output
    import shutil
    shutil.copy(config_path, output_dir / "config.json")
    shutil.copy(vocab_path, output_dir / "vocab.json")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Calculator LLM")
    parser.add_argument(
        "--config", default="config/config.json", help="Path to config.json"
    )
    parser.add_argument(
        "--vocab", default="config/vocab.json", help="Path to vocab.json"
    )
    parser.add_argument(
        "--output", default="output", help="Output directory for model artifacts"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.95,
        help="Minimum test accuracy to pass (exits with error if below)",
    )
    args = parser.parse_args()

    train(
        config_path=args.config,
        vocab_path=args.vocab,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_accuracy=args.min_accuracy,
    )


if __name__ == "__main__":
    main()
