"""
Training Script for Calculator LLM

Trains the model and saves artifacts for deployment.

Training process:
1. Load configuration and tokenizer
2. Generate training and test data
3. Create model and optimizer
4. Train for N epochs with cross-entropy loss
5. Evaluate on test set
6. Save model and config files

The model learns via next-token prediction:
- Input: "two plus three equals"
- Target: "plus three equals five" (shifted by 1)
- Loss: How wrong was our prediction at each position?
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
    """
    Load model configuration from JSON.

    Config contains hyperparameters like:
    - vocab_size: 36
    - embed_dim: 64
    - num_heads: 4
    - num_layers: 2
    - ff_dim: 256
    - max_seq_len: 16
    """
    with open(config_path) as f:
        return json.load(f)


def prepare_dataloader(
    data: list[dict],
    tokenizer: Tokenizer,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Convert data to PyTorch DataLoader.

    This prepares our text data for training by:
    1. Tokenizing each example (text -> token IDs)
    2. Padding to fixed length (for batching)
    3. Converting to PyTorch tensors
    4. Wrapping in DataLoader (for batching and shuffling)

    Args:
        data: List of examples with 'full' key (e.g., "two plus three equals five")
        tokenizer: Tokenizer for encoding text
        max_seq_len: Maximum sequence length (pad/truncate to this)
        batch_size: Number of examples per batch
        shuffle: Whether to shuffle data each epoch

    Returns:
        DataLoader yielding batches of token ID tensors
    """
    # Tokenize and pad each example
    sequences = [
        pad_sequence(
            tokenizer.encode(item["full"]),  # "two plus three equals five" -> [1, 6, 32, 7, 35, 9, 2]
            max_seq_len,
            tokenizer.pad_token_id,
        )
        for item in data
    ]

    # Convert to tensor: shape [num_examples, max_seq_len]
    tensor_data = torch.tensor(sequences)

    # Wrap in TensorDataset and DataLoader
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

    This is the main training function that:
    1. Sets up the model and data
    2. Runs the training loop
    3. Evaluates on test data
    4. Saves the trained model

    Args:
        config_path: Path to model config JSON
        vocab_path: Path to vocabulary JSON
        output_dir: Where to save model artifacts
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        min_accuracy: Minimum test accuracy required (exits with error if below)

    Returns:
        Test accuracy (0.0 to 1.0)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Load config and tokenizer ===
    print("Loading configuration...")
    config = load_config(config_path)
    tokenizer = Tokenizer.from_file(vocab_path)

    # === Step 2: Generate training and test data ===
    print("Generating training and test data...")
    train_data, test_data = save_data(output_dir)

    # Load the generated data back from files
    with open(output_dir / "training_data.json") as f:
        train_data = json.load(f)
    with open(output_dir / "test_data.json") as f:
        test_data = json.load(f)

    print(f"  Training examples: {len(train_data):,}")
    print(f"  Test examples: {len(test_data):,}")

    # === Step 3: Prepare data loader ===
    train_loader = prepare_dataloader(
        train_data, tokenizer, config["max_seq_len"], batch_size
    )

    # === Step 4: Create model ===
    print("Creating model...")
    model = CalculatorLLM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0,  # No dropout for this small model (helps convergence)
    )
    print(f"  Parameters: {model.count_parameters():,}")

    # === Step 5: Set up optimizer and loss ===
    # AdamW: Adam with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Cosine annealing: gradually reduce learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Cross-entropy loss: measures how wrong our predictions are
    # ignore_index tells it to skip padding tokens when computing loss
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # === Step 6: Training loop ===
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()  # Enable training mode (affects dropout, etc.)
        total_loss = 0

        for batch in train_loader:
            # Get the batch of sequences
            x = batch[0]  # Shape: [batch_size, max_seq_len]

            # Split into input and target (shifted by 1)
            # Input: all tokens except the last
            # Target: all tokens except the first
            # This is next-token prediction: predict token[i+1] from token[0:i]
            inputs = x[:, :-1]   # [batch, seq_len-1]
            targets = x[:, 1:]   # [batch, seq_len-1]

            # Forward pass
            optimizer.zero_grad()  # Clear gradients from previous step
            outputs = model(inputs)  # [batch, seq_len-1, vocab_size]

            # Compute loss
            # Reshape for cross-entropy: [batch*seq_len, vocab_size] vs [batch*seq_len]
            loss = criterion(
                outputs.reshape(-1, config["vocab_size"]),
                targets.reshape(-1),
            )

            # Backward pass
            loss.backward()  # Compute gradients

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()
            total_loss += loss.item()

        # Update learning rate
        scheduler.step()

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # === Step 7: Evaluate on test data ===
    print("\nEvaluating on test data...")
    accuracy, errors = evaluate_model(model, tokenizer, test_data)
    print(f"  Test Accuracy: {accuracy:.1%}")

    # Show sample errors for debugging
    if errors:
        print(f"\n  Sample errors (showing first 5):")
        for e in errors[:5]:
            print(f"    {e['input']} = {e['got']} (expected: {e['expected']})")

    # Check minimum accuracy requirement
    if accuracy < min_accuracy:
        print(f"\n ERROR: Accuracy {accuracy:.1%} is below minimum {min_accuracy:.1%}")
        sys.exit(1)

    # === Step 8: Save model and config ===
    print("\nSaving model...")
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"  Saved to {output_dir / 'model.pt'}")

    # Copy config files to output for easy deployment
    import shutil
    shutil.copy(config_path, output_dir / "config.json")
    shutil.copy(vocab_path, output_dir / "vocab.json")

    return accuracy


def main():
    """Command-line interface for training."""
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
