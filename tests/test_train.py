"""Tests for train.py - Run with: pytest tests/test_train.py -v"""

import pytest
import torch
import sys
sys.path.insert(0, 'src')

from tokenizer import Tokenizer, pad_sequence
from model import CalculatorLLM
from train import prepare_dataloader


class TestPrepareDataloader:
    """Test the prepare_dataloader function."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        vocab = {
            "[PAD]": 0, "[START]": 1, "[END]": 2, "[UNK]": 3,
            "zero": 4, "one": 5, "two": 6, "three": 7, "four": 8,
            "five": 9, "six": 10, "seven": 11, "eight": 12, "nine": 13,
            "ten": 14, "eleven": 15, "twelve": 16, "thirteen": 17,
            "fourteen": 18, "fifteen": 19, "sixteen": 20, "seventeen": 21,
            "eighteen": 22, "nineteen": 23,
            "twenty": 24, "thirty": 25, "forty": 26, "fifty": 27,
            "sixty": 28, "seventy": 29, "eighty": 30, "ninety": 31,
            "plus": 32, "minus": 33, "times": 34, "equals": 35,
        }
        return Tokenizer(vocab)

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        return [
            {"input": "two plus three", "output": "five", "full": "two plus three equals five"},
            {"input": "five minus two", "output": "three", "full": "five minus two equals three"},
            {"input": "three times two", "output": "six", "full": "three times two equals six"},
            {"input": "one plus one", "output": "two", "full": "one plus one equals two"},
        ]

    def test_returns_dataloader(self, tokenizer, sample_data):
        """Test that function returns a DataLoader."""
        loader = prepare_dataloader(sample_data, tokenizer, max_seq_len=16, batch_size=2)
        assert hasattr(loader, '__iter__')

    def test_batch_size(self, tokenizer, sample_data):
        """Test that batches have correct size."""
        loader = prepare_dataloader(sample_data, tokenizer, max_seq_len=16, batch_size=2)
        batch = next(iter(loader))
        assert batch[0].shape[0] == 2  # batch_size

    def test_sequence_length(self, tokenizer, sample_data):
        """Test that sequences are padded to max_seq_len."""
        max_seq_len = 16
        loader = prepare_dataloader(sample_data, tokenizer, max_seq_len=max_seq_len, batch_size=2)
        batch = next(iter(loader))
        assert batch[0].shape[1] == max_seq_len

    def test_data_is_tensor(self, tokenizer, sample_data):
        """Test that data is converted to tensor."""
        loader = prepare_dataloader(sample_data, tokenizer, max_seq_len=16, batch_size=2)
        batch = next(iter(loader))
        assert isinstance(batch[0], torch.Tensor)

    def test_data_contains_valid_token_ids(self, tokenizer, sample_data):
        """Test that all token IDs are valid."""
        loader = prepare_dataloader(sample_data, tokenizer, max_seq_len=16, batch_size=2)
        batch = next(iter(loader))
        assert batch[0].min() >= 0  # No negative IDs
        assert batch[0].max() < 36  # All within vocab


class TestTrainingLoop:
    """Test the training loop mechanics."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return CalculatorLLM(
            vocab_size=36,
            embed_dim=32,  # Smaller for faster tests
            num_heads=2,
            num_layers=1,
            ff_dim=64,
            max_seq_len=16,
        )

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        vocab = {
            "[PAD]": 0, "[START]": 1, "[END]": 2, "[UNK]": 3,
            "zero": 4, "one": 5, "two": 6, "three": 7, "four": 8,
            "five": 9, "six": 10, "seven": 11, "eight": 12, "nine": 13,
            "ten": 14, "eleven": 15, "twelve": 16, "thirteen": 17,
            "fourteen": 18, "fifteen": 19, "sixteen": 20, "seventeen": 21,
            "eighteen": 22, "nineteen": 23,
            "twenty": 24, "thirty": 25, "forty": 26, "fifty": 27,
            "sixty": 28, "seventy": 29, "eighty": 30, "ninety": 31,
            "plus": 32, "minus": 33, "times": 34, "equals": 35,
        }
        return Tokenizer(vocab)

    def test_loss_computes(self, model, tokenizer):
        """Test that loss can be computed."""
        x = torch.tensor([[1, 6, 32, 7, 35, 9, 2, 0, 0]])  # padded sequence
        output = model(x)

        # Compute loss (next token prediction)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Shift: predict next token
        logits = output[:, :-1, :].contiguous().view(-1, 36)
        targets = x[:, 1:].contiguous().view(-1)
        loss = criterion(logits, targets)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_loss_decreases_with_training(self, model, tokenizer):
        """Test that loss decreases over a few steps."""
        x = torch.tensor([[1, 6, 32, 7, 35, 9, 2, 0, 0]])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        initial_loss = None
        for i in range(10):
            optimizer.zero_grad()
            output = model(x)
            logits = output[:, :-1, :].contiguous().view(-1, 36)
            targets = x[:, 1:].contiguous().view(-1)
            loss = criterion(logits, targets)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        # Loss should decrease (model is learning)
        assert final_loss < initial_loss

    def test_gradients_flow(self, model, tokenizer):
        """Test that gradients flow to all parameters."""
        x = torch.tensor([[1, 6, 32, 7, 35, 9, 2]])
        output = model(x)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
