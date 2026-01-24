"""Tests for model.py - Run with: pytest tests/test_model.py -v"""

import pytest
import torch
import math
import sys
sys.path.insert(0, 'src')

from model import (
    PositionalEncoding,
    TokenEmbedding,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    CalculatorLLM,
    create_causal_mask,
)


class TestPositionalEncoding:
    """Test the PositionalEncoding class."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        pe = PositionalEncoding(embed_dim=64, max_seq_len=100)
        x = torch.randn(2, 10, 64)  # batch=2, seq=10, dim=64
        output = pe(x)
        assert output.shape == x.shape

    def test_adds_position_info(self):
        """Test that position information is added (output differs from input)."""
        pe = PositionalEncoding(embed_dim=64, max_seq_len=100, dropout=0.0)
        x = torch.zeros(1, 5, 64)
        output = pe(x)
        # Output should not be all zeros since PE adds position info
        assert not torch.allclose(output, x)

    def test_different_positions_different_encodings(self):
        """Test that different positions get different encodings."""
        pe = PositionalEncoding(embed_dim=64, max_seq_len=100, dropout=0.0)
        x = torch.zeros(1, 5, 64)
        output = pe(x)
        # Position 0 should differ from position 1
        assert not torch.allclose(output[0, 0], output[0, 1])


class TestTokenEmbedding:
    """Test the TokenEmbedding class."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        embed = TokenEmbedding(vocab_size=36, embed_dim=64, max_seq_len=16)
        x = torch.tensor([[1, 6, 32, 7, 2]])  # [START] two plus three [END]
        output = embed(x)
        assert output.shape == (1, 5, 64)  # batch=1, seq=5, dim=64

    def test_different_tokens_different_embeddings(self):
        """Test that different tokens get different embeddings."""
        embed = TokenEmbedding(vocab_size=36, embed_dim=64, max_seq_len=16)
        x = torch.tensor([[6, 7]])  # "two", "three"
        output = embed(x)
        # Token "two" (6) should differ from "three" (7)
        assert not torch.allclose(output[0, 0], output[0, 1])

    def test_same_token_same_embedding(self):
        """Test that same token at same position gets same embedding."""
        embed = TokenEmbedding(vocab_size=36, embed_dim=64, max_seq_len=16, dropout=0.0)
        embed.eval()  # Disable dropout for deterministic output
        # Same token at same position
        x1 = torch.tensor([[6]])  # "two" at position 0
        x2 = torch.tensor([[6]])  # "two" at position 0
        # Same token at same position should produce identical embedding
        assert torch.allclose(embed(x1), embed(x2))


class TestMultiHeadAttention:
    """Test the MultiHeadAttention class."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        output, weights = attn(x)
        assert output.shape == x.shape

    def test_attention_weights_shape(self):
        """Test attention weights have correct shape."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        output, weights = attn(x)
        # weights: [batch, num_heads, seq_len, seq_len]
        assert weights.shape == (2, 4, 10, 10)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along last dimension."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.0)
        attn.eval()  # Disable dropout for deterministic output
        x = torch.randn(2, 10, 64)
        output, weights = attn(x)
        # Each row should sum to 1 (softmax output)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_masked_attention(self):
        """Test that masking prevents attending to future tokens."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(1, 5, 64)
        mask = create_causal_mask(5)
        output, weights = attn(x, mask=mask)
        # Upper triangle of attention weights should be ~0
        for i in range(5):
            for j in range(i + 1, 5):
                assert weights[0, :, i, j].max() < 1e-5


class TestFeedForward:
    """Test the FeedForward class."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        ff = FeedForward(embed_dim=64, ff_dim=256)
        x = torch.randn(2, 10, 64)
        output = ff(x)
        assert output.shape == x.shape

    def test_nonlinearity_applied(self):
        """Test that the network applies nonlinearity."""
        ff = FeedForward(embed_dim=64, ff_dim=256)
        # Linear input
        x = torch.ones(1, 1, 64)
        output = ff(x)
        # Output should differ from input due to nonlinearity
        assert not torch.allclose(output, x)


class TestTransformerBlock:
    """Test the TransformerBlock class."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        block = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(2, 10, 64)
        output, attn_weights = block(x)  # Returns tuple (output, attn_weights)
        assert output.shape == x.shape

    def test_with_mask(self):
        """Test that block works with causal mask."""
        block = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=256)
        x = torch.randn(2, 10, 64)
        mask = create_causal_mask(10)
        output, attn_weights = block(x, mask=mask)  # Returns tuple (output, attn_weights)
        assert output.shape == x.shape


class TestCreateCausalMask:
    """Test the create_causal_mask function."""

    def test_mask_shape(self):
        """Test that mask has correct shape."""
        mask = create_causal_mask(5)
        assert mask.shape == (1, 1, 5, 5)

    def test_mask_is_lower_triangular(self):
        """Test that mask is lower triangular (1s below/on diagonal)."""
        mask = create_causal_mask(4)
        expected = torch.tensor([
            [1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 1.],
        ])
        assert torch.allclose(mask.squeeze(), expected)

    def test_first_token_sees_only_itself(self):
        """Test that first token can only see itself."""
        mask = create_causal_mask(5)
        assert mask[0, 0, 0, 0] == 1  # Can see itself
        assert mask[0, 0, 0, 1] == 0  # Cannot see position 1
        assert mask[0, 0, 0, 4] == 0  # Cannot see position 4


class TestCalculatorLLM:
    """Test the complete CalculatorLLM model."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return CalculatorLLM(
            vocab_size=36,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            max_seq_len=16,
        )

    def test_output_shape(self, model):
        """Test that output has correct shape [batch, seq, vocab_size]."""
        x = torch.tensor([[1, 6, 32, 7, 2]])  # [START] two plus three [END]
        output = model(x)
        assert output.shape == (1, 5, 36)  # batch=1, seq=5, vocab=36

    def test_output_is_logits(self, model):
        """Test that output contains logits (not probabilities)."""
        x = torch.tensor([[1, 6, 32, 7, 2]])
        output = model(x)
        # Logits can be negative or > 1
        # If softmax was applied, all would be in [0, 1]
        has_negative = (output < 0).any()
        has_greater_than_one = (output > 1).any()
        assert has_negative or has_greater_than_one

    def test_different_inputs_different_outputs(self, model):
        """Test that different inputs produce different outputs."""
        x1 = torch.tensor([[1, 6, 32, 7, 2]])   # two plus three
        x2 = torch.tensor([[1, 7, 33, 6, 2]])   # three minus two
        output1 = model(x1)
        output2 = model(x2)
        assert not torch.allclose(output1, output2)

    def test_count_parameters(self, model):
        """Test that model has reasonable parameter count."""
        params = model.count_parameters()
        assert params > 0
        assert params < 1_000_000  # Should be ~105K for our config

    def test_model_is_differentiable(self, model):
        """Test that gradients flow through the model."""
        x = torch.tensor([[1, 6, 32, 7, 2]])
        output = model(x)
        loss = output.sum()
        loss.backward()
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
