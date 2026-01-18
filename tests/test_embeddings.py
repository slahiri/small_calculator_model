"""Tests for the embedding classes."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from embeddings import TokenEmbedding, PositionalEncoding, InputEmbedding
from tokenizer import Tokenizer


def test_token_embedding_shape():
    """Test TokenEmbedding output shape."""
    vocab_size = 36
    embed_dim = 64
    batch_size = 2
    seq_len = 5

    token_emb = TokenEmbedding(vocab_size, embed_dim)
    token_ids = torch.tensor([[5, 31, 6, 0, 0], [10, 32, 7, 2, 0]])
    embeddings = token_emb(token_ids)

    assert embeddings.shape == (batch_size, seq_len, embed_dim), \
        f"Expected shape ({batch_size}, {seq_len}, {embed_dim}), got {embeddings.shape}"

    print(f"✓ TokenEmbedding output shape: {embeddings.shape}")


def test_same_token_same_embedding():
    """Test that the same token always gets the same embedding."""
    token_emb = TokenEmbedding(vocab_size=36, embed_dim=64)

    # Token 5 in different positions and batches
    ids1 = torch.tensor([[5]])
    ids2 = torch.tensor([[5, 31, 6]])

    emb1 = token_emb(ids1)[0, 0, :]
    emb2 = token_emb(ids2)[0, 0, :]

    assert torch.equal(emb1, emb2), "Same token should have same embedding"
    print("✓ Same token -> same embedding")


def test_positional_encoding_shape():
    """Test PositionalEncoding output shape."""
    max_seq_len = 32
    embed_dim = 64
    batch_size = 2
    seq_len = 5

    pos_enc = PositionalEncoding(max_seq_len, embed_dim)
    embeddings = torch.randn(batch_size, seq_len, embed_dim)
    pos_embeddings = pos_enc(embeddings)

    assert pos_embeddings.shape == (batch_size, seq_len, embed_dim), \
        f"Expected shape ({batch_size}, {seq_len}, {embed_dim}), got {pos_embeddings.shape}"

    print(f"✓ PositionalEncoding output shape: {pos_embeddings.shape}")


def test_different_positions_different_embeddings():
    """Test that same token at different positions has different embeddings."""
    vocab_size = 36
    embed_dim = 64
    max_seq_len = 32

    token_emb = TokenEmbedding(vocab_size, embed_dim)
    pos_enc = PositionalEncoding(max_seq_len, embed_dim)

    # Same token (5 = "two") repeated three times
    same_token = torch.tensor([[5, 5, 5]])
    emb_same = token_emb(same_token)
    emb_with_pos = pos_enc(emb_same)

    # Position 0 and position 1 should be different
    assert not torch.equal(emb_with_pos[0, 0, :], emb_with_pos[0, 1, :]), \
        "Same token at different positions should have different embeddings"

    # Position 1 and position 2 should be different
    assert not torch.equal(emb_with_pos[0, 1, :], emb_with_pos[0, 2, :]), \
        "Same token at different positions should have different embeddings"

    print("✓ Different positions -> different embeddings")


def test_input_embedding_shape():
    """Test InputEmbedding output shape."""
    vocab_size = 36
    embed_dim = 64
    max_seq_len = 32
    batch_size = 2
    seq_len = 5

    input_emb = InputEmbedding(vocab_size, embed_dim, max_seq_len)
    token_ids = torch.tensor([[5, 31, 6, 0, 0], [10, 32, 7, 2, 0]])
    final_embeddings = input_emb(token_ids)

    assert final_embeddings.shape == (batch_size, seq_len, embed_dim), \
        f"Expected shape ({batch_size}, {seq_len}, {embed_dim}), got {final_embeddings.shape}"

    print(f"✓ InputEmbedding output shape: {final_embeddings.shape}")


def test_parameter_count():
    """Test total parameter count in InputEmbedding."""
    vocab_size = 36
    embed_dim = 64
    max_seq_len = 32

    input_emb = InputEmbedding(vocab_size, embed_dim, max_seq_len)
    total_params = sum(p.numel() for p in input_emb.parameters())

    # Expected: vocab_size * embed_dim + max_seq_len * embed_dim
    # = 36 * 64 + 32 * 64 = 2,304 + 2,048 = 4,352
    expected_params = vocab_size * embed_dim + max_seq_len * embed_dim

    assert total_params == expected_params, \
        f"Expected {expected_params} parameters, got {total_params}"

    print(f"✓ Total parameters in InputEmbedding: {total_params:,}")


def test_full_pipeline():
    """Test complete flow from text to embeddings."""
    tokenizer = Tokenizer()
    input_emb = InputEmbedding(vocab_size=36, embed_dim=64, max_seq_len=32)

    # Test single input
    text = "two plus three"
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor([token_ids])  # Add batch dimension
    embeddings = input_emb(token_tensor)

    assert embeddings.shape == (1, 3, 64), f"Expected (1, 3, 64), got {embeddings.shape}"
    print(f"✓ Single input: '{text}' -> shape {embeddings.shape}")

    # Test batch of inputs
    texts = ["two plus three", "seven minus four", "six times eight"]

    # Encode all texts
    token_lists = [tokenizer.encode(t) for t in texts]

    # Pad to same length
    max_len = max(len(t) for t in token_lists)
    padded = [t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in token_lists]

    # Convert to tensor
    batch_tensor = torch.tensor(padded)
    assert batch_tensor.shape == (3, 3), f"Expected (3, 3), got {batch_tensor.shape}"

    # Get embeddings
    batch_embeddings = input_emb(batch_tensor)
    assert batch_embeddings.shape == (3, 3, 64), f"Expected (3, 3, 64), got {batch_embeddings.shape}"

    print(f"✓ Batch input: {len(texts)} texts -> shape {batch_embeddings.shape}")


def test_gradient_flow():
    """Test that gradients flow through embeddings."""
    input_emb = InputEmbedding(vocab_size=36, embed_dim=64, max_seq_len=32)
    token_ids = torch.tensor([[5, 31, 6]])

    embeddings = input_emb(token_ids)

    # Compute a simple loss and backprop
    loss = embeddings.sum()
    loss.backward()

    # Check that gradients exist
    for name, param in input_emb.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    print("✓ Gradients flow through all parameters")


def test_deterministic_output():
    """Test that same input gives same output (no randomness in forward pass)."""
    input_emb = InputEmbedding(vocab_size=36, embed_dim=64, max_seq_len=32)
    input_emb.eval()  # Set to eval mode

    token_ids = torch.tensor([[5, 31, 6]])

    emb1 = input_emb(token_ids)
    emb2 = input_emb(token_ids)

    assert torch.equal(emb1, emb2), "Same input should give same output"
    print("✓ Output is deterministic")


def test_different_batch_sizes():
    """Test that model works with different batch sizes."""
    input_emb = InputEmbedding(vocab_size=36, embed_dim=64, max_seq_len=32)

    for batch_size in [1, 2, 4, 8, 16]:
        token_ids = torch.randint(0, 36, (batch_size, 5))
        embeddings = input_emb(token_ids)
        assert embeddings.shape == (batch_size, 5, 64), \
            f"Failed for batch_size={batch_size}"

    print("✓ Works with various batch sizes")


def test_different_sequence_lengths():
    """Test that model works with different sequence lengths."""
    input_emb = InputEmbedding(vocab_size=36, embed_dim=64, max_seq_len=32)

    for seq_len in [1, 5, 10, 20, 32]:
        token_ids = torch.randint(0, 36, (2, seq_len))
        embeddings = input_emb(token_ids)
        assert embeddings.shape == (2, seq_len, 64), \
            f"Failed for seq_len={seq_len}"

    print("✓ Works with various sequence lengths (up to max_seq_len)")


def run_all_tests():
    """Run all embedding tests."""
    print("=" * 50)
    print("Running Embedding Tests")
    print("=" * 50)

    test_token_embedding_shape()
    test_same_token_same_embedding()
    test_positional_encoding_shape()
    test_different_positions_different_embeddings()
    test_input_embedding_shape()
    test_parameter_count()
    test_full_pipeline()
    test_gradient_flow()
    test_deterministic_output()
    test_different_batch_sizes()
    test_different_sequence_lengths()

    print("=" * 50)
    print("All embedding tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
