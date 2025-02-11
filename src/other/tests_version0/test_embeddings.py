import torch
from models.embedding import TokenEmbedding, PositionalEmbedding
from models.configuration import ModelConfig

def test_token_embedding():
    config = ModelConfig(vocab_size=10000, hidden_dim=32, num_heads=4, num_layers=2, num_tokens=8, max_seq_len=50)
    embedding = TokenEmbedding(config)
    x = torch.randint(0, config.vocab_size, (2, 20))  # (2 , 20) == (batch_size, seq_len)
    output = embedding(x)
    assert output.shape == (2, 20, 32), "Incorrect TokenEmbedding output shape." # (batch_size, seq_len, hidden_dim)

def test_positional_embedding():
    config = ModelConfig(vocab_size=10000, hidden_dim=32, num_heads=4, num_layers=2, num_tokens=8, max_seq_len=50)
    positional = PositionalEmbedding(config)
    x = torch.randint(0, config.vocab_size, (2, 20))  # (batch_size, seq_len)
    output = positional(x)
    assert output.shape == (1, 20, 32), "Incorrect PositionalEmbedding output shape." # (1, seq_len, hidden_dim)
