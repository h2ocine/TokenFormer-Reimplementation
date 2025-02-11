import torch
from models.attention import Attention
from models.configuration import ModelConfig

def test_attention():
    config = ModelConfig(vocab_size=10000, hidden_dim=16, num_heads=4, num_layers=2, num_tokens=10, max_seq_len=50)
    batch_size = 4
    seq_len = 8

    # Tenseur d'entrée
    x = torch.randn(batch_size, seq_len, config.hidden_dim)

    # Instanciation et passage en avant
    attention = Attention(config)
    output = attention(x)

    # Vérification des dimensions
    assert output.shape == (batch_size, seq_len, config.hidden_dim), f"Incorrect output shape: {output.shape}"