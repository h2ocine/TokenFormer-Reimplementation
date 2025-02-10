import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

    def forward(self, x):
        """
        Forward pass for TokenEmbedding.
        """
        return self.embedding(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, config.hidden_dim))

    def forward(self, x):
        """
        Forward pass for PositionalEmbedding.
        """
        seq_len = x.size(1)
        return self.positional_embedding[:, :seq_len, :]
    




    