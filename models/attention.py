import torch
import torch.nn as nn
import torch.nn.functional as F
from models.configuration import ModelConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.configuration import ModelConfig

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        """
        Classic Token-Token Attention with automatic Q, K, V projection.
        Args:
            config: ModelConfig object with model configurations.
        """
        super(Attention, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.scale = torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))

        # Projections linéaires pour Q, K, V
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        """
        Forward pass for classic attention.
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim).
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        # 1. Projeter les vecteurs Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)    # (batch_size, seq_len, hidden_dim)
        V = self.value(x)  # (batch_size, seq_len, hidden_dim)

        # 2. Calculer les scores d'attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, seq_len, seq_len)

        # 3. Appliquer softmax pour normaliser les scores
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # 4. Calculer la sortie pondérée
        output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_dim)

        return output



# class MultiHeadAttention(nn.Module):
#     def __init__(self, config: ModelConfig):
#         """
#         Multi-Head Token-Token Attention.
#         Args:
#             config: ModelConfig object with model configurations.
#         """
#         super(MultiHeadAttention, self).__init__()
#         assert config.hidden_dim % config.num_heads == 0, "hidden_dim must be divisible by num_heads."
#         self.num_heads = config.num_heads
#         self.head_dim = config.hidden_dim // config.num_heads

#         # Projections pour Q, K, V
#         self.query = nn.Linear(config.hidden_dim, config.hidden_dim)
#         self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
#         self.value = nn.Linear(config.hidden_dim, config.hidden_dim)

#         # Projection finale
#         self.projection = nn.Linear(config.hidden_dim, config.hidden_dim)

#     def forward(self, x):
#         """
#         Forward pass for multi-head attention.
#         Args:
#             x: Input tensor of shape (batch_size, seq_len, hidden_dim).
#         Returns:
#             Output tensor of shape (batch_size, seq_len, hidden_dim).
#         """
#         batch_size, seq_len, hidden_dim = x.shape

#         # Projections Q, K, V
#         Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
#         K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
#         V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(2, 0, 1, 3)

#         # Attention pour chaque tête
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
#         attn_weights = F.softmax(scores, dim=-1)
#         multi_head_output = torch.matmul(attn_weights, V)

#         # Concaténation et projection finale
#         multi_head_output = multi_head_output.permute(1, 2, 0, 3).contiguous()
#         multi_head_output = multi_head_output.view(batch_size, seq_len, hidden_dim)
#         return self.projection(multi_head_output)
