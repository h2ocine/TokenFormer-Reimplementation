import torch
import torch.nn as nn
import torch.nn.functional as F
from models.configuration import ModelConfig

class Pattention(nn.Module):
    def __init__(self, config: ModelConfig):
        """
        Token-Parameter Attention Layer.
        Args:
            config: ModelConfig with model parameters.
            num_tokens: Number of parameter tokens (e.g., nq, nk, nv).
        """
        super(Pattention, self).__init__()
        self.hidden_dim = config.hidden_dim
        num_tokens = config.num_tokens
        self.num_tokens = num_tokens

        # Paramètres pour K_P et V_P
        self.key_params = nn.Parameter(torch.randn(num_tokens, self.hidden_dim))  # (num_tokens, hidden_dim)
        self.value_params = nn.Parameter(torch.randn(num_tokens, self.hidden_dim))  # (num_tokens, hidden_dim)

    def forward(self, x):
        """
        Forward pass for Pattention.
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim).
        Returns:
            output: Tensor of shape (batch_size, seq_len, hidden_dim).
        """
        scores = torch.matmul(x, self.key_params.T)  # (batch_size, seq_len, num_tokens)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, num_tokens)
        output = torch.matmul(attn_weights, self.value_params)  # (batch_size, seq_len, hidden_dim)

        return output

class PattentionMultiHead(nn.Module):
    def __init__(self, config: ModelConfig):
        """
        Multi-Head Token-Parameter Attention Layer.
        Args:
            config: ModelConfig with model parameters.
            num_tokens: Number of parameter tokens (e.g., nq, nk, nv).
            num_heads: Number of attention heads.
        """
        super(PattentionMultiHead, self).__init__()
        self.hidden_dim = config.hidden_dim
        num_tokens = config.num_tokens
        self.num_tokens = num_tokens
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads  # La dimension de chaque tête

        # Paramètres pour K_P et V_P, spécifiques à chaque tête
        self.key_params = nn.Parameter(torch.randn(self.num_heads, num_tokens, self.head_dim))  # (num_heads, num_tokens, head_dim)
        self.value_params = nn.Parameter(torch.randn(self.num_heads, num_tokens, self.head_dim))  # (num_heads, num_tokens, head_dim)

    def forward(self, x):
        """
        Forward pass for multi-head Pattention.
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim).
        Returns:
            output: Tensor of shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Calcul des scores et attention pour chaque tête
        scores = torch.einsum("bshd,nkhd->bnsk", x, self.key_params)  # (batch_size, seq_len, num_heads, num_tokens)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, num_heads, num_tokens)

        # Calcul des sorties de chaque tête
        output_heads = torch.einsum("bnsq,nqhd->bnhd", attn_weights, self.value_params)  # (batch_size, seq_len, num_heads, head_dim)

        # Concaténer les sorties des têtes d'attention
        output = output_heads.view(batch_size, seq_len, self.hidden_dim)  # (batch_size, seq_len, hidden_dim)

        return output