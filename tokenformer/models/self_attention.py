import torch
import torch.nn as nn
import math
from models.pattention import Pattention

class SelfAttention(nn.Module):
    """self attention for tokenformer"""
    def __init__(self, hidden_size, num_attention_heads, attention_dropout=0.1, token_num=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        self.query = Pattention(hidden_size, hidden_size, token_num, torch.nn.init.xavier_uniform_, torch.nn.init.xavier_uniform_, "l2_norm_gelu")
        self.key = Pattention(hidden_size, hidden_size, token_num, torch.nn.init.xavier_uniform_, torch.nn.init.xavier_uniform_, "l2_norm_gelu")
        self.value = Pattention(hidden_size, hidden_size, token_num, torch.nn.init.xavier_uniform_, torch.nn.init.xavier_uniform_, "l2_norm_gelu")
        self.out_proj = Pattention(hidden_size, hidden_size, token_num, torch.nn.init.xavier_uniform_, torch.nn.init.xavier_uniform_, "l2_norm_gelu")

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.norm_factor = math.sqrt(self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        query_layer = self.query(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_layer = self.key(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        value_layer = self.value(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)

        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / self.norm_factor
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.out_proj(context_layer)
