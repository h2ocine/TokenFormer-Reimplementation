import torch
import torch.nn as nn
from models.self_attention import SelfAttention
from models.pattention import Pattention

class TokenformerLayer(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_attention_heads, max_seq_len, attention_dropout=0.1, hidden_dropout=0.1, token_num=10):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.attention = SelfAttention(hidden_size, num_attention_heads, attention_dropout, token_num)
        self.mlp = Pattention(hidden_size, hidden_size, token_num, torch.nn.init.xavier_uniform_, torch.nn.init.xavier_uniform_, "l2_norm_gelu")
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding(torch.arange(x.size(1), device=x.device).unsqueeze(0))
        attention_output = self.attention(x)
        mlp_output = self.mlp(attention_output)
        logits = self.lm_head(self.dropout(mlp_output))
        return logits
