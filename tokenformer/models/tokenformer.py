import torch
import torch.nn as nn
from models.self_attention import SelfAttention
from models.pattention import Pattention

class TokenformerLayer(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_attention_heads, max_seq_len, attention_dropout=0.1, hidden_dropout=0.1, token_num=10):
        super().__init__()
        self.token_num=token_num
        self.max_seq_len = max_seq_len
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

    def expand_tokenformer(self, new_token_num):
        """Augmente token_num sans réinitialiser les poids"""
        if new_token_num > self.attention.query.param_token_num:
            num_new_tokens = new_token_num - self.attention.query.param_token_num
            self.attention.query.expand_tokens(num_new_tokens)
            self.attention.key.expand_tokens(num_new_tokens)
            self.attention.value.expand_tokens(num_new_tokens)
            self.attention.out_proj.expand_tokens(num_new_tokens)
            self.mlp.expand_tokens(num_new_tokens)

    def scale_token_num(self, new_token_num):
        """Ajoute de nouveaux tokens paramétriques en conservant les poids appris."""
        if new_token_num <= self.token_num:
            print("⚠️ Le nombre de tokens doit être supérieur à l'ancien.")
            return

        num_new_tokens = new_token_num - self.token_num
        self.token_num = new_token_num

        # Mise à jour des couches Pattention pour inclure de nouveaux tokens
        for module in self.modules():
            if isinstance(module, Pattention):
                module.expand_tokens(num_new_tokens)
        
        print(f"✅ TokenFormer scalé à {new_token_num} tokens.")





# def __init__(self, hidden_size, vocab_size, num_attention_heads, max_seq_len, attention_dropout=0.1, hidden_dropout=0.1, token_num=10):
#         super().__init__()
#         self.max_seq_len = max_seq_len
#         self.token_embedding = nn.Embedding(vocab_size, hidden_size)
#         self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
#         self.attention = Pattention(hidden_size, hidden_size, token_num, torch.nn.init.xavier_uniform_, torch.nn.init.xavier_uniform_, "l2_norm_gelu")
#         self.mlp = Pattention(hidden_size, hidden_size, token_num, torch.nn.init.xavier_uniform_, torch.nn.init.xavier_uniform_, "l2_norm_gelu")
#         self.lm_head = nn.Linear(hidden_size, vocab_size)
#         self.dropout = nn.Dropout(hidden_dropout)



