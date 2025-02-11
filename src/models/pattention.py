import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Pattention(nn.Module):
    """Pattention Layer with learnable parameter tokens."""

    def __init__(self, d1, d2, n, param_key_init_method, param_value_init_method, norm_activation_type):
        super().__init__()

        self.param_token_num = n
        self.param_key_dim = d1
        self.param_value_dim = d2
        self.norm_activation_type = norm_activation_type
        
        self.key_param_tokens = nn.Parameter(torch.rand((n, d1)))  
        self.value_param_tokens = nn.Parameter(torch.rand((n, d2)))  

        param_key_init_method(self.key_param_tokens)
        param_value_init_method(self.value_param_tokens)

    def expand_tokens(self, num_new_tokens):
        """Ajoute de nouveaux tokens paramétriques sans perdre les poids déjà appris."""
        device = self.key_param_tokens.device

        new_key_tokens = nn.Parameter(torch.zeros((num_new_tokens, self.param_key_dim), device=device))
        new_value_tokens = nn.Parameter(torch.zeros((num_new_tokens, self.param_value_dim), device=device))

        self.key_param_tokens = nn.Parameter(torch.cat([self.key_param_tokens, new_key_tokens], dim=0))
        self.value_param_tokens = nn.Parameter(torch.cat([self.value_param_tokens, new_value_tokens], dim=0))

        self.param_token_num += num_new_tokens
    
    def nonlinear_norm_func(self, inputs, normalize_type, dim=-1):
        if normalize_type == 'softmax': 
            nonlinear_outputs = torch.exp(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=1, dim=dim, keepdim=True) * inputs.shape[dim]
        elif normalize_type == 'gelu_l2_norm':
            nonlinear_outputs = F.gelu(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True) * math.sqrt(nonlinear_outputs.shape[dim])
        elif normalize_type == 'l2_norm_gelu':
            norm_outputs = inputs / torch.norm(inputs, p=2, dim=dim, keepdim=True) * math.sqrt(inputs.shape[dim])
            nonlinear_outputs = F.gelu(norm_outputs)
        return nonlinear_outputs

    def forward(self, inputs, dropout_p=0.0, attn_mask=None, scale=None):
        query = inputs
        key, value = self.key_param_tokens, self.value_param_tokens
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 if scale is None else scale 

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = self.nonlinear_norm_func(attn_weight, self.norm_activation_type, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = attn_weight @ value

        return output
