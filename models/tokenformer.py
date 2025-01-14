import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding import TokenEmbedding, PositionalEmbedding
from models.pattention import Pattention, PattentionMultiHead
from models.configuration import ModelConfig

class TokenFormerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TokenFormerLayer, self).__init__()

        # Embeddings des tokens et des positions
        self.token_embedding = TokenEmbedding(config)
        self.positional_embedding = PositionalEmbedding(config)

        # Pattention (Token-Parameter Attention)
        self.pattention = Pattention(config) 
        # Pour utiliser la version multihead : self.pattention = PattentionMultiHead(config)

        # Projection linéaire pour transformer la sortie de hidden_dim vers vocab_size
        self.fc_out = nn.Linear(config.hidden_dim, config.vocab_size)

        # Normalisation de couche
        self.layer_norm_1 = nn.LayerNorm(config.hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_dim)
        self.layer_norm_3 = nn.LayerNorm(config.hidden_dim)
        self.layer_norm_4 = nn.LayerNorm(config.hidden_dim)

    def forward(self, token_ids):
        # Embedding des tokens et ajout des embeddings positionnels
        token_embedded = self.token_embedding(token_ids)
        positional_embedded = self.positional_embedding(token_embedded)
        X = token_embedded + positional_embedded

        # Appliquer la normalisation de la première couche
        X_norm = self.layer_norm_1(X)

        # Appliquer Pattention pour obtenir Q, K, V
        Q = self.pattention(X_norm)  
        K = self.pattention(X_norm)  
        V = self.pattention(X_norm)  

        # Calcul de l'attention classique (Token-Token Attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.pattention.hidden_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        Xatt = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_dim)

        # Appliquer la normalisation de la deuxième couche
        Xatt_norm = self.layer_norm_2(Xatt)

        # Appliquer la Pattention sur Xatt pour ajuster la sortie
        Oatt = self.pattention(Xatt_norm)

        # Appliquer la normalisation de la troisième couche
        Oatt_norm = self.layer_norm_3(Oatt)

        # Appliquer la Pattention pour le Feed-Forward Network (FFN)
        Xffn = self.pattention(Oatt_norm)

        # Appliquer la normalisation de la quatrième couche
        Xffn_norm = self.layer_norm_4(Xffn)

        # Appliquer la projection linéaire pour obtenir les logits
        logits = self.fc_out(Xffn_norm)  # Shape: (batch_size, seq_len, vocab_size)

        return logits
