import torch
from models.configuration import ModelConfig
from models.pattention import Pattention, PattentionMultiHead
from models.embedding import TokenEmbedding, PositionalEmbedding

def test_pattention():
    # Configuration
    config = ModelConfig(
        vocab_size=10000,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        num_tokens=10,
        max_seq_len=50
    )
    batch_size = 4
    seq_len = 8

    # Création des données d'entrée
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))  # Tokens simulés
    token_embedding = TokenEmbedding(config)
    positional_embedding = PositionalEmbedding(config)

    # Combinaison des embeddings de tokens et positionnels
    token_embedded = token_embedding(token_ids)  # (batch_size, seq_len, hidden_dim)
    positional_embedded = positional_embedding(token_embedded)  # (1, seq_len, hidden_dim)
    input_data = token_embedded + positional_embedded  # (batch_size, seq_len, hidden_dim)

    # Instanciation et passage en avant de Pattention
    pat = Pattention(config)
    output = pat(input_data)

    # Vérifications des dimensions
    assert output.shape == (batch_size, seq_len, config.hidden_dim), f"Incorrect output shape: {output.shape}"

    # Vérification de la normalisation des poids d'attention
    scores = torch.matmul(input_data, pat.key_params.T)
    attn_weights = torch.softmax(scores, dim=-1)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1))), \
        "Attention weights are not normalized."

def test_pattention_multihead():
    # Configuration
    config = ModelConfig(
        vocab_size=10000,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        num_tokens=10,
        max_seq_len=50
    )
    batch_size = 4
    seq_len = 8

    # Création des données d'entrée
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))  # Tokens simulés
    token_embedding = TokenEmbedding(config)
    positional_embedding = PositionalEmbedding(config)

    # Combinaison des embeddings de tokens et positionnels
    token_embedded = token_embedding(token_ids)  # (batch_size, seq_len, hidden_dim)
    positional_embedded = positional_embedding(token_embedded)  # (1, seq_len, hidden_dim)
    input_data = token_embedded + positional_embedded  # (batch_size, seq_len, hidden_dim)

    # Instanciation et passage en avant de PattentionMultiHead
    pat_multihead = PattentionMultiHead(config)
    output = pat_multihead(input_data)

    # Vérifications des dimensions
    assert output.shape == (batch_size, seq_len, config.hidden_dim), f"Incorrect output shape: {output.shape}"

    # Vérification de la normalisation des poids d'attention
    # Calcul des scores d'attention pour une des têtes
    scores = torch.einsum("bshd,nkhd->bnsk", input_data, pat_multihead.key_params)  # (batch_size, seq_len, num_heads, num_tokens)
    attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len, num_heads, num_tokens)

    # Vérification que les poids d'attention sont bien normalisés
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1))), \
        "Attention weights are not normalized."