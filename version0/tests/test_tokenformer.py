import torch
from models.configuration import ModelConfig
from models.tokenformer import TokenFormerLayer

# Fonction de test
def test_tokenformer():
    # Créer une configuration modèle
    config = ModelConfig(vocab_size=10655, hidden_dim=32, num_heads=4, num_layers=2, num_tokens=10, max_seq_len=50)

    # Créer une instance du modèle TokenFormerLayer
    model = TokenFormerLayer(config)

    # Tester avec un batch de tokens d'entrée
    token_ids = torch.randint(0, config.vocab_size, (16, 50))  # (batch_size=16, seq_len=50)

    # Passer les tokens dans le modèle
    logits = model(token_ids)

    # Assertion pour vérifier la forme correcte
    assert logits.shape == (16, 50, config.vocab_size), f"Erreur de forme: {logits.shape}"
