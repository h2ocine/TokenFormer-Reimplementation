import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.configuration import ModelConfig
from models.tokenformer import TokenFormerLayer

# Préparer les données
class PokemonDataset(Dataset):
    def __init__(self, file_path, vocab_size, max_seq_len):
        with open(file_path, 'r') as f:
            self.text = f.read().lower().split()  # Convertir le texte en liste de mots
        
        # Créer un vocabulaire
        self.vocab = {word: idx for idx, word in enumerate(set(self.text))}
        self.vocab_size = len(self.vocab)
        self.max_seq_len = max_seq_len

        # Convertir le texte en indices
        self.tokens = [self.vocab[word] for word in self.text]
        self.data = [
            self.tokens[i : i + max_seq_len] for i in range(len(self.tokens) - max_seq_len)
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx][:-1]  # Séquence d'entrée
        y = self.data[idx][1:]   # Séquence cible (shiftée)
        return torch.tensor(x), torch.tensor(y)

# Configurer le modèle
def train():
    # Configurations
    config = ModelConfig(
        vocab_size=10000,    # Valeur temporaire, sera mise à jour avec vocab_size réel
        hidden_dim=32,
        num_heads=4, 
        num_layers=2,
        num_tokens=10,
        max_seq_len=50
    )

    # Préparer les données
    dataset = PokemonDataset(file_path='training/pokemon.txt', vocab_size=config.vocab_size, max_seq_len=config.max_seq_len)
    config.vocab_size = dataset.vocab_size  # Mettre à jour vocab_size après avoir créé le vocabulaire
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Instancier le modèle
    model = TokenFormerLayer(config)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("debut de l'entrainement")
    num_epochs=10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            device = next(model.parameters()).device

            # Déplacer les données sur le même appareil que le modèle (GPU ou CPU)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(x_batch)  # Shape: (batch_size, seq_len, vocab_size)

            # Aplatir les logits pour la CrossEntropyLoss
            outputs = outputs.view(-1, config.vocab_size)  # Shape: (batch_size * seq_len, vocab_size)

            # Aplatir les labels
            y_batch = y_batch.view(-1)  # Shape: (batch_size * seq_len)

            # Calculer la perte
            loss = criterion(outputs, y_batch)

            # Backpropagation et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Afficher la perte moyenne par epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    # # Entraîner le modèle
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     model.train()
    #     total_loss = 0
    #     for x_batch, y_batch in dataloader:
    #         # Obtenir le device sur lequel se trouve le modèle
    #         device = next(model.parameters()).device

    #         # Déplacer les données sur le même appareil
    #         x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    #         # Forward pass
    #         outputs = model(x_batch)  # (batch_size, seq_len, hidden_dim)

    #         print('vocab size : ',config.vocab_size)
    #         print('output shape : ', outputs.shape)
    #         #outputs = outputs.view(-1, config.vocab_size)  # Ajuster pour CrossEntropyLoss

    #         print('output shape after view: ', outputs.shape)
    #         print('y batch shape before views : ',y_batch.shape)
    #         y_batch = y_batch.view(-1)  # Ajuster pour CrossEntropyLoss
    #         print('y batch shape : ',y_batch.shape)
    #         # Calculer la perte
    #         loss = criterion(outputs, y_batch) #outputs (N,C) ; y_batch (N,)

    #         # Backpropagation et optimisation
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()

    #     # Afficher la perte moyenne par epoch
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    train()
