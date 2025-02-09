import torch
import torch.nn as nn
import torch.optim as optim
import time
from data.dataset import get_dataloaders
from models.tokenformer import TokenformerLayer
from utils.metrics import estimate_perplexity
import platform

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    """
    Détecte automatiquement le meilleur device disponible :
    - CUDA pour les GPUs NVIDIA (Windows/Linux)
    - MPS pour les Macs M1/M2
    - CPU par défaut
    """
    if torch.cuda.is_available():
        return torch.device("cuda")  # Windows/Linux NVIDIA
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device("mps")  # Mac M1/M2
    else:
        return torch.device("cpu")  # Fallback CPU

def train(
    file_path='data/pokemon.txt',
    use_metrics=True,
    hidden_dim=32,
    num_heads=4,
    max_seq_len=32,
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001,
    token_num=32,
    val_ratio=0.1
):
    device = get_device()
    print(f"Training on {device}")

    # Charger les données
    train_loader, val_loader, vocab_size = get_dataloaders(file_path, max_seq_len, batch_size, val_ratio)
    print(f"Vocab Size: {vocab_size}")

    # Initialiser le modèle
    model = TokenformerLayer(
        hidden_size=hidden_dim,
        vocab_size=vocab_size,
        num_attention_heads=num_heads,
        max_seq_len=max_seq_len,
        attention_dropout=0.1,  
        hidden_dropout=0.1,          
        token_num=token_num
    )

    model = model.to(device)
    print(f"Trainable Parameters: {count_trainable_params(model)}")
    print(f"Training Samples: {len(train_loader.dataset)} | Validation Samples: {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting Training...")
    total_training_time = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)

            outputs = outputs.view(-1, vocab_size)
            y_batch = y_batch.view(-1)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_time = time.time() - start_time
        total_training_time += epoch_time

        if use_metrics:
            val_loss, perplexity = estimate_perplexity(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss / len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.4f} | Time: {epoch_time:.2f}s")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss / len(train_loader):.4f} | Time: {epoch_time:.2f}s")

    print(f"Total Training Time: {total_training_time:.2f}s")
