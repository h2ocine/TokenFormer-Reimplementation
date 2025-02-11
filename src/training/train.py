import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import csv
from tqdm import tqdm  # Barre de progression
from data.dataset import get_dataloaders, get_dataloaders_ver2, OpenWebTextDataset
from models.tokenformer import TokenformerLayer
from models.transformer import TransformerModel
from utils.metrics import estimate_perplexity
from torch.utils.data import Subset


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
        #return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Mac M1/M2
    else:
        return torch.device("cpu")  # Fallback CPU

def compute_flops(hidden_dim, num_layers, num_heads, seq_len):
    """Estimation approximative du coût computationnel (FLOPS)."""
    flops_per_layer = 4 * hidden_dim * seq_len * (seq_len + hidden_dim)
    total_flops = num_layers * num_heads * flops_per_layer
    return total_flops



def train(
    file_path='data/pokemon.txt',
    use_tokenformer=True,
    hidden_dim=32,
    num_heads=4,
    num_layers=12,
    max_seq_len=32,
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001,
    token_num=32,
    val_ratio=0.1,
    test_ratio=0.1,
    model_checkpoint_file_name="model_checkpoint.pth",
    model_results_file_name="training_results.csv"
):
    """Entraîne un TokenFormer ou un Transformer classique."""

    device = get_device()
    print(f"Training on {device}")

    # Chargement des données
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        file_path, max_seq_len, batch_size, val_ratio, test_ratio
    )
    print(f"Vocab Size: {vocab_size}")

    # Initialisation du modèle
    if use_tokenformer:
        model = TokenformerLayer(
            hidden_size=hidden_dim,
            vocab_size=vocab_size,
            num_attention_heads=num_heads,
            max_seq_len=max_seq_len,
            attention_dropout=0.1,  
            hidden_dropout=0.1,          
            token_num=token_num
        )
        model_type = "TokenFormer"
    else:
        model = TransformerModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )
        model_type = "Transformer"

    model = model.to(device)
    trainable_params = count_trainable_params(model)
    total_flops = compute_flops(hidden_dim, num_layers, num_heads, max_seq_len)

    print(f"Model Type: {model_type}")
    print(f"Trainable Parameters: {trainable_params:_}")
    print(f"Approximate Computational Cost (FLOPS): {total_flops:_}")
    print(f"Training Samples: {len(train_loader.dataset)}, Validation Samples: {len(val_loader.dataset)}, Test Samples: {len(test_loader.dataset)}")

    # Définition des composants d'entraînement
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Création du répertoire de sauvegarde
    checkpoint_dir = "Saved_Models_Checkpoints"
    result_dir = "Saved_Models_Results"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Chemins de sauvegarde des fichiers
    results_path = os.path.join(result_dir, model_results_file_name)
    checkpoint_path = os.path.join(checkpoint_dir, model_checkpoint_file_name)

    # Initialisation du fichier CSV pour sauvegarder les métriques
    with open(results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_perplexity", "test_loss", "test_perplexity", "epoch_time", "total_time", "flops"])

    print("Starting Training...")
    total_training_time = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        # Barre de progression avec tqdm
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)

        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)

            outputs = outputs.view(-1, vocab_size)
            y_batch = y_batch.view(-1)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Mise à jour de la barre de progression
            progress_bar.set_postfix(loss=loss.item())

        epoch_time = time.time() - start_time
        total_training_time += epoch_time

        # Évaluation sur validation
        val_loss, val_perplexity = estimate_perplexity(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.4f} | Time: {epoch_time:.2f}s")

        # Sauvegarde des métriques
        with open(results_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, epoch_loss / len(train_loader), val_loss, val_perplexity, None, None, epoch_time, total_training_time, total_flops])

        # Sauvegarde du modèle après chaque époque (écrase l'ancien)
        torch.save(model.state_dict(), checkpoint_path)

    # Évaluation finale sur le test set
    print("\nFinal evaluation on test set...")
    test_loss, test_perplexity = estimate_perplexity(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_perplexity:.4f}")
    print(f"Total Training Time: {total_training_time:.2f}s")

    # Ajout des résultats de test dans le fichier CSV
    with open(results_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["FINAL", None, None, None, test_loss, test_perplexity, None, total_training_time, total_flops])

    # Sauvegarde du modèle final
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Final model saved: {checkpoint_path}")



def train_with_scaling(
    file_path,
    initial_token_num,
    scaling_steps,
    new_tokens_per_step,
    hidden_dim,
    num_heads,
    max_seq_len,
    batch_size,
    num_epochs,
    learning_rate=0.001,
    val_ratio=0.1,
    test_ratio=0.1,
    model_base_name="tokenformer_scaled"
):
    """Entraîne un modèle TokenFormer en augmentant token_num progressivement tout en conservant les poids."""

    device = get_device()
    print(f"Training on {device}")

    # Chargement des données
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders_ver2(
        file_path, max_seq_len, batch_size, val_ratio, test_ratio
    )
    print(f"Vocab Size: {vocab_size}")

    # Création des répertoires
    checkpoint_dir = "Saved_Models_Checkpoints"
    result_dir = "Saved_Models_Results"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Initialisation du modèle avec `token_num = initial_token_num`
    model = TokenformerLayer(
        hidden_size=hidden_dim,
        vocab_size=vocab_size,
        num_attention_heads=num_heads,
        max_seq_len=max_seq_len,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        token_num=initial_token_num
    ).to(device)

    print(f"Initial Token Num: {initial_token_num}")
    print(f"Trainable Parameters: {count_trainable_params(model):_}")

    # Définition des composants d'entraînement
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_training_time = 0.0

    for step in range(scaling_steps + 1):  # +1 pour inclure l'entraînement initial
        step_results_path = os.path.join(result_dir, f"{model_base_name}_step_{step}.csv")

        # Initialisation du fichier CSV pour ce scaling step
        with open(step_results_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_perplexity", "test_loss", "test_perplexity", "epoch_time", "total_time"])

        if step > 0:
            # Scale token_num et préserve les poids appris
            new_token_num = model.token_num + new_tokens_per_step[step - 1]
            model.scale_token_num(new_token_num)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print(f"🔼 Scaling step {step}: token_num = {new_token_num}")

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            start_time = time.time()

            # Barre de progression
            progress_bar = tqdm(train_loader, desc=f"Step {step}, Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)

            for x_batch, y_batch in progress_bar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)

                outputs = outputs.view(-1, vocab_size)
                y_batch = y_batch.view(-1)

                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Mise à jour de tqdm
                progress_bar.set_postfix(loss=loss.item())

            epoch_time = time.time() - start_time
            total_training_time += epoch_time

            # Évaluation sur validation
            val_loss, val_perplexity = estimate_perplexity(model, val_loader, criterion, device)

            print(f"Step {step} | Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss / len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.4f} | Time: {epoch_time:.2f}s")

            # Sauvegarde des métriques pour ce scaling step
            with open(step_results_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch+1, epoch_loss / len(train_loader), val_loss, val_perplexity, None, None, epoch_time, total_training_time])

        # Évaluation finale après le scaling step
        print(f"\n🔍 Evaluation finale après Scaling Step {step}...")
        test_loss, test_perplexity = estimate_perplexity(model, test_loader, criterion, device)

        print(f"Final Evaluation for Step {step} | Test Loss: {test_loss:.4f} | Test Perplexity: {test_perplexity:.4f}")

        # Sauvegarde du modèle après chaque scaling step
        model_checkpoint_path = os.path.join(checkpoint_dir, f"{model_base_name}_step_{step}.pth")
        torch.save(model.state_dict(), model_checkpoint_path)
        print(f"✅ Modèle sauvegardé : {model_checkpoint_path}")

        # Ajout des résultats de test dans le fichier CSV
        with open(step_results_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["FINAL", None, None, None, test_loss, test_perplexity, None, total_training_time])

    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Final results saved in: {result_dir}")
















def train_with_scaling_ver2(
    initial_token_num,
    scaling_steps,
    new_tokens_per_step,
    hidden_dim,
    num_heads,
    max_seq_len,
    batch_size,
    num_epochs,
    learning_rate=0.001,
    val_ratio=0.1,
    test_ratio=0.1,
    model_base_name="tokenformer_scaled",
    results_file_name="scaling_results.csv",
    subset_size=1000
):
    device = get_device()
    print(f"Training on {device}")

    # Préparer les données
    dataset = OpenWebTextDataset(max_seq_len=max_seq_len)
    dataset = Subset(dataset, range(min(subset_size, len(dataset))))
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders_ver2(dataset, max_seq_len, batch_size, val_ratio, test_ratio)

    # Création des répertoires pour sauvegarde
    checkpoint_dir = "Saved_Models_Checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialisation du modèle
    model = TokenformerLayer(hidden_dim, vocab_size, num_heads, max_seq_len, token_num=initial_token_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_training_time = 0.0

    with open(results_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_perplexity", "test_loss", "test_perplexity", "epoch_time", "total_time"])

    for step in range(scaling_steps + 1):
        if step > 0:
            model.scale_token_num(model.token_num + new_tokens_per_step[step - 1])
            print(f"🔼 Scaling step {step}: token_num = {model.token_num}")

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            start_time = time.time()

            progress_bar = tqdm(train_loader, desc=f"Step {step}, Epoch {epoch+1}/{num_epochs}", leave=False)

            for x_batch, y_batch in progress_bar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch).view(-1, vocab_size)
                y_batch = y_batch.view(-1)

                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_time = time.time() - start_time
            total_training_time += epoch_time

            val_loss, val_perplexity = estimate_perplexity(model, val_loader, criterion, device)
            test_loss, test_perplexity = estimate_perplexity(model, test_loader, criterion, device)

            print(f"Step {step} | Epoch {epoch+1} | Train Loss: {epoch_loss / len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f} | Epoch Time: {epoch_time:.2f}s")

            with open(results_file_name, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch+1, epoch_loss / len(train_loader), val_loss, val_perplexity, test_loss, test_perplexity, epoch_time, total_training_time])

        torch.save(model.state_dict(), f"{checkpoint_dir}/{model_base_name}_step_{step}.pth")

    print(f"Final results saved in: {results_file_name}")
