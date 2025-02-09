import torch
from torch.utils.data import Dataset, DataLoader

class PokemonDataset(Dataset):
    def __init__(self, file_path, max_seq_len, split='train', val_ratio=0.1):
        with open(file_path, 'r') as f:
            self.text = f.read().lower().split()

        vocab = sorted(set(self.text))
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        self.vocab_size = len(self.vocab)  # ✅ Correction ici

        self.tokens = [self.vocab[word] for word in self.text]
        self.data = [self.tokens[i : i + max_seq_len] for i in range(len(self.tokens) - max_seq_len)]

        # Séparation Train/Validation
        num_val = int(len(self.data) * val_ratio)
        if split == 'train':
            self.data = self.data[:-num_val]
        elif split == 'val':
            self.data = self.data[-num_val:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return torch.tensor(x), torch.tensor(y)

def get_dataloaders(file_path, max_seq_len, batch_size, val_ratio=0.1):
    train_dataset = PokemonDataset(file_path, max_seq_len, split='train', val_ratio=val_ratio)
    val_dataset = PokemonDataset(file_path, max_seq_len, split='val', val_ratio=val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.vocab_size  # ✅ Correction ici
