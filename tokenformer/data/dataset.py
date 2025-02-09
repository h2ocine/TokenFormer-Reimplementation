import torch
from torch.utils.data import Dataset, DataLoader, random_split

class PokemonDataset(Dataset):
    def __init__(self, file_path, max_seq_len):
        with open(file_path, 'r') as f:
            self.text = f.read().lower().split()

        vocab = sorted(set(self.text))
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        self.vocab_size = len(self.vocab)

        self.tokens = [self.vocab[word] for word in self.text]
        self.data = [self.tokens[i : i + max_seq_len] for i in range(len(self.tokens) - max_seq_len)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return torch.tensor(x), torch.tensor(y)

def get_dataloaders(file_path, max_seq_len, batch_size, val_ratio=0.1, test_ratio=0.1):
    dataset = PokemonDataset(file_path, max_seq_len)
    vocab_size = dataset.vocab_size

    # Déterminer la taille des ensembles
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Création des DataLoaders avec `shuffle=True` pour le train
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab_size
