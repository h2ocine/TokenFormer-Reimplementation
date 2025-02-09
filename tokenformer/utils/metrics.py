import torch

def estimate_perplexity(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            outputs = model(x_batch)  
            outputs = outputs.view(-1, outputs.shape[-1])
            y_batch = y_batch.view(-1)

            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * y_batch.size(0)  
            total_tokens += y_batch.size(0)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()
