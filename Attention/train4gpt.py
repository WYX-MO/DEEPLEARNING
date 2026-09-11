#train4gpt
import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))                 # 当前
import torch
from Attention.datasets.shakespeare import ShakespeareDataset
from Attention.models.GPT import GPT
from torch.utils.data import DataLoader

def train_model(model, epochs, train_loader,learning_rate, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        
        total_loss = 0.0
        for  b_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Hyperparameters
    max_seq_len = 32
    d_model = 192
    num_heads = 4
    d_ff = 768
    num_layers = 6
    batch_size = 64
    epochs = 10
    learning_rate = 1e-4

    # Prepare dataset and dataloader
    dataset = ShakespeareDataset("Attention/data/shakespeare.txt", max_seq_len)

    vocab_size = dataset.chars  # Adjust based on dataset
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers).to(device)
    print(f"Using device: {device}")
    print("start training;exp0")
    train_model(model, epochs, train_loader, learning_rate, device)