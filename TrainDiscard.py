import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class DiscardDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DiscardMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 34)  # 34-dim
        )
    
    def forward(self, x):
        return self.net(x)  # Return all 34 logits

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            
            # Get true tile indices for entire batch
            true_tiles = torch.argmax(y_batch, dim=-1)  # Shape: (batch_size,)
            
            # Get top-5 indices for entire batch
            top5_indices = torch.topk(logits, 5, dim=-1).indices  # Shape: (batch_size, 5)
            
            matches = (top5_indices == true_tiles.unsqueeze(1)).any(dim=-1)  # Shape: (batch_size,)
            correct += matches.sum().item()
            total += X_batch.size(0)
    
    avg_loss = total_loss / total
    top5_accuracy = correct / total
    return avg_loss, top5_accuracy


def train():
    DISCARD_ROOT = 'discard_logs'
    data = None
    with open(f'{DISCARD_ROOT}/discard.json', 'r') as f:
        data = json.load(f)
    feature_start = 392
    feature_end = 426
    
    raw_data = np.array(data, dtype=np.float32)
    X_before = raw_data[:, :feature_start]
    X_after = raw_data[:, feature_end:]
    
    X = np.concatenate([X_before, X_after], axis=1)
    y = raw_data[:, feature_start:feature_end]

    print(f"Total: {len(X)} discard cases")
    print(f"\tDiscard taken: {len(X)} ({100*len(X)/len(X):.1f}%)")
    print(f"Features per state: {X.shape[1]}")

    # Split data set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_ds = DiscardDataset(X_train, y_train)
    val_ds = DiscardDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # Create model
    input_dim = X.shape[1]
    model = DiscardMLP(input_dim)
    print(f"Model input dim: {input_dim}")

    # Loss + optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # Logits + sigmoid + BCE
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train for 50 epochs
    EPOCHS = 50
    best_val_acc = 0.0
    best_model_state = None

    print("\n=== TRAINING DISCARD MODEL ===")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
            # Top-5 accuracy
            true_tiles = torch.argmax(y_batch, dim=-1)  # Shape: (batch_size,)
            top5_indices = torch.topk(logits, 5, dim=-1).indices  # Shape: (batch_size, 5)
            matches = (top5_indices == true_tiles.unsqueeze(1)).any(dim=-1)
            train_correct += matches.sum().item()
            train_total += X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total  # Now Top-5 accuracy
        
        # Validation phase  
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | ",
            f"Train: {train_loss:.4f} acc={train_acc:.3f} | ",
            f"Val: {val_loss:.4f} acc={val_acc:.3f} | ",
            f"Best: {best_val_acc:.3f}")

    # Load best model weights
    model.load_state_dict(best_model_state)
    print(f"\nBest validation accuracy: {best_val_acc:.3f}")

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'best_val_acc': best_val_acc,
        'feature_info': 'discard_features'
    }, 'discard_model.pth')
    print("Model saved as 'discard_model.pth'")


if __name__ == "__main__":
    train()