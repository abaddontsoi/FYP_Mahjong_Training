import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PongMLP import PongMLP, PongDataset

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train():
    PONG_ROOT = 'pong_logs'
    data = None
    with open(f'{PONG_ROOT}/pong.json', 'r') as f:
        data = json.load(f)
    X_pong = np.array(data, dtype=np.float32)
    y_pong = np.ones((X_pong.shape[0],), dtype=np.float32)

    pass_data = None
    with open(f'{PONG_ROOT}/pass_pong.json', 'r') as f:
        pass_data = json.load(f)
    X_pass_pong = np.array(pass_data, dtype=np.float32)
    y_pass_pong = np.zeros((X_pass_pong.shape[0],), dtype=np.float32)

    X = np.vstack((X_pong, X_pass_pong))
    y = np.hstack((y_pong, y_pass_pong))
    print(f"Total: {len(X)} pong cases")
    print(f"\tPong taken: {len(X_pong)} ({100*len(X_pong)/len(X):.1f}%)")
    print(f"\tPong passed: {len(X_pass_pong)} ({100*len(X_pass_pong)/len(X):.1f}%)")
    print(f"Features per state: {X.shape[1]}")

    # Split data set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_ds = PongDataset(X_train, y_train)
    val_ds = PongDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # Create model
    input_dim = X.shape[1]
    model = PongMLP(input_dim)
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

    print("\n=== TRAINING PONG MODEL ===")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_acc += (preds == y_batch).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        
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
        'feature_info': 'pong_features'
    }, 'pong_model.pth')
    print("Model saved as 'pong_model.pth'")


if __name__ == "__main__":
    train()