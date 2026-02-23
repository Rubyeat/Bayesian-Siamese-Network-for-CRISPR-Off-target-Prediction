import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ------------------------
# Reproducibility
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# ------------------------
# Dataset
# ------------------------
def one_hot_encode(seq, max_len=23):
    mapping = {"A":0, "C":1, "G":2, "T":3, "U":4}
    encoding = torch.zeros((max_len, len(mapping)))
    for i, base in enumerate(seq[:max_len]):
        idx = mapping.get(base.upper(), 4)
        encoding[i, idx] = 1.0
    return encoding

class gRNADataset(Dataset):
    def __init__(self, pairs, labels, max_len=23):
        self.pairs = pairs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq1, seq2 = self.pairs[idx]
        x1 = one_hot_encode(seq1, self.max_len)
        x2 = one_hot_encode(seq2, self.max_len)
        x = torch.cat([x1, x2], dim=1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, label

def load_pairs_and_labels(csv_file):
    pairs, labels = [], []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["on_seq"], row["off_seq"]))
            labels.append(float(row["label"]))
    return pairs, labels

# ------------------------
# CCLMOff Model
# ------------------------
class CCLMOff(nn.Module):
    def __init__(self, seq_len=23, input_dim=7, hidden_dim=64, dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(seq_len*input_dim*2, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_dim//2, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x).squeeze(-1)

# ------------------------
# Training, Evaluation, Uncertainty
# ------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(y.numpy())
    return r2_score(all_labels, all_preds)

def predict_with_uncertainty(model, loader, device, T=50):
    model.train()
    all_preds = []
    for t in range(T):
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                preds.extend(logits.cpu().numpy())
        all_preds.append(preds)
    all_preds = np.array(all_preds)
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    return mean_pred, std_pred

# ------------------------
# Main
# ------------------------
def main():
    csv_file = "Deep Learning/circle_seq_train_ratio_preserved.csv"
    pairs, labels = load_pairs_and_labels(csv_file)
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pairs, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = gRNADataset(train_pairs, train_labels)
    test_dataset = gRNADataset(test_pairs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCLMOff().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        r2 = evaluate(model, test_loader, device)
        print(f"[CCLMOff] Epoch {epoch+1}: Loss={loss:.4f} R2={r2:.4f}")

    mean_pred, std_pred = predict_with_uncertainty(model, test_loader, device, T=50)
    for i in range(5):
        print(f"[CCLMOff] Pred: {mean_pred[i]:.4f} ± {std_pred[i]:.4f}, True: {test_labels[i]}")

    torch.save(model.state_dict(), "CCLMOff_model.pth")
    print("✅ CCLMOff model saved.")

if __name__ == "__main__":
    main()