
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchbnn as bnn
import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, log_loss

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# -----------------------
# Data Utilities
# -----------------------
def one_hot_encode(seq, max_len=23):
    mapping = {"A":0, "C":1, "G":2, "T":3, "N":4}
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
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x1, x2, label

# -----------------------
# LSTM Encoder with Attention
# -----------------------
class LSTMEncoderWithAttention(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.attention_fc = nn.Linear(hidden_dim*2, 1)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention_fc(outputs), dim=1)
        context = torch.sum(attn_weights * outputs, dim=1)
        embedding = self.fc(context)
        return embedding

# -----------------------
# Siamese Network with BNN Classifier
# -----------------------
class SiameseBNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.encoder = LSTMEncoderWithAttention(input_dim, hidden_dim)

        self.classifier = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim*2, out_features=64),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=64, out_features=1)
        )

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)

        diff = torch.abs(e1 - e2)
        prod = e1 * e2

        combined = torch.cat([diff, prod], dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(-1)

# -----------------------
# KL Loss for BNN
# -----------------------
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

# -----------------------
# Training + Evaluation
# -----------------------
def train_epoch(model, dataloader, criterion, optimizer, device, kl_weight=0.01):
    model.train()
    total_loss = 0.0
    for x1, x2, y in dataloader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x1, x2)
        bce = criterion(logits, y)
        kl = kl_loss(model)
        loss = bce + kl_weight * kl
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def predict_with_uncertainty(model, dataloader, device, n_samples=50):
    model.eval()
    all_mean, all_uncertainty, all_labels = [], [], []

    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            preds = []
            for _ in range(n_samples):
                logits = model(x1, x2)
                probs = torch.sigmoid(logits)
                preds.append(probs.cpu().numpy())
            preds = np.array(preds)
            mean_pred = preds.mean(axis=0)
            uncertainty = preds.std(axis=0)
            all_mean.extend(mean_pred)
            all_uncertainty.extend(uncertainty)
            all_labels.extend(y.numpy())
    return np.array(all_mean), np.array(all_uncertainty), np.array(all_labels)

def compute_metrics(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    nll = log_loss(y_true, y_pred)
    return acc, f1, auroc, auprc, nll

# -----------------------
# CSV Loader
# -----------------------
def load_pairs_and_labels(csv_file):
    pairs, labels = [], []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            on_seq = row["on_seq"].replace("-", "").replace("_", "")
            off_seq = row["off_seq"].replace("-", "").replace("_", "")
            pairs.append((on_seq, off_seq))
            labels.append(float(row["label"]))
    return pairs, labels

# -----------------------
# Main
# -----------------------
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
    model = SiameseBNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        mean_pred, uncertainty, labels_true = predict_with_uncertainty(model, test_loader, device)
        acc, f1, auroc, auprc, nll = compute_metrics(labels_true, mean_pred)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | NLL: {nll:.4f}")

    # Save model
    save_path = "Deep Learning/siamese_bnn_offtarget_model.pth"
    torch.save(model.state_dict(), save_path)
    print("✅ Model saved at:", save_path)

if __name__ == "__main__":
    main()