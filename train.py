import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from collections import Counter
import json
import pickle
import numpy as np
import os
import re
from path_utils import DATA_RAW, DATA_PROCESSED, MODELS, CHARTS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBED_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 3
LR = 1e-3

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        # We don't pack for simplicity here, just standard LSTM on padded
        output, (hn, cn) = self.lstm(embedded)
        # hn shape: (num_layers * num_directions, batch, hidden_size)
        # Concat the final forward and backward hidden states
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        out = self.dropout(hidden)
        # We return logits because BCEWithLogitsLoss is more stable
        return self.fc(out)

def build_dataset():
    logger.info("Loading IMDB Dataset...")
    dataset = load_dataset("imdb")
    
    logger.info("Building Vocabulary...")
    word_counts = Counter()
    for item in dataset['train']:
        word_counts.update(clean_text(item['text']))
    
    # 0 is PAD, 1 is UNK
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_counts.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
        
    def encode_and_pad(text):
        tokens = clean_text(text)
        seq = [vocab.get(word, 1) for word in tokens]
        if len(seq) < MAX_LEN:
            seq = seq + [0] * (MAX_LEN - len(seq))
        else:
            seq = seq[:MAX_LEN]
        return seq

    logger.info("Encoding Data...")
    X_train = np.array([encode_and_pad(item['text']) for item in dataset['train']])
    y_train = np.array([item['label'] for item in dataset['train']])
    
    X_test = np.array([encode_and_pad(item['text']) for item in dataset['test']])
    y_test = np.array([item['label'] for item in dataset['test']])
    
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(DATA_PROCESSED / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
        
    return X_train, y_train, X_test, y_test

def train_model():
    X_train, y_train, X_test, y_test = build_dataset()
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.float)), batch_size=BATCH_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    logger.info(f"Starting Training on {device}...")
    
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total
        
        # Eval
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        val_acc = correct / total
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS / "model.pth")
            
    # Save Results
    results = {"accuracy": best_acc, "epochs": EPOCHS, "vocab_size": VOCAB_SIZE}
    with open(MODELS / "results.json", "w") as f:
        json.dump(results, f)
    logger.info(f"✅ Training Complete. Best Accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    train_model()
