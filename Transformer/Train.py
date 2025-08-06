import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import os
from Extractonmove import ASLDataset, rotate_landmarks, scale_landmarks, translate_landmarks, time_warp, crop_pad, \
    frame_dropout

# Paramètres
DATA_PATH = 'MP_DatafinalTransmove'
actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
num_classes = len(actions)
sequence_length = 30
num_landmarks = 33 * 3 + 2 * 21 * 3 + 20 * 3  # Corps (33*3) + mains (2*21*3) + visage (20*3)
batch_size = 16
num_epochs = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Modèle Transformer
class ASLTransformer(nn.Module):
    def __init__(self, num_landmarks, seq_length=30, dim=64, num_heads=4, num_layers=3, num_classes=num_classes):
        super().__init__()
        self.embedding = nn.Linear(num_landmarks, dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=256, dropout=0.4,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)


# Dataset et DataLoader
dataset = ASLDataset(DATA_PATH, actions, sequence_length=sequence_length, augment=True)
print(f"Dataset size: {len(dataset)}")  # Debug: verify dataset size
if len(dataset) == 0:
    raise ValueError("Dataset is empty. Check DATA_PATH and .npy files.")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Entraînement
model = ASLTransformer(num_landmarks=num_landmarks).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    model.train()
    train_loss, train_preds, train_labels = 0, [], []
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}')


    model.eval()
    val_loss, val_preds, val_labels = 0, [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    if epoch == 300:
        # Juste après avoir calculé val_acc :
        conf_mat = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=actions, yticklabels=actions)
        plt.xlabel('Prédits')
        plt.ylabel('Réels')
        # \vplt.title(f'Matrice de confusion - Epoch {epoch + 1}')
        plt.tight_layout()
        plt.show()

    print(f'Validation Loss: {val_loss / len(val_loader):.4f}, Validation Acc: {val_acc:.4f}')

torch.save(model.state_dict(), 'asl_transformerhandmove.pth')

