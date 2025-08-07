import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# Paramètres
DATA_PATH = r'E:\Gesture-Recognition-using-3D-CNN\MP_DatafinalTransmove'
actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
num_classes = len(actions)
sequence_length = 30
num_landmarks = 33 * 3 + 2 * 21 * 3  # Pose (33*3, only shoulders 11-12 non-zero) + hands (2*21*3) = 132
batch_size = 16
num_epochs = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Augmentation functions
def rotate_landmarks(landmarks, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad), 0], [np.sin(angle_rad), np.cos(angle_rad), 0], [0, 0, 1]])
    rotated = np.copy(landmarks)
    for i in range(0, len(landmarks), 3):
        rotated[i:i + 3] = np.dot(rotation_matrix, landmarks[i:i + 3])
    return rotated


def scale_landmarks(landmarks, scale_factor):
    return landmarks * scale_factor


def translate_landmarks(landmarks, tx, ty):
    translated = np.copy(landmarks)
    translated[0::3] += tx
    translated[1::3] += ty
    return translated


def time_warp(sequence, factor):
    from scipy.interpolate import interp1d
    t_orig = np.linspace(0, 1, len(sequence))
    t_new = np.linspace(0, 1, int(len(sequence) * factor))
    interp = interp1d(t_orig, sequence, axis=0, kind='linear', fill_value="extrapolate")
    return interp(t_new)


def crop_pad(sequence, factor):
    new_length = int(len(sequence) * factor)
    if factor < 1.0:
        return sequence[:new_length]
    else:
        pad = np.repeat(sequence[-1:], int(new_length - len(sequence)), axis=0)
        return np.concatenate([sequence, pad], axis=0)


def frame_dropout(sequence, drop_rate):
    keep_mask = np.random.choice([True, False], size=len(sequence), p=[1 - drop_rate, drop_rate])
    if keep_mask.sum() == 0:  # Ensure at least one frame is kept
        keep_mask[0] = True
    return sequence[keep_mask]


def adjust_sequence_length(keypoints_sequence, target_length=30):
    """Adjust sequence to target_length by subsampling or padding."""
    current_length = len(keypoints_sequence) if isinstance(keypoints_sequence, (list, np.ndarray)) else 0
    if current_length == target_length:
        return np.array(keypoints_sequence)

    if current_length > target_length:
        # Uniformly subsample to target_length
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return np.array(keypoints_sequence)[indices]
    else:
        # Pad with last frame or zeros
        pad_length = target_length - current_length
        if current_length > 0:
            pad = np.repeat([keypoints_sequence[-1]], pad_length, axis=0)
            return np.concatenate([keypoints_sequence, pad], axis=0)
        else:
            return np.zeros((target_length, 33 * 3 + 21 * 3 + 21 * 3))


# Dataset
class ASLDataset(Dataset):
    def __init__(self, data_path, actions, sequence_length=30, augment=True):
        self.data_path = data_path
        self.actions = actions
        self.sequence_length = sequence_length
        self.augment = augment
        self.data = []
        self.labels = []

        for action_idx, action in enumerate(actions):
            action_path = os.path.join(data_path, action)
            for video_idx in os.listdir(action_path):
                video_path = os.path.join(action_path, video_idx)
                if os.path.isdir(video_path):
                    frames = [np.load(os.path.join(video_path, f)) for f in sorted(os.listdir(video_path)) if
                              f.endswith('.npy')]
                    if len(frames) > 0:
                        if len(frames) != sequence_length or frames[0].shape[0] != num_landmarks:
                            print(
                                f"Warning: Sequence {video_path} has {len(frames)} frames (expected {sequence_length}) "
                                f"or frame shape {frames[0].shape if frames else None} (expected {num_landmarks})")
                            continue
                        self.data.append(frames)
                        self.labels.append(action_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = np.array(self.data[idx])
        label = self.labels[idx]

        # Ensure sequence has exactly sequence_length frames
        sequence = adjust_sequence_length(sequence, self.sequence_length)

        # Augmentations
        if self.augment:
            if np.random.rand() < 0.3:
                sequence = rotate_landmarks(sequence.flatten(), np.random.uniform(-7, 7)).reshape(sequence.shape)
            if np.random.rand() < 0.3:
                sequence = scale_landmarks(sequence, np.random.uniform(0.95, 1.05))
            if np.random.rand() < 0.3:
                sequence = translate_landmarks(sequence, np.random.uniform(-0.04, 0.04), np.random.uniform(-0.04, 0.04))
            if np.random.rand() < 0.2:
                sequence = time_warp(sequence, np.random.uniform(0.88, 1.12))
                if len(sequence) != self.sequence_length:
                    sequence = crop_pad(sequence, self.sequence_length / len(sequence))
            if np.random.rand() < 0.2:
                sequence = frame_dropout(sequence, 0.05)
                if len(sequence) != self.sequence_length:
                    sequence = crop_pad(sequence, self.sequence_length / len(sequence))

        # Ensure final sequence length
        if len(sequence) != self.sequence_length:
            sequence = adjust_sequence_length(sequence, self.sequence_length)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


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
    if epoch % 50 == 49 or epoch == num_epochs - 1:  # Plot every 50 epochs and at the end
        conf_mat = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=actions, yticklabels=actions)
        plt.xlabel('Prédits')
        plt.ylabel('Réels')
        plt.title(f'Matrice de confusion - Epoch {epoch + 1}')
        plt.tight_layout()
        plt.show()

    print(f'Validation Loss: {val_loss / len(val_loader):.4f}, Validation Acc: {val_acc:.4f}')

torch.save(model.state_dict(), 'asl_transformer_hands_shoulders.pth')