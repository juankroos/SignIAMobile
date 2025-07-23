import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import cv2

# ----------------------------
# 1. Préparation des Données
# ----------------------------
class SignLanguageDataset(Dataset):
    def __init__(self, video_paths, seq_length=30, missing_rate=0.4):
        self.seq_length = seq_length
        self.missing_rate = missing_rate
        
        # Extraction des landmarks avec MediaPipe
        self.mp_hands = mp.solutions.hands.Hands()
        self.data = []
        
        for path in video_paths:
            cap = cv2.VideoCapture(path)
            sequence = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Conversion et traitement
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in results.multi_hand_landmarks[0].landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    sequence.append(landmarks)
            
            # Découpage en séquences
            for i in range(0, len(sequence) - self.seq_length, self.seq_length):
                seq = sequence[i:i+self.seq_length]
                self.data.append(np.array(seq))
        
        cap.release()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        complete_seq = self.data[idx]
        
        # Créer version avec points manquants
        missing_seq = complete_seq.copy()
        mask = np.random.random(missing_seq.shape) < self.missing_rate
        missing_seq[mask] = 0
        
        return torch.FloatTensor(missing_seq), torch.FloatTensor(complete_seq)

# ----------------------------
# 2. Architecture du VAE
# ----------------------------
class SpatioTemporalVAE(nn.Module):
    def __init__(self, input_dim=63, latent_dim=32, hidden_dim=128):
        super().__init__()
        
        # Encodeur
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Décodeur
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden[-1]  # Dernière couche cachée
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_length):
        # Préparer l'entrée pour le décodeur
        z = self.decoder_input(z)
        z = z.unsqueeze(1).repeat(1, seq_length, 1)
        
        # Passage à travers le LSTM
        output, _ = self.decoder(z)
        
        # Reconstruction finale
        return self.fc_out(output)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x.size(1)), mu, logvar

# ----------------------------
# 3. Fonction de Perte
# ----------------------------
def vae_loss(recon_x, x, mu, logvar, beta=0.7):
    # Terme de reconstruction
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # Terme KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

# ----------------------------
# 4. Entraînement
# ----------------------------
def train_vae(model, dataloader, epochs=100, lr=0.0003):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for batch_idx, (missing, complete) in enumerate(dataloader):
            missing, complete = missing.to(device), complete.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(missing)
            
            loss = vae_loss(recon_batch, complete, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item()/len(missing):.6f}')
        
        avg_loss = total_loss / len(dataloader.dataset)
        print(f'====> Epoch {epoch+1} Average loss: {avg_loss:.6f}')
        
        # Validation intermédiaire
        if epoch % 10 == 0:
            validate(model, dataloader, device)
    
    return model

# ----------------------------
# 5. Validation
# ----------------------------
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_rmse = 0
    
    with torch.no_grad():
        for missing, complete in dataloader:
            missing, complete = missing.to(device), complete.to(device)
            recon, mu, logvar = model(missing)
            
            # Calcul RMSE
            rmse = torch.sqrt(nn.functional.mse_loss(recon, complete))
            total_rmse += rmse.item()
            
            # Calcul perte
            loss = vae_loss(recon, complete, mu, logvar)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_rmse = total_rmse / len(dataloader)
    print(f'Validation Loss: {avg_loss:.6f} | RMSE: {avg_rmse:.6f}')
    
    return avg_rmse

# ----------------------------
# 6. Utilisation en Temps Réel
# ----------------------------
class LandmarkReconstructor:
    def __init__(self, model_path, seq_length=30):
        self.model = torch.load(model_path)
        self.model.eval()
        self.seq_buffer = []
        self.seq_length = seq_length
        
        # Initialiser MediaPipe
        self.mp_hands = mp.solutions.hands.Hands()
    
    def process_frame(self, frame):
        # Conversion et traitement
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(frame_rgb)
        
        landmarks = np.zeros(63)  # Valeur par défaut
        
        if results.multi_hand_landmarks:
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks)
        
        # Gestion du buffer
        self.seq_buffer.append(landmarks)
        if len(self.seq_buffer) > self.seq_length:
            self.seq_buffer.pop(0)
        
        # Reconstruction si nécessaire
        if np.any(np.array(self.seq_buffer) == 0):
            return self.reconstruct()
        
        return np.array(self.seq_buffer)
    
    def reconstruct(self):
        # Préparer l'entrée
        input_seq = np.array(self.seq_buffer)
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)
        
        # Reconstruction
        with torch.no_grad():
            recon, _, _ = self.model(input_tensor)
        
        return recon.squeeze(0).numpy()

# ----------------------------
# Exécution Principale
# ----------------------------
if __name__ == "__main__":
    # 1. Préparer les données
    video_paths = ["signe1.mp4", "signe2.mp4", ...]  # Vos chemins de vidéos
    dataset = SignLanguageDataset(video_paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialiser le modèle
    model = SpatioTemporalVAE()
    
    # 3. Entraînement
    trained_model = train_vae(model, dataloader, epochs=100)
    
    # 4. Sauvegarde
    torch.save(trained_model, "sign_language_vae.pth")
    
    # 5. Exemple d'utilisation en temps réel
    reconstructor = LandmarkReconstructor("sign_language_vae.pth")
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Traitement et reconstruction
        reconstructed_sequence = reconstructor.process_frame(frame)
        
        # Utiliser les landmarks reconstruits pour la suite du pipeline...
        # current_landmarks = reconstructed_sequence[-1]  # Dernière frame
        
        # Affichage (optionnel)
        cv2.imshow('Sign Language Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()