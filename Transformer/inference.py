import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import os
import time
import csv
from datetime import datetime
from collections import deque
#source = 'rtsp://192.168.100.18:8080/h264_pcm.sdp'
# Supprimer l'avertissement oneDNN de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialiser MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
shoulder_landmarks = mp.solutions.holistic.PoseLandmark

# Paramètres
sequence_length = 30
num_landmarks = 33 * 3 + 2 * 21 * 3  # Pose (33*3, only shoulders 11-12 non-zero) + hands (2*21*3) = 132
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r'E:\Gesture-Recognition-using-3D-CNN\VAE\asl_transformer_hands_shoulders.pth'
DATA_PATH = 'MP_DatafinalTransmove'
CSV_PATH = 'predictions.csv'
actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
num_classes = len(actions)


# Fonctions utilitaires
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    # Draw shoulders (landmarks 11 and 12)
    if results.pose_landmarks:
        custom_connections = [(shoulder_landmarks.LEFT_SHOULDER, shoulder_landmarks.RIGHT_SHOULDER)]
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            custom_connections,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    # Draw left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    # Draw right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    # Draw connections from shoulders to hand wrists
    if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
        h, w, _ = image.shape
        if results.left_hand_landmarks and results.pose_landmarks.landmark[
            shoulder_landmarks.LEFT_SHOULDER].visibility > 0.5:
            shoulder_left = results.pose_landmarks.landmark[shoulder_landmarks.LEFT_SHOULDER]
            wrist_left = results.left_hand_landmarks.landmark[0]
            cv2.line(image,
                     (int(shoulder_left.x * w), int(shoulder_left.y * h)),
                     (int(wrist_left.x * w), int(wrist_left.y * h)),
                     (80, 44, 121), 2)
        if results.right_hand_landmarks and results.pose_landmarks.landmark[
            shoulder_landmarks.RIGHT_SHOULDER].visibility > 0.5:
            shoulder_right = results.pose_landmarks.landmark[shoulder_landmarks.RIGHT_SHOULDER]
            wrist_right = results.right_hand_landmarks.landmark[0]
            cv2.line(image,
                     (int(shoulder_right.x * w), int(shoulder_right.y * h)),
                     (int(wrist_right.x * w), int(wrist_right.y * h)),
                     (80, 44, 121), 2)


def normalize_landmarks(landmarks, ref_point):
    landmarks = landmarks - ref_point
    scale = np.linalg.norm(landmarks[11] - landmarks[12]) if landmarks[11].sum() != 0 and landmarks[
        12].sum() != 0 else 1.0
    if scale > 0:
        landmarks /= scale
    return landmarks


def extract_keypoints(results):
    # Initialize arrays
    pose = np.zeros(33 * 3)  # Only store shoulders (11, 12), others zero
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    # Extract shoulders (landmarks 11 and 12)
    if results.pose_landmarks:
        pose_3d = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
        pose = np.zeros(33 * 3)
        pose[11 * 3:12 * 3] = pose_3d[11, :].flatten()  # Left shoulder
        pose[12 * 3:13 * 3] = pose_3d[12, :].flatten()  # Right shoulder

    # Extract hands
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()

    # Reference point for normalization (pose landmark 0 or average of shoulders)
    ref_point = np.array([results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y,
                          results.pose_landmarks.landmark[0].z]) if results.pose_landmarks else np.zeros(3)

    # Normalize
    pose_3d = pose.reshape(-1, 3)
    lh_3d = lh.reshape(-1, 3)
    rh_3d = rh.reshape(-1, 3)

    pose_3d = normalize_landmarks(pose_3d, ref_point)
    lh_3d = normalize_landmarks(lh_3d, ref_point)
    rh_3d = normalize_landmarks(rh_3d, ref_point)

    return np.concatenate([pose_3d.flatten(), lh_3d.flatten(), rh_3d.flatten()])  # 132 dimensions


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


# Fonction pour enregistrer les prédictions dans un CSV
def save_prediction_to_csv(timestamp, gesture, confidence):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Gesture', 'Confidence'])
        writer.writerow([timestamp, gesture, f'{confidence:.2%}'])


# Inférence en temps réel avec détection basée sur la présence des mains
def main():
    # Charger le modèle
    model = ASLTransformer(num_landmarks=num_landmarks, seq_length=sequence_length, num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return
    model.eval()

    # Initialiser la

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la webcam")
        return

    # Buffer pour stocker la séquence
    sequence_buffer = deque(maxlen=sequence_length)
    gesture_active = False
    last_prediction_time = time.time()
    no_hand_frames = 0
    NO_HAND_THRESHOLD = 3  # Nombre de frames sans mains pour considérer le geste terminé

    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Erreur : Impossible de lire la frame")
                break
            frame = cv2.resize(frame,(500,500))
            # Extraire les landmarks
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)

            # Détecter la présence des mains
            hands_detected = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

            # Logique de détection du geste
            status_text = "En attente d’un geste..."
            if hands_detected and not gesture_active:
                # Début d’un nouveau geste
                gesture_active = True
                sequence_buffer.clear()  # Réinitialiser le buffer
                status_text = "Geste en cours..."
            elif hands_detected and gesture_active:
                # Continuer à capturer le geste
                sequence_buffer.append(keypoints)
                status_text = "Geste en cours..."
                no_hand_frames = 0
            elif not hands_detected and gesture_active:
                # Compter les frames sans mains
                no_hand_frames += 1
                if no_hand_frames >= NO_HAND_THRESHOLD:
                    # Geste terminé
                    if len(sequence_buffer) >= sequence_length // 2:  # Exiger au moins la moitié des frames
                        # Compléter la séquence si nécessaire
                        while len(sequence_buffer) < sequence_length:
                            sequence_buffer.append(sequence_buffer[-1] if sequence_buffer else np.zeros(num_landmarks))

                        # Préparer la séquence pour la prédiction
                        sequence = np.array(sequence_buffer)
                        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(
                            device)  # [1, 30, 132]

                        # Prédiction
                        with torch.no_grad():
                            output = model(sequence_tensor)
                            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                            pred_idx = np.argmax(probs)
                            confidence = probs[pred_idx]

                        # Afficher la prédiction en console
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        pred_action = actions[pred_idx]
                        if time.time() - last_prediction_time > 1.0:
                            print(f"{timestamp} - Geste détecté : {pred_action}, Précision : {confidence:.2%}")
                            save_prediction_to_csv(timestamp, pred_action, confidence)
                            last_prediction_time = time.time()

                        # Afficher la prédiction à l’écran
                        status_text = f"Geste : {pred_action} ({confidence:.2%})"

                        # Réinitialiser pour le prochain geste
                        gesture_active = False
                        sequence_buffer.clear()
                    else:
                        # Geste trop court, réinitialiser
                        gesture_active = False
                        sequence_buffer.clear()
                        status_text = "Geste trop court, réessayez"

            # Afficher le texte à l’écran
            cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Inférence ASL', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()