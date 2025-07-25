import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import mediapipe as mp
import csv
from datetime import datetime

OUTPUT_CSV = 'action_predictions.csv'

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join(r'E:\Gesture-Recognition-using-3D-CNN\MP_Datafinal1')

actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])

if len(actions) == 0:
    print(f"Erreur : Aucun dossier d'action trouvé dans {DATA_PATH}.")
    exit()

print(f"Actions détectées : {actions}")

# Charger le modèle CNN 3D
model = load_model('action_3dcnn_improved.h5')

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

def normalize_keypoints(keypoints):
    keypoints = keypoints.copy()
    pose_landmarks = keypoints[:33*4].reshape(33, 4)
    torso_center = np.mean(pose_landmarks[11:13, :2], axis=0)
    keypoints[:33*4:4] -= torso_center[0]
    keypoints[1:33*4:4] -= torso_center[1]
    return keypoints

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)] * len(actions)
colors = colors[:len(actions)]

sequence = []
sentence = []
predictions = []
threshold = 0.7  # Augmenter le seuil pour plus de stabilité

# Initialiser le fichier CSV
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Action', 'Probability'])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire la frame.")
            break

        image, results = mediapipe_detection(frame, holistic)

        # Débogage
        print("Main gauche détectée :", results.left_hand_landmarks is not None)
        print("Main droite détectée :", results.right_hand_landmarks is not None)

        # Dessiner les landmarks
        draw_styled_landmarks(image, results)

        # Extraire et normaliser les keypoints
        keypoints = extract_keypoints(results)
        keypoints = normalize_keypoints(keypoints)
        keypoints = keypoints[:256].reshape(16, 16, 1)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0)  # Forme : (1, 30, 16, 16, 1)
            res = model.predict(input_data)[0]
            predicted_action = actions[np.argmax(res)]
            print(f"Action prédite : {predicted_action} (Probabilité : {res[np.argmax(res)]:.4f})")
            predictions.append(np.argmax(res))

            # Logique de stabilité
            if len(predictions) >= 5 and np.unique(predictions[-5:]).size == 1 and np.unique(predictions[-5:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) == 0 or (len(sentence) > 0 and predicted_action != sentence[-1]):
                        sentence.append(predicted_action)
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(OUTPUT_CSV, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, predicted_action, f"{res[np.argmax(res)]:.4f}"])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualiser les probabilités
            image = prob_viz(res, actions, image, colors)

        # Afficher la phrase prédite
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Afficher à l'écran
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
