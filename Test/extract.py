# keypoints_extraction.py

import cv2
import numpy as np
import os
import mediapipe as mp

# Initialisation des modules MediaPipe
mp_holistic = mp.solutions.holistic

mp_drawing = mp.solutions.drawing_utils

# --- Fonction : détection MediaPipe + conversion image
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# --- Fonction : affichage des landmarks
def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

# --- Fonction : extraire les keypoints (pose + main gauche + main droite)
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
                    ).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
                  ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
                  ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])  # Taille : 258

# --- Paramètres
DATA_PATH = os.path.join('MP_Datafinal12')     # Où sauvegarder les séquences
VIDEO_PATH = r'E:\Dataset1'                    # Chemin vers les vidéos
sequence_length = 30                           # Longueur max par vidéo

# --- Liste des classes (actions)
actions = [d for d in os.listdir(VIDEO_PATH) if os.path.isdir(os.path.join(VIDEO_PATH, d))]

# --- Détection Holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        action_path = os.path.join(VIDEO_PATH, action)
        video_files = [f for f in os.listdir(action_path) if f.endswith(('.mp4', '.avi', '.mov'))]

        for video_idx, video_file in enumerate(video_files):
            video_path = os.path.join(action_path, video_file)
            cap = cv2.VideoCapture(video_path)

            sequence_path = os.path.join(DATA_PATH, action, str(video_idx))
            os.makedirs(sequence_path, exist_ok=True)

            frame_num = 0
            while cap.isOpened() and frame_num < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Affichage démarrage
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.waitKey(500)

                # Affichage de l’image avec les landmarks
                cv2.imshow('OpenCV Feed', image)

                # Extraction et sauvegarde
                keypoints = extract_keypoints(results)
                np.save(os.path.join(sequence_path, str(frame_num)), keypoints)

                frame_num += 1
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
    cv2.destroyAllWindows()
