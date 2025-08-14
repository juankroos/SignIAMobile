import cv2
import numpy as np
import os
import mediapipe as mp
import torch
from torch.utils.data import Dataset

# Initialiser MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Indices des landmarks faciaux pertinents pour l’ASL
FACIAL_LANDMARKS = [
    10, 152, 234, 263, 33, 133, 61, 291, 0, 13,
    78, 308, 80, 310, 159, 145, 468, 473, 21, 251
]


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    # Dessiner les landmarks du corps
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    # Dessiner les mains
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    # Dessiner les landmarks faciaux (seulement si détectés)
    if results.face_landmarks:
        for idx in FACIAL_LANDMARKS:
            try:
                lm = results.face_landmarks.landmark[idx]
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)
            except IndexError:
                print(f"Warning: Facial landmark index {idx} out of range in frame")
                continue


def normalize_landmarks(landmarks, ref_point):
    landmarks = landmarks - ref_point
    scale = np.linalg.norm(landmarks[11] - landmarks[12]) if landmarks[11].sum() != 0 and landmarks[
        12].sum() != 0 else 1.0
    if scale > 0:
        landmarks /= scale
    return landmarks


def extract_keypoints(results):
    # Corps
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    # Mains
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    # Visage
    face = np.zeros(len(FACIAL_LANDMARKS) * 3)
    if results.face_landmarks:
        try:
            face = np.array([[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y,
                              results.face_landmarks.landmark[i].z]
                             for i in FACIAL_LANDMARKS]).flatten()
        except IndexError as e:
            print(f"Warning: Error accessing facial landmarks: {e}")

    # Normalisation
    ref_point = np.array([results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y,
                          results.pose_landmarks.landmark[0].z]) if results.pose_landmarks else np.zeros(3)
    pose_3d = pose.reshape(-1, 4)[:, :3] if pose.size > 0 else np.zeros((33, 3))
    lh_3d = lh.reshape(-1, 3) if lh.size > 0 else np.zeros((21, 3))
    rh_3d = rh.reshape(-1, 3) if rh.size > 0 else np.zeros((21, 3))
    face_3d = face.reshape(-1, 3) if face.size > 0 else np.zeros((len(FACIAL_LANDMARKS), 3))

    pose_3d = normalize_landmarks(pose_3d, ref_point)
    lh_3d = normalize_landmarks(lh_3d, ref_point)
    rh_3d = normalize_landmarks(rh_3d, ref_point)
    face_3d = normalize_landmarks(face_3d, ref_point)

    return np.concatenate([pose_3d.flatten(), lh_3d.flatten(), rh_3d.flatten(), face_3d.flatten()])


# Fonctions d’augmentation
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
    return sequence[keep_mask]


# Dataset pour Transformer
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
                        self.data.append(frames)
                        self.labels.append(action_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = np.array(self.data[idx])
        label = self.labels[idx]

        # Normaliser la longueur
        if len(sequence) < self.sequence_length:
            sequence = np.pad(sequence, ((0, self.sequence_length - len(sequence)), (0, 0)), mode='edge')
        elif len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]

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
                sequence = crop_pad(sequence, 1.0)
            if np.random.rand() < 0.2:
                sequence = frame_dropout(sequence, 0.05)
                sequence = crop_pad(sequence, 1.0)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Extraction des landmarks
DATA_PATH = os.path.join('MP_DatafinalTrans')
VIDEO_PATH = r'E:\Dataset1'
sequence_length = 30

actions = [d for d in os.listdir(VIDEO_PATH) if os.path.isdir(os.path.join(VIDEO_PATH, d))]

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        action_path = os.path.join(VIDEO_PATH, action)
        video_files = [f for f in os.listdir(action_path) if f.endswith(('.mp4', '.avi', '.mov'))]

        for video_idx, video_file in enumerate(video_files):
            video_path = os.path.join(action_path, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                continue

            sequence_path = os.path.join(DATA_PATH, action, str(video_idx))
            os.makedirs(sequence_path, exist_ok=True)

            frame_num = 0
            keypoints_sequence = []
            while cap.isOpened() and frame_num < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: End of video {video_path} at frame {frame_num}")
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Afficher l’état
                if frame_num == 0:
                    cv2.putText(image, f'Collecting {action} video {video_idx}', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.waitKey(500)
                cv2.imshow('OpenCV Feed', image)

                # Extraire et sauvegarder les keypoints
                keypoints = extract_keypoints(results)
                keypoints_sequence.append(keypoints)
                npy_path = os.path.join(sequence_path, str(frame_num))
                np.save(npy_path, keypoints)

                frame_num += 1
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Remplir la séquence si trop courte
            while len(keypoints_sequence) < sequence_length:
                keypoints_sequence.append(keypoints_sequence[-1] if keypoints_sequence else np.zeros(
                    33 * 4 + 21 * 3 + 21 * 3 + len(FACIAL_LANDMARKS) * 3))

            cap.release()
    cv2.destroyAllWindows()