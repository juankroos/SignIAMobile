import cv2
import numpy as np
import os
import mediapipe as mp
import torch
from torch.utils.data import Dataset

# Suppress TensorFlow oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
shoulder_landmarks = mp.solutions.holistic.PoseLandmark


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


def adjust_sequence_length(keypoints_sequence, target_length=30):
    """Adjust sequence to target_length by subsampling or padding."""
    current_length = len(keypoints_sequence)
    if current_length == target_length:
        return np.array(keypoints_sequence)

    if current_length > target_length:
        # Uniformly subsample to target_length
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return np.array(keypoints_sequence)[indices]
    else:
        # Pad with last frame or zeros
        pad_length = target_length - current_length
        if keypoints_sequence:
            pad = np.repeat([keypoints_sequence[-1]], pad_length, axis=0)
            return np.concatenate([keypoints_sequence, pad], axis=0)
        else:
            return np.zeros((target_length, 33 * 3 + 21 * 3 + 21 * 3))


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


if __name__ == '__main__':
    DATA_PATH = os.path.join('MP_DatafinalTransmove')
    VIDEO_PATH = r'E:\Dataset1'
    sequence_length = 30

    actions = [d for d in os.listdir(VIDEO_PATH) if os.path.isdir(os.path.join(VIDEO_PATH, d))]

    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.5) as holistic:
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
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                hand_detected = False
                print(f"Processing {video_path}: {total_frames} frames")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"End of video {video_path} at frame {frame_num}")
                        break

                    image, results = mediapipe_detection(frame, holistic)

                    # Start collecting when a hand is detected
                    if not hand_detected and (results.left_hand_landmarks or results.right_hand_landmarks):
                        hand_detected = True
                        print(f"Hand detected at frame {frame_num}")

                    if hand_detected:
                        keypoints = extract_keypoints(results)
                        keypoints_sequence.append(keypoints)

                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, f'Collecting {action} video {video_idx}', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.waitKey(500)
                    cv2.imshow('OpenCV Feed', image)

                    frame_num += 1
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                # Adjust sequence to exactly 30 frames
                keypoints_sequence = adjust_sequence_length(keypoints_sequence, sequence_length)

                # Save frames
                for i, keypoints in enumerate(keypoints_sequence):
                    npy_path = os.path.join(sequence_path, str(i))
                    np.save(npy_path, keypoints)

                # Print verification
                print(f"Sequence {os.path.join(action, str(video_idx))}:")
                print(f"  Collected {len(keypoints_sequence)} frames after hand detection")
                print(
                    f"  Non-zero values: {np.sum(np.array(keypoints_sequence) != 0)}/{len(keypoints_sequence) * (33 * 3 + 21 * 3 + 21 * 3)}")

                cap.release()
        cv2.destroyAllWindows()