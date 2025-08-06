import os
import shutil

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
KEYWORD = 'miroir'
SOURCE_DIR = r'E:\Dataset1 - Copy'
MIRROR_DIR = '/chemin/vers/dossier_miroir'

def is_video(file):
    return os.path.splitext(file)[1].lower() in VIDEO_EXTENSIONS

def move_mirror_videos(base_dir, target_dir):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if is_video(file) and KEYWORD in file.lower():
                relative_path = os.path.relpath(root, base_dir)
                dest_dir = os.path.join(target_dir, relative_path)
                os.makedirs(dest_dir, exist_ok=True)

                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)

                shutil.move(src_file, dest_file)

def collect_non_mirror_video_counts(base_dir):
    video_counts = {}
    for root, _, files in os.walk(base_dir):
        non_mirror = [f for f in files if is_video(f) and KEYWORD not in f.lower()]
        if non_mirror:
            video_counts[root] = non_mirror
    return video_counts

def balance_video_distribution(video_counts):
    counts = {folder: len(videos) for folder, videos in video_counts.items()}
    min_count = min(counts.values())
    max_allowed = min_count + 3

    for folder, videos in video_counts.items():
        excess = len(videos) - max_allowed
        if excess > 0:
            candidates = videos[:excess]
            for video in candidates:
                # Trouve un dossier cible avec peu de vidéos
                targets = [f for f, c in counts.items() if c < max_allowed and f != folder]
                if not targets:
                    break
                target = min(targets, key=lambda x: counts[x])
                shutil.move(os.path.join(folder, video), os.path.join(target, video))
                counts[folder] -= 1
                counts[target] += 1

# Étapes exécutées
move_mirror_videos(SOURCE_DIR, MIRROR_DIR)
video_counts = collect_non_mirror_video_counts(SOURCE_DIR)
balance_video_distribution(video_counts)