import os
import json

# Path to your JSON file
json_file_path = r"E:\New folder (3)\WASL\WLASL_v0.3.json"  # Replace with the path to your JSON file

# Directory containing video files (e.g., "4100.mp4")
video_directory = r"E:\New folder (3)\WASL\videos"  # Replace with the path to your video directory

# Load the JSON file
try:
    with open(json_file_path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON file not found at {json_file_path}")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in {json_file_path}")
    exit(1)

# Create a mapping of video_id to gloss
video_id_to_gloss = {}
for gloss_entry in data:
    gloss = gloss_entry.get("gloss")
    if not gloss:
        print("Warning: Gloss entry missing 'gloss' key. Skipping.")
        continue
    for instance in gloss_entry.get("instances", []):
        video_id = instance.get("video_id")
        if video_id:
            video_id_to_gloss[video_id] = gloss
        else:
            print("Warning: Instance missing 'video_id'. Skipping.")

# Rename video files
for filename in os.listdir(video_directory):
    if filename.endswith(".mp4"):  # Adjust extension if needed (e.g., ".mov")
        video_id = filename.split(".")[0]  # Extract video_id (e.g., "4100" from "4100.mp4")
        if video_id in video_id_to_gloss:
            gloss = video_id_to_gloss[video_id]
            base_new_filename = f"{gloss}.mp4"  # Initial new name (e.g., "book.mp4")
            new_filename = base_new_filename
            old_file_path = os.path.join(video_directory, filename)
            new_file_path = os.path.join(video_directory, new_filename)

            # Handle filename conflicts (e.g., multiple "book" videos)
            counter = 1
            while os.path.exists(new_file_path):
                # Try appending video_id first
                new_filename = f"{gloss}_{video_id}.mp4"
                new_file_path = os.path.join(video_directory, new_filename)
                if not os.path.exists(new_file_path):
                    break
                # If still conflicts, append a counter
                new_filename = f"{gloss}_{counter}.mp4"
                new_file_path = os.path.join(video_directory, new_filename)
                counter += 1

            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {filename} → {new_filename}")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")
        else:
            print(f"No gloss found for video_id: {video_id}. Skipping.")
    else:
        print(f"Skipping non-video file: {filename}")
