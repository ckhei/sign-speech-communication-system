import json
import os
import shutil

# Paths
json_path = "../data/WLASL/output/asl_videos.json"
video_root = "../data/WLASL/archive/videos" 
output_root = "../data/raw/asl"

os.makedirs(output_root, exist_ok=True)

# Load selected IDs
with open(json_path, "r", encoding="utf-8") as f:
    selected = json.load(f)

# Build a lookup of all video files by filename stem
all_files = {}
for root, dirs, files in os.walk(video_root):
    for file in files:
        name, ext = os.path.splitext(file)
        all_files[name] = os.path.join(root, file)

# Copy matching files into label folders
for label, video_ids in selected.items():
    label_folder = os.path.join(output_root, label.replace(" ", "_"))
    os.makedirs(label_folder, exist_ok=True)

    for vid in video_ids:
        if vid in all_files:
            src = all_files[vid]
            dst = os.path.join(label_folder, os.path.basename(src))
            shutil.copy2(src, dst)
        else:
            print(f"Missing file for ID: {vid}")

print("Done copying selected videos.")