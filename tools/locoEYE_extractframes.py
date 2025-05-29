"""
video_to_frames.py

Extracts one frame per second from a video and saves them as images.
Configuration is loaded from a YAML file: config/preprocess_config.yaml

Usage:
    python video_to_frames.py

YAML Config Format:
    video_path: "path/to/input/video.mp4"
    output_dir: "path/to/output/frames/"
"""
import cv2
import os
from tqdm import tqdm
import yaml

# 🔧 Load configuration from YAML
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from tools/
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "preprocess_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

video_path = os.path.join(ROOT_DIR, config["video_path"])
output_dir = os.path.join(ROOT_DIR, config["clips_path"])

# 🗂️ Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 🎥 Open the video file
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = int(total_frames / fps)

print(f"🎥 FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {duration_sec} sec")

frame_id = 0

# 🖼️ Extract one frame per second
with tqdm(total=duration_sec, desc='Extracting frames') as pbar:
    for sec in range(duration_sec):
        target_frame = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_dir, f"{str(frame_id).zfill(4)}.jpg")
        cv2.imwrite(filename, frame)
        frame_id += 1
        pbar.update(1)

# ✅ Clean up
cap.release()
print(f"✅ Extracted {frame_id} frames (1 per second) to {output_dir}")