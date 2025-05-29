import os
import random
import yaml

# 🔧 Load configuration from YAML
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from tools/
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "preprocess_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

images_dir = os.path.join(ROOT_DIR, config["clips_path"])
bmasks_dir = os.path.join(ROOT_DIR, config["bmask_dir_cleanup"])
imasks_dir = os.path.join(ROOT_DIR, config["instance_mask"])
train_txt = os.path.join(ROOT_DIR, config["train_txt"])
val_txt = os.path.join(ROOT_DIR, config["val_txt"])

# Get all cleaned binary mask files
binary_mask_files = sorted([
    f for f in os.listdir(bmasks_dir)
    if f.startswith('cleaned_') and f.endswith(('.jpg', '.png'))
])

# Infer corresponding image file names
image_files = []
for mask_file in binary_mask_files:
    base_name = mask_file.replace('cleaned_', '')  # e.g., "cleaned_001.jpg" -> "001.jpg"
    image_path = os.path.join(images_dir, base_name)
    if os.path.exists(image_path):
        image_files.append(base_name)

# Shuffle and split into train and val (80% train, 20% val)
random.shuffle(image_files)
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def write_list(file_list, txt_file):
    with open(txt_file, 'w') as f:
        for img_file in file_list:
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(images_dir, img_file).replace('\\', '/')
            binary_mask_name = f'cleaned_{base_name}.jpg'
            instance_mask_name = f'seg_{base_name}.jpg'

            binary_mask_path = os.path.join(bmasks_dir, binary_mask_name).replace('\\', '/')
            instance_mask_path = os.path.join(imasks_dir, instance_mask_name).replace('\\', '/')

            f.write(f"{img_path} {binary_mask_path} {instance_mask_path}\n")

def verify_dataset_file(txt_file):
    missing_images = []
    missing_binary_masks = []
    missing_instance_masks = []

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            print(f"⚠️ Skipping malformed line: {line.strip()}")
            continue

        img_path, bmask_path, imask_path = parts

        if not os.path.exists(img_path):
            missing_images.append(img_path)
        if not os.path.exists(bmask_path):
            missing_binary_masks.append(bmask_path)
        if not os.path.exists(imask_path):
            missing_instance_masks.append(imask_path)

    print(f"\nChecking {txt_file}:")
    if not missing_images and not missing_binary_masks and not missing_instance_masks:
        print("✅ All image, binary mask, and instance mask paths exist!")
    else:
        if missing_images:
            print("❌ Missing image files:")
            for p in missing_images:
                print(f"   {p}")
        if missing_binary_masks:
            print("❌ Missing binary mask files:")
            for p in missing_binary_masks:
                print(f"   {p}")
        if missing_instance_masks:
            print("❌ Missing instance mask files:")
            for p in missing_instance_masks:
                print(f"   {p}")

# out_path = os.path.join('./Tracks', 'train.txt')
write_list(train_files, train_txt)
verify_dataset_file(train_txt)
# out_path = os.path.join('./Tracks', 'val.txt')
write_list(val_files, val_txt)
verify_dataset_file(val_txt)