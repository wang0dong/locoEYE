import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.morphology import skeletonize
from sklearn.metrics import pairwise_distances_argmin
import yaml
import argparse

# 🔧 Utility Functions

def filter_by_shape(mask, min_area = 20):
    """
    Remove small blobs based on area from the binary mask.
    """    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            filtered_mask = cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filtered_mask

def filter_by_length(mask, min_length=20):
    """
    Filter out short segments using skeleton length.
    """    
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255

    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered_mask = np.zeros_like(mask)

    for cnt in contours:
        length = cv2.arcLength(cnt, closed=False)
        if length >= min_length:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=3)  # re-thicken line

    return filtered_mask

def convert_binary_to_instance(binary, color_img, special_colors, visualize=False):
    """
    Convert binary mask into instance mask using nearest color matching.
    """    
    # Step 1: Ensure binary mask (0 or 255)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    # Step 2: Connected components on binary mask
    num_labels, labels = cv2.connectedComponents(binary)

    # Step 3: For each component, compute mean color in color image
    segment_colors = []
    component_masks = []
    for i in range(1, num_labels):  # skip background (0)
        mask = (labels == i).astype(np.uint8)
        mean_color = cv2.mean(color_img, mask)[0:3]  # BGR
        segment_colors.append(mean_color[::-1])  # Convert to RGB
        component_masks.append(mask)

    # Step 4: Assign each segment to nearest special color
    segment_colors = np.array(segment_colors, dtype=np.uint8)
    assigned_ids = pairwise_distances_argmin(segment_colors, special_colors)

    # Step 5: Build instance mask
    instance_mask = np.zeros_like(binary)
    for i, group_id in enumerate(assigned_ids):
        instance_mask[component_masks[i] == 1] = group_id + 1  # Label starts at 1

    # Optional cleanup (remove very small blobs)
    kernel = np.ones((3, 3), np.uint8)
    instance_mask = cv2.morphologyEx(instance_mask, cv2.MORPH_OPEN, kernel)

    # Optional visualization
    if visualize:
        color_mask = cv2.applyColorMap((instance_mask * 40).astype(np.uint8), cv2.COLORMAP_JET)
        return instance_mask, color_mask

    return instance_mask

def create_binary_mask(img, special_colors, tolerance=5):
    """
    Generate a cleaned binary mask from the input color image.
    """
    height, width = img.shape[:2]
    roi_start = height // 2
    
    roi = img[roi_start:, :]

    # Convert to RGB to match manual color codes
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Create binary mask from color matches
    tolerant_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)

    for color in special_colors:
        lower = np.clip(np.array(color) - tolerance, 0, 255)
        upper = np.clip(np.array(color) + tolerance, 0, 255)
        match = cv2.inRange(roi_rgb, lower, upper)
        tolerant_mask = cv2.bitwise_or(tolerant_mask, match)

    # Step 1: Dilate to merge small fragmented regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_mask = cv2.dilate(tolerant_mask, kernel, iterations=3)

    # Step 2: Close to fill gaps and holes
    closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 3: Filter small blobs and noise
    filtered_mask = filter_by_shape(closed_mask)
    filtered_mask = filter_by_length(filtered_mask)

    # Place into full image space
    final_mask = np.zeros((height, width), dtype=np.uint8)
    final_mask[roi_start:, :] = filtered_mask

    return final_mask

# 🔧  Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate rail track masks from labeled images"
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['binary', 'instance'],
        default='binary',
        help=(
            "Choose mask generation type:\n"
            "  'binary'   - generate a binary mask where all rails are labeled as foreground\n"
            "  'instance' - generate an instance mask where each rail segment has a unique ID"
        )
    )

    return parser.parse_args()

# 🔧 Load configuration from YAML
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from tools/
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "preprocess_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

manual_mask = os.path.join(ROOT_DIR, config["manual_mask"])
bmask_dir = os.path.join(ROOT_DIR, config["binary_mask"])
insmask_dir = os.path.join(ROOT_DIR, config["instance_mask"])
bmask_dir_cleanup = os.path.join(ROOT_DIR, config["bmask_dir_cleanup"])
rail_colors = config["rail_colors"]

os.makedirs(bmask_dir, exist_ok=True)
os.makedirs(insmask_dir, exist_ok=True)

# --- Main Execution ---
args = parse_args()

if args.mode == 'binary': 
    # clean up binary mask file manual
    for filename in tqdm(sorted(os.listdir(manual_mask))):
        if not filename.endswith('.jpg'):
            continue
        # Read manual mask
        mask_path = os.path.join(manual_mask, filename)
        img = cv2.imread(mask_path)
        binary_mask = create_binary_mask(img, rail_colors)
        output_path = os.path.join(bmask_dir, filename)
        cv2.imwrite(output_path, binary_mask) 

elif args.mode == 'instance':
    mask_files = [f for f in os.listdir(bmask_dir_cleanup) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for fname in tqdm(mask_files):
        binary_mask = cv2.imread(os.path.join(bmask_dir_cleanup, fname), cv2.IMREAD_GRAYSCALE)  # Load the binary mask
        color_img_file = fname.replace("cleaned_", "")
        color_img = cv2.imread(os.path.join(manual_mask, color_img_file))
        if binary_mask is None or color_img is None:
            print(f"Skipping {fname}: could not load binary or color image.")
            continue
        # Run instance segmentation
        instance_mask, color_mask = convert_binary_to_instance(binary_mask, color_img, rail_colors, visualize=True)
        new_name = fname.replace("cleaned_", "seg_")
        out_path = os.path.join(insmask_dir, new_name)
        vis_mask = (instance_mask * (255.0 / instance_mask.max())).astype(np.uint8)
        cv2.imwrite(out_path, vis_mask)  # Save the instance mask
        new_name = fname.replace("cleaned_", "seg_col")
        out_path = os.path.join(insmask_dir, new_name)
        cv2.imwrite(out_path, color_mask)  # Save the instance mask

else:
    raise ValueError("Invalid mode. Use 'binary' or 'instance'.")