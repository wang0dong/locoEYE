import os
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from model.lanenet.LaneNet import LaneNet
from model.utils.cli_helper_test import parse_args

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_test_data(img_path, transform):
    img = Image.open(img_path).convert('RGB')  # Ensure RGB
    img = transform(img)
    return img

def test():
    args = parse_args()
    test_list_path = './datasets/tracks/test.txt'  # file containing image paths
    resize_height = args.height
    resize_width = args.width

    # Define output subfolders
    output_root = 'test_output'
    folders = {
        'input': os.path.join(output_root, 'input'),
        'binary_heatmap': os.path.join(output_root, 'binary_heatmap'),
        'binary_output': os.path.join(output_root, 'binary_output'),
        'instance_output': os.path.join(output_root, 'instance_output')
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # Define transforms
    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor()
    ])

    # Load model once
    model_path = args.model
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    with open(test_list_path, 'r') as f:
        img_paths = [line.strip() for line in f if line.strip()]

    for img_path in img_paths:
        print(f"Processing {img_path}")
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Load and preprocess input
        input_tensor = load_test_data(img_path, data_transform).to(DEVICE)
        input_tensor = input_tensor.unsqueeze(0)  # batch dim

        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)

        # Save original resized image
        input_img = Image.open(img_path).convert('RGB')
        input_img_resized = input_img.resize((resize_width, resize_height))
        input_np = np.array(input_img_resized)
        input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(folders['input'], f"{base_name}_input.jpg"), input_bgr)

        # Instance segmentation output
        instance_logits = outputs['instance_seg_logits']  # [B,C,H,W]
        instance_pred = torch.squeeze(instance_logits.detach().cpu()).numpy()
        instance_pred_img = (instance_pred * 255).astype(np.uint8)
        if instance_pred_img.ndim == 3:
            instance_pred_img = instance_pred_img.transpose((1, 2, 0))
        cv2.imwrite(os.path.join(folders['instance_output'], f"{base_name}_instance_output.jpg"), instance_pred_img)

        # Binary segmentation output
        binary_logits = outputs['binary_seg_logits']  # [B,1,H,W]
        binary_probs = torch.sigmoid(binary_logits)
        binary_heatmap = binary_probs.squeeze().cpu().numpy()
        if binary_heatmap.ndim == 3:
            binary_heatmap = binary_heatmap[0]
        binary_heatmap_img = (binary_heatmap * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(folders['binary_heatmap'], f"{base_name}_binary_heatmap.jpg"), binary_heatmap_img)

        threshold = 0.7
        binary_pred = (binary_probs > threshold).squeeze().cpu().numpy().astype(np.uint8) * 255
        if binary_pred.ndim == 3:
            binary_pred = binary_pred[0]
        cv2.imwrite(os.path.join(folders['binary_output'], f"{base_name}_binary_output.jpg"), binary_pred)

if __name__ == "__main__":
    test()
