import time
import os
import sys
import yaml

import torch
from dataloader.locoEYE_data_loaders import RailDataset
from dataloader.locoEYE_transformers import Rescale, ToTensor, Compose
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader, dataloader
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from model.utils.cli_helper_eval import parse_args
from model.eval_function import Eval_Score

import numpy as np
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EvalDatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get tuple from base dataset
        img, label_binary, label_instance = self.base_dataset[idx]
        # Convert to dict for transforms
        sample = {'image': img, 'label': label_binary, 'instance': label_instance}
        # Apply transforms (if any) expecting dict
        if self.base_dataset.transform:
            sample = self.base_dataset.transform(sample)
        return sample['image'], sample['label'], sample['instance']
    
def evaluation():
    args = parse_args()
    # resize_height = args.height
    # resize_width = args.width
    # dataset_file = os.path.join(args.dataset, 'test.txt')
    # model_path = args.model
    # model = LaneNet(arch=args.model_type)


    # 🔧 Load configuration from YAML
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(ROOT_DIR, "config", "preprocess_config.yaml")
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    resize_height = config["height"]
    resize_width = config["width"]
    dataset_file = os.path.join(ROOT_DIR, config["val_txt"])
    model_path = os.path.join(ROOT_DIR, config["save_path"], 'best_model.pth')
    model = LaneNet(arch=config["model_type"])

    # data_transform = transforms.Compose([
    #     transforms.Resize((resize_height,  resize_width)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    data_transform = Compose([
        Rescale((resize_width, resize_height)),
        ToTensor()
    ])

    # target_transforms = transforms.Compose([
    #     Rescale((resize_width, resize_height)),
    # ])

    # Eval_Dataset = RailDataset(dataset_file, transform=data_transform, target_transform=target_transforms)
    Eval_Dataset = RailDataset(
        dataset_file,
        transform=data_transform,
        target_transform=None
    )

    eval_dataloader = DataLoader(Eval_Dataset, batch_size=1, shuffle=True)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    iou, dice = 0, 0
    with torch.no_grad():
        for x, target_binary, target_instance in eval_dataloader:
            # y = model(x.to(DEVICE))
            # y_pred = torch.squeeze(y['binary_seg_logits'].to('cpu')).numpy()
            # y_true = torch.squeeze(target).numpy()
            # Score = Eval_Score(y_pred, y_true)
            # dice += Score.Dice()
            # iou += Score.IoU()

            outputs = model(x)
            
            # Process binary segmentation logits
            binary_logits = outputs['binary_seg_logits']
            binary_probs = torch.sigmoid(binary_logits)  # if binary is single channel logits
            binary_pred = (binary_probs > 0.5).float()
            # y_pred_binary = torch.squeeze(binary_pred.to('cpu')).numpy()
            # Properly convert tensor to numpy and squeeze dims
            y_pred_binary = binary_pred.detach().cpu().numpy()
            y_pred_binary = np.squeeze(y_pred_binary)  # remove batch and channel dims if present            


            # Ensure it's a NumPy array now
            if not isinstance(y_pred_binary, np.ndarray):
                raise TypeError(f"y_pred_binary is not a NumPy array: {type(y_pred_binary)}")

            # Process instance segmentation logits
            instance_logits = outputs['instance_seg_logits']
            # Depending on instance output format (e.g., per-pixel embedding vectors or masks)
            # You might need to do argmax, clustering, or some post-processing here
            # For now, if it's logits for instance classes, do:
            instance_probs = F.softmax(instance_logits, dim=1)
            instance_pred = torch.argmax(instance_probs, dim=1)
            y_pred_instance = torch.squeeze(instance_pred.to('cpu')).numpy()

            # Now use y_pred_binary and y_pred_instance for evaluation
            # Example:
            # Score = Eval_Score(y_pred_binary, target_binary.numpy())

            target_binary = target_binary.numpy()
            # Ensure shapes match for metric calculation:
            # Resize prediction if shape mismatch
            if y_pred_binary.shape != target_binary.shape:
                # Make sure type is uint8 (required by OpenCV)
                y_pred_binary_resized = cv2.resize(
                    y_pred_binary.astype(np.uint8),
                    (target_binary.shape[1], target_binary.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                y_pred_binary_resized = y_pred_binary

            Score = Eval_Score(y_pred_binary_resized, target_binary)

            dice += Score.Dice()
            iou += Score.IoU()            
    
    print('Final_IoU: %s'% str(iou/len(eval_dataloader.dataset)))
    print('Final_F1: %s'% str(dice/len(eval_dataloader.dataset)))


if __name__ == "__main__":
    evaluation()