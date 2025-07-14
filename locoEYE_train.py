import os
import yaml
import torch
import pandas as pd

# Local imports
from model.lanenet.train_lanenet import train_model
from dataloader.locoEYE_data_loaders import RailDataset
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from dataloader.locoEYE_transformers import Rescale, ToTensor, Compose
from model.utils.cli_helper import parse_args

# Set device: use GPU if available
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    args = parse_args()
    resize_height = args.height
    resize_width = args.width

    # 🔧 Load configuration from YAML
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(ROOT_DIR, "config", "preprocess_config.yaml")
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    save_path =  os.path.join(ROOT_DIR, config["save_path"])
    train_dataset_file = os.path.join(ROOT_DIR, config["train_txt"])
    val_dataset_file = os.path.join(ROOT_DIR, config["val_txt"])

    # === Define data transforms for training and validation ===
    data_transforms = {
        'train': Compose([
            Rescale((resize_width, resize_height)),
            ToTensor()
        ]),
        'val': Compose([
            Rescale((resize_width, resize_height)),
            ToTensor()
        ])
    }
   
    # === Load datasets ===
    train_dataset = RailDataset(train_dataset_file, transform=data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    val_dataset = RailDataset(val_dataset_file, transform=data_transforms['val'])
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    dataloaders = {
        'train' : train_loader,
        'val' : val_loader
    }
    dataset_sizes = {'train': len(train_loader.dataset), 'val' : len(val_loader.dataset)}

    # === Initialize model ===
    model = LaneNet(arch=args.model_type)
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{args.epochs} epochs {len(train_dataset)} training samples\n")

    # === Train the model ===
    model, log = train_model(model, optimizer, scheduler=None, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=DEVICE, loss_type=args.loss_type, num_epochs=args.epochs)
    # === Save training log to CSV ===
    df=pd.DataFrame({'epoch':[],'training_loss':[],'val_loss':[]})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['val_loss'] = log['val_loss']

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename, columns=['epoch','training_loss','val_loss'], header=True,index=False,encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))

    # === Save best model checkpoint ===
    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))

if __name__ == '__main__':
    train()
