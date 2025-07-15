import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
from model.lanenet.loss import DiscriminativeLoss, FocalLoss
from .loss import FocalLoss
import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
def compute_loss(net_output, binary_label, instance_label, loss_type = 'FocalLoss'):
    k_binary = 10    #1.7
    k_instance = 0.3
    k_dist = 1.0

    if(loss_type == 'FocalLoss'):
        loss_fn = FocalLoss(gamma=2, alpha=[0.25, 0.75])
    elif(loss_type == 'CrossEntropyLoss'):
        loss_fn = nn.CrossEntropyLoss()
    else:
        # print("Wrong loss type, will use the default CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss()
    
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = loss_fn(binary_seg_logits, binary_label)

    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss
    total_loss = binary_loss + instance_loss
    out = net_output["binary_seg_pred"]

    return total_loss, binary_loss, instance_loss, out

focal_loss_fn = FocalLoss(gamma=2)
cross_entropy_fn = nn.CrossEntropyLoss()

def compute_loss(outputs, binary_targets, instance_targets=None, loss_type='FocalLoss'):
    binary_logits = outputs['binary_seg_logits']

    # Create a mask to focus loss only on the lower half
    _, _, H, W = binary_logits.shape
    roi_mask = torch.zeros_like(binary_targets, dtype=torch.bool)  # shape: [B, H, W]
    roi_mask[:, H // 2:, :] = True  # only include lower half

    # Now squeeze channel dim and apply mask
    targets = binary_targets.squeeze(1)  # shape: [B, H, W]

    logits = binary_logits.permute(0, 2, 3, 1)[roi_mask]  # [N, 2]
    labels = targets[roi_mask]                             # [N]

    if loss_type == 'FocalLoss':
        loss = focal_loss_fn(logits, labels)
    elif loss_type == 'CrossEntropyLoss':
        loss = cross_entropy_fn(logits, labels)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Instance loss (simple example with MSE, adjust as needed)
    instance_logits = outputs.get('instance_seg_logits', None)
    if instance_logits is not None and instance_targets is not None:
        instance_targets = instance_targets.squeeze(1).float()
        instance_loss_fn = torch.nn.MSELoss()
        instance_loss = instance_loss_fn(instance_logits, instance_targets)
    else:
        instance_loss = 0

    total_loss = loss + instance_loss
    return total_loss, loss, instance_loss


def compute_loss(binary_logits, binary_labels, loss_type='FocalLoss'):
    if loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
        binary_logits = binary_logits.squeeze(1)  # if using single-channel output
        binary_labels = binary_labels.float()
    elif loss_type == 'FocalLoss':
        from model.lanenet.loss import FocalLoss  # or wherever your custom loss is
        criterion = FocalLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return criterion(binary_logits, binary_labels)
'''

def compute_loss(net_output, binary_label, instance_label, loss_type='FocalLoss'):
    k_binary = 10    # weight for binary loss
    k_instance = 0.3 # weight for instance variance loss
    k_dist = 1.0     # weight for instance distance loss

    if loss_type == 'FocalLoss':
        loss_fn = FocalLoss(gamma=2, alpha=[0.25, 0.75])
    elif loss_type == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    binary_seg_logits = net_output["binary_seg_logits"]  # shape: [B, C, H, W]
    pix_embedding = net_output["instance_seg_logits"]    # shape: [B, embedding_dim, H, W]

    B, C, H, W = binary_seg_logits.shape

    # Create ROI mask: only lower half (height//2 to H)
    roi_mask = torch.zeros((B, H, W), dtype=torch.bool, device=binary_seg_logits.device)
    roi_mask[:, H//2:, :] = True

    # Apply ROI mask to binary logits and labels
    # First, flatten spatial dims and batch
    binary_logits_flat = binary_seg_logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    binary_label_flat = binary_label.reshape(-1)                               # [B*H*W]
    roi_mask_flat = roi_mask.reshape(-1)                                      # [B*H*W]

    # Filter with ROI mask
    binary_logits_roi = binary_logits_flat[roi_mask_flat]
    binary_label_roi = binary_label_flat[roi_mask_flat]

    binary_loss = loss_fn(binary_logits_roi, binary_label_roi)

    # For instance segmentation loss, mask embeddings and labels similarly
    # pix_embedding shape: [B, emb_dim, H, W] -> flatten spatial dims
    B, emb_dim, H, W = pix_embedding.shape
    pix_embedding_roi = pix_embedding[:, :, H//2:, :]   # shape: [B, emb_dim, H/2, W]
    instance_label_roi = instance_label[:, H//2:, :]    # shape: [B, H/2, W]


    # The DiscriminativeLoss should support batch inputs of shape [N, emb_dim]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)

    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding_roi, instance_label_roi)

    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss

    total_loss = binary_loss + instance_loss
    out = net_output["binary_seg_logits"]

    return total_loss, binary_loss, instance_loss, out

def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type = 'FocalLoss', num_epochs=25):
    since = time.time()
    training_log = {
        'epoch': [],
        'training_loss': [],
        'val_loss': [],
        'binary_loss_train': [],
        'binary_loss_val': [],
        'instance_loss_train': [],
        'instance_loss_val': []
    }

    best_loss = float("inf")

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        training_log['epoch'].append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss_b = 0.0
            running_loss_i = 0.0

            # Iterate over data.
            for inputs, binarys, instances in dataloaders[phase]:
                inputs = inputs.type(torch.FloatTensor).to(device)
                binarys = binarys.type(torch.LongTensor).to(device)
                instances = instances.type(torch.FloatTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                  
                    # Compute ROI-aware loss
                    total_loss, binary_loss, instance_loss, _ = compute_loss(outputs, binarys, instances, loss_type)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                # statistics
                running_loss += total_loss.item() * inputs.size(0)
                running_loss_b += binary_loss.item() * inputs.size(0)
                running_loss_i += instance_loss.item() * inputs.size(0)

            if scheduler is not None and phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            binary_loss = running_loss_b / dataset_sizes[phase]
            instance_loss = running_loss_i / dataset_sizes[phase]
            print(f'{phase} Total Loss: {epoch_loss:.4f} | Binary Loss: {binary_loss:.4f} | Instance Loss: {instance_loss:.4f}')

            # deep copy the model
            if phase == 'train':
                training_log['training_loss'].append(epoch_loss)
                training_log['binary_loss_train'].append(binary_loss)
                training_log['instance_loss_train'].append(instance_loss)                
            if phase == 'val':
                training_log['val_loss'].append(epoch_loss)
                training_log['binary_loss_val'].append(binary_loss)
                training_log['instance_loss_val'].append(instance_loss)                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val_loss: {:4f}'.format(best_loss))
    training_log['training_loss'] = np.array(training_log['training_loss'])
    training_log['val_loss'] = np.array(training_log['val_loss'])
    training_log['binary_loss_train'] = np.array(training_log['binary_loss_train'])
    training_log['binary_loss_val'] = np.array(training_log['binary_loss_val'])
    training_log['instance_loss_train'] = np.array(training_log['instance_loss_train'])
    training_log['instance_loss_val'] = np.array(training_log['instance_loss_val'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable