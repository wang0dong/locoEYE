
<img src="/media/tigereyewide.jpg" alt="Known Tracks" title="banner" style="width: 100%; height: auto;">

# locoEYE: Real-Time Rail Detection
### 🚧 Problem Statement

Autonomous train navigation systems rely heavily on GPS for localization. However, during system initialization, GPS often suffers from location uncertainty, especially in rail yards, tunnels, or dense urban environments. This makes it impossible to determine which specific track the train is on using GPS alone — a critical issue for ensuring safe and accurate autonomous operation.

**locoEYE** addresses this challenge by using deep learning and computer vision to detect railway tracks in real time from camera input, providing an additional source of situational awareness when GPS input is unreliable.

### ✅ Advantages over Traditional Solutions

Conventional solutions often rely on physical infrastructure, such as RFID tags, inductive loops, or trackside transponders to assist with train localization. These approaches, while effective, require costly installation and maintenance, and are not scalable across all track environments.

In contrast, **locoEYE** offers key benefits:

- 🚫 **No additional infrastructure required** — operates solely on vision data from onboard cameras.
- 💰 **Cost-effective** — eliminates the need for trackside sensors or embedded tags.
- ⚙️ **Easily deployable** — works with existing trains equipped with cameras.
- 🌐 **Scalable and adaptable** — can be applied to various track layouts and environments without hardware changes.

## Introduction
**locoEYE** is a computer vision project that implements and trains a deep neural network for **real-time railway track detection**. This project adapts the [lanenet-lane-detection-pytorch](https://github.com/IrohXu/lanenet-lane-detection-pytorch.git), originally designed for lane detection in autonomous vehicles to autonomous train navigation tasks.


The project also utilizes the [Roboflow](https://roboflow.com) pipeline to build and train computer vision models
* [Roboflow 3.0 Object Detection (Fast)](https://app.roboflow.com/track-7uajr/loco-eye-version-2/models/loco-eye-version-2/2)
* [RF-DETR (Base)](https://app.roboflow.com/track-7uajr/loco-eye-version-2/models/loco-eye-version-2/1)

<div style="display: flex; flex-direction: row; gap: 20px;">
  <div style="text-align: center;">
    <img src="/media/Loco EYE demo (known tracks).png" alt="Known Tracks" title="Known Tracks Demo" width="400" height="auto">
    <p><a href="https://www.youtube.com/watch?v=tx-OoWvB8pA" target="_blank">▶️ Watch Known Tracks Demo</a></p>
  </div>
  <div style="text-align: center;">
    <img src="/media/Loco EYE demo (unknown tracks).png" alt="Unknown Tracks" title="Unknown Tracks Demo" width="400" height="auto">
    <p><a href="https://www.youtube.com/watch?v=BqSLziecSYA" target="_blank">▶️ Watch Unknown Tracks Demo</a></p>
  </div>
</div>

## Generate Rail training set/validation set/test tet
First, download cabview video here[TBD]. 

This includes both scripts and manual procedures used to create the dataset from raw video input.<br>
**Scripts**  
The following scripts are located in the tools/ folder:  
* locoEYE_extractframes.py  
Extracts 1 frame per second from a cabview video using OpenCV.  
* locoEYE_create_mask.py  
Generates binary and instance masks from labeled images.  
python .\locoEYE_create_mask.py --mode binary  
create binary mask  
python .\locoEYE_create_mask.py --mode instance  
Creates train.txt and val.txt files containing image paths, binary mask paths, and instance mask paths.  
python .\locoEYE_create_txt.py  

**Manual Steps**  
1. Labeling:  
Use an image editor (e.g., Photoshop, GIMP, or CVAT) to assign unique colors to each individual rail segment. This is required for generating accurate instance-level masks.  
2. Clean up binary mask files:  
Manually review and clean the generated binary mask files (cleaned_*.jpg) to remove noise or misclassified areas before training.

## 
**locoEYE_train.py**  
This script is used to train the railway track detection model on your labeled dataset. It supports configurable options such as batch size, image resolution, number of epochs, optimizer settings, and model architecture.

python locoEYE_train.py --dataset ./Tracks --save ./Tracks/output --height 256 --width 512 --bs 8 --lr 0.001 --epochs 20 --loss_type FocalLoss --model_type ENet

Key Features:
* Supports training with FocalLoss or CrossEntropy.
* Logs training/validation loss per epoch.
* Saves the best model checkpoint automatically to the specified --save path.

**locoEYE_test.py**  
This script is used to test a trained model on new data or validation sets. It outputs predicted binary and instance masks for each input image and saves the results to disk.

python locoEYE_test.py  --height 256 --width 512 --model .\checkpoints\best_model.pth

Key Features:
* Loads the trained .pth model and applies it to the dataset in inference mode.
* Outputs:
  Input image
  Binary segmentation prediction
  Instance segmentation prediction
*Useful for qualitative visualization of model performance.

**locoEYE_eval.py**

python locoEYE_eval.py

Key Features:
* Compares predicted masks with ground truth binary and instance masks from validation set.
* Prints and logs evaluation metrics.
* Helps in comparing different models or hyperparameter settings.

The testing result  
<div style="text-align: center;">
<img src="/media/0300_0333_inputs.gif" alt="input" title="input" title="input image" style="width: 100%; height: auto;">
<img src="/media/0300_0333_binary_outputs.gif" alt="binary_output" title="binary_output image" style="width: 100%; height: auto;">
<img src="/media/0300_0333_instance_outputs.gif" alt="instance_output" title="instance_output image" style="width: 100%; height: auto;">
</div>