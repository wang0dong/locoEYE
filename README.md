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


