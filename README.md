# Pallet-detect-segment-on-edge

## Overview

This project focuses on developing a robust, real-time pallet detection and segmentation application using ROS2, tailored for manufacturing and warehousing environments. The solution is optimized for edge devices such as the NVIDIA Jetson AGX Orin, aiming to meet the demands of mobile robotics applications in dynamic environments.

## Objective

Create a ROS2-based solution for detecting and segmenting pallets in real-time with optimized performance for edge deployment, ensuring consistent results across varying lighting and environmental conditions.

## Key Components

1. **Dataset Acquisition and Preparation**
2. **Object Detection and Semantic Segmentation**
3. **ROS2 Node Development**
4. **Edge Deployment Optimization**

## Dataset Acquisition and Preparation

### Dataset and Annotation Tools

An open-source dataset containing pallet and ground images in various scenarios was selected for this project. The annotation process involved:

- **Ground Detection**: Using Grounding DINO and the Segment Anything Model (SAM) to detect and segment the ground accurately.
- **Pallet Detection**: Leveraging two pre-trained object detection models from Roboflow due to inconsistent results from the initial models. Combined results from these models were refined with SAM for segmentation.

### Data Augmentation

To simulate real-world conditions, a range of data augmentation techniques were applied, including flips, rotations, brightness adjustments, and distortions. These augmentations increased the dataset’s diversity and robustness.

### Dataset Splitting

The dataset was split into training, validation, and test sets using an 80-10-10 ratio to ensure reliable model evaluation across various scenarios.

## Object Detection and Semantic Segmentation

### Model Development

#### Object Detection (YOLOv11)

- **Model**: YOLOv11 was selected for its efficiency in real-time object detection tasks.
- **Training**: Used custom hyperparameters, data augmentations, and early stopping to prevent overfitting and improve performance.

#### Semantic Segmentation (DeepLabV3+)

- **Model**: DeepLabV3+ with a ResNet101 backbone was used for semantic segmentation, focusing on pallets and ground areas.
- **Training**: The model was fine-tuned with our augmented dataset, leveraging pixel-wise accuracy and IoU metrics for evaluation.

### Performance Evaluation

- **Object Detection**:
  - **Metrics**: Evaluated using mAP (mean Average Precision), precision, and recall metrics.
  - **Best Validation mAP@0.50:0.95**: 0.5036.
  
  ![Object Detection Prediction Results](https://github.com/yash016/Pallet-detect-segment-on-edge/blob/main/Images/val_batch1_pred%20(1).jpg)

- **Semantic Segmentation**:
  - **Metrics**: Evaluated using Intersection over Union (IoU) and pixel-wise accuracy.
  - **Best Validation Mean IoU**: 0.7131.
  
  ![Semantic Segmentation Results](https://github.com/yash016/Pallet-detect-segment-on-edge/blob/main/Images/segmentation_result_5.png)


## ROS2 Node Development

The ROS2 package was developed to integrate object detection and segmentation functionality into a single framework.

### ROS2 Nodes

There are two primary script files provided for object detection and segmentation:

1. **`object_detection_node.py`**:
   - This script takes an input video file and performs object detection and segmentation on each frame.
   - Ideal for scenarios where pre-recorded video data is used for analysis.

2. **`object_detection_node_tensorrt.py`**:
   - This script subscribes to a live camera feed and performs real-time object detection and segmentation.
   - Optimized for edge devices with TensorRT support, leveraging hardware acceleration for improved performance.

These nodes are designed for modular integration, enabling easy adaptation in various warehouse or manufacturing environments where ROS2 is employed.

## Edge Deployment Optimization

### Model Optimization

To ensure efficient performance on edge devices, both the object detection and segmentation models were converted and optimized as follows:

1. **Quantization**: Dynamic quantization was applied to reduce model size and enhance inference speed.
2. **TensorRT Conversion**: The quantized ONNX models were converted to TensorRT engines to leverage NVIDIA’s hardware acceleration, with FP16 precision enabled for further performance gains.

### Dockerization

The entire application is encapsulated within a Docker container to ensure portability and consistency across different devices. The Docker image includes all necessary dependencies and is configured to run on NVIDIA Jetson devices with NVIDIA drivers installed, supporting optimized edge deployment.

## Build and Run the Docker Container

Follow these instructions to build and run your Docker container:

### Build the Docker Image

```bash
docker build -t ros2_jazzy_workspace .
```

## Run the Docker Container
```bash
docker run -it --name ros2_jazzy_container ros2_jazzy_workspace
```


### Resources

- [Automated Dataset Annotation with Grounding DINO and SAM](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/automated-dataset-annotation-and-evaluation-with-grounding-dino-and-sam.ipynb)
- [Train YOLOv11 on Custom Dataset](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolo11-object-detection-on-custom-dataset.ipynb)
- [Semantic Segmentation Using DeepLabV3+ PyTorch](https://www.kaggle.com/code/squidbob/semantic-segmentation-using-deeplabv3-pytorch)
- **Research Papers**:
  - **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**  
    [DeepLabV3+ Paper](https://arxiv.org/pdf/1802.02611)
  - **You Only Look Once: Unified, Real-Time Object Detection**  
    [YOLO Paper](https://arxiv.org/pdf/1506.02640)



