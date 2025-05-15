# 🌍 Post-Disaster Critical Area Identification Using Image Processing

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## 📋 Overview

This project implements a novel Siamese U-Net architecture with Alpha Blending for automatic detection and classification of building damage from satellite imagery after natural disasters. By analyzing pre- and post-disaster images through a dual-path neural network with shared weights, the system can accurately identify affected buildings and classify damage severity, providing critical intelligence for disaster response teams.
<p align="center">
  <img src="sample_images/disaster_detection.png" alt="Disaster Detection Example" width="700"/>
</p>

## 🌟 Key Features

- Advanced Siamese architecture with shared encoder for pre/post-disaster image comparison
- Innovative alpha blending mechanism for adaptive feature fusion
- Multi-class damage severity classification (background, no damage, minor, major, destroyed)
- State-of-the-art performance in mixed damage scenarios
- Optimized training pipeline with progressive resizing and mixed precision

## 📊 Dataset

The project leverages the **xView2 dataset**, developed by the Defense Innovation Unit and Carnegie Mellon University, containing:

- Pre-disaster images (baseline state)
- Post-disaster images (showing damage)
- Labeled geoJSON files with building footprints and damage classification
- Coverage of multiple disaster types: earthquakes, hurricanes, floods, etc.
- High-resolution satellite imagery from global disaster events

## 🔧 Data Processing Pipeline

### Class Balancing

The dataset exhibited significant class imbalance with most samples showing undamaged buildings. Our solutions included:

- **Strategic augmentation** of underrepresented damage classes
- **Selective cleaning** of non-informative samples (< 0.25% building coverage)
- **Class weighting** in the loss function to handle imbalance
- **CutMix** implementation to enhance learning of rare classes

### Advanced Augmentation Techniques

Data augmentation applied with 50% probability using:
- Rotation (random angles)
- Horizontal/Vertical flipping
- Zoom in/Zoom out transformations
- Mixed augmentation strategies

### Specialized Training Optimizations

- Progressive Resizing: Training begins with smaller 256×256 images and gradually increases to 512×512
- Mixed Precision Training: Using FP16 calculations for faster training
- Gradient Accumulation: Effective batch size scaling for better convergence
- Combined Loss Function: Weighted Cross-Entropy + Dice Loss for balanced multi-class performance
- Checkpoint Management: Robust save/resume system with early stopping


## 🧠 Model Implemented

### Siamese U-Net with Alpha Blending 
The core innovation of this project is the Siamese U-Net with Alpha Blending architecture, designed and implemented by Nihal Choutapelly with assistance from Sujit Jaiswal. Unlike traditional single-branch networks, this approach processes pre and post-disaster images simultaneously through twin networks with shared weights.

Architecture Design
The architecture consists of:

- Shared Encoder: A single encoder with shared weights processes both pre- and post-disaster images, ensuring consistent feature extraction.
- Alpha Blending Module: Instead of simple concatenation, adaptive alpha blending modules fuse features from pre and post-disaster pathways:
  blended = α × post_features + (1 - α) × pre_features.
   where α is a learnable parameter that determines the importance of each feature set.
- Decoder with Skip Connections: A U-Net style decoder with skip connections from blended encoder features to maintain spatial resolution.

## 🛠️ Implementation Details

- **Framework**: PyTorch
- **Training Optimizations**:
  - Progressive resizing strategy (256×256 → 512×512)
  - Mixed precision training (FP16)
  - Gradient accumulation for effective batch size increase
  - Combined loss function (Cross-entropy + Dice)
  - Early stopping with checkpoint management


## 📈 Comparative Evaluation & Results
After extensive comparison with other architectures (ResNet-UNet and BDANet), the Siamese U-Net with Alpha Blending demonstrated superior performance, particularly in real-world mixed-damage scenarios:

| Criterion | ResNet-UNet | BDANet | Siamese U-Net |
|:----------|:------------|:-------|:--------------|
| Multi-class Prediction | Poor (Red dominant) | Unstable (Class biased) | **Good (Balanced)** |
| Mixed Severities Performance | Weak | Inconsistent | **Strong** |
| Class Bias | High (Red) | High (Varies) | **Low (Except green)** |
| Practical Utility | Limited | Limited | **High** |

The Siamese U-Net consistently outperformed other models in maintaining balanced predictions across all damage classes, even when they coexist in the same image - a critical capability for real-world disaster response.

Models were rigorously evaluated using multiple metrics:

- **Dice coefficient**: Measuring overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Quantifying segmentation quality
- **Class-wise metrics**: Ensuring balanced performance across damage categories
- **Real-world practical utility**: Performance on mixed-damage scenarios

## Final Dice score obtained : 0.623
## Final IOU coefffiecient : 0.592
<p align="center">
  <img src="[sample_images/disaster_detection.png](https://github.com/user-attachments/assets/8f36e601-84f0-45b4-8f3a-1955c1f8999e)" alt="Disaster Detection Example" width="700"/>

</p>
