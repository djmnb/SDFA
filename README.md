# SDFA: Spectral-Spatial Degradation Fusion Attention for Underwater Object Detection

SDFA is a physics-inspired attention module specifically designed to tackle optical degradation (attenuation and blurring) in complex underwater media. By constructing a **spatial-frequency dual-stream synergistic architecture**, SDFA effectively reconstructs robust feature representations from severe background noise.

## 🏗️ Network Architecture 

SDFA is a lightweight, plug-and-play unit. To maximize the use of high-level semantics to guide the reconstruction of low-level features, we recommend deploying it **after the output of the FPN (Feature Pyramid Network)**. The figure below illustrates the integration of SDFA within the FCOS framework:

![Model Architecture of FCOS with SDFA. SDFA is placed after FPN outputs to compensate for details.](img/Figure_1.png)

## 🔍 Core Motivation

Traditional attention mechanisms primarily rely on the statistical aggregation of spatial intensity. However, in underwater environments with low Signal-to-Noise Ratios (SNR), it is often difficult to distinguish between **high-intensity scattering noise** and **weak target signals**.

**SDFA addresses this challenge through the following dual-stream design:**

- **Spatial Domain**: Introduces a **Local Standard Deviation** mechanism to decouple fine texture information from smooth backgrounds, thereby precisely suppressing Caustic Interference.
- **Frequency Domain**: Utilizes **Fourier Analysis** to explicitly model the spectral characteristics of underwater media degradation. Through an adaptive recalibration mechanism, it compensates for high-frequency semantic details filtered out by the water body.

------

## ✨ Technical Highlights

- **Physics-Informed**: Designed based on the physical model of underwater light propagation, offering strong interpretability.
- **Cross-Domain Synergy**: Beautifully combines the advantages of spatial feature decoupling and frequency-domain spectral recalibration.
- **Feature Compensation**: Unlike traditional feature weighting, SDFA focuses on actively reconstructing and restoring lost semantic information.
- **Universal & Plug-and-Play**: As a lightweight unit, it can be seamlessly integrated into the Neck part of mainstream detectors (such as FCOS and the YOLO series) with negligible computational overhead (recommended after FPN outputs).

## 📈 Visual Comparisons

### Detection Results

By introducing SDFA, the detection robustness of the model in extremely turbid and low-SNR environments is significantly enhanced. (From left to right: Ground Truth, Baseline Predictions, **Baseline + SDFA Predictions**)

![检测效果对比](img/Figure_2.jpg)

### Feature Map Comparison

The feature map visualizations demonstrate that SDFA can effectively distinguish target signals from complex background noise and compensate for missing features.

![检测效果对比](img/Figure_3.jpg)

## 📊 Quantitative Results 

We conducted extensive ablation and comparative experiments across multiple mainstream architectures. The results indicate that **whether for one-stage, two-stage, or YOLO series detectors, SDFA consistently delivers stable and significant performance leaps.**

| **Detector**             | **AP**   | **AP50** | **AP75** | **APS**  | **APM**  | **APL**  |
| ------------------------ | -------- | -------- | -------- | -------- | -------- | -------- |
| **Two-stage Detectors**  |          |          |          |          |          |          |
| Faster R-CNN             | 55.5     | 76.2     | 64.0     | 54.1     | 56.5     | 54.5     |
| **w/ SDFA**              | **57.6** | **78.1** | **65.9** | **60.2** | **56.8** | **54.4** |
| Cascade R-CNN            | 56.7     | 77.3     | 64.9     | 42.2     | 57.6     | 56.6     |
| **w/ SDFA**              | **58.4** | **77.9** | **66.4** | **46.4** | **59.2** | **56.3** |
| **One-stage Detectors**  |          |          |          |          |          |          |
| RetinaNet                | 51.2     | 72.4     | 63.5     | 39.0     | 54.4     | 51.0     |
| **w/ SDFA**              | **53.0** | **73.8** | **64.4** | **42.0** | **55.2** | **51.2** |
| RepPoints                | 58.0     | 79.2     | 65.4     | 41.2     | 61.5     | 56.1     |
| **w/ SDFA**              | **59.6** | **81.0** | **66.9** | **45.2** | **63.0** | **55.5** |
| ATSS                     | 58.2     | 80.1     | 66.5     | 43.9     | 60.6     | 55.9     |
| **w/ SDFA**              | **59.9** | **81.7** | **67.8** | **45.3** | **62.2** | **55.7** |
| GFL                      | 59.1     | 81.2     | 67.0     | 47.5     | 62.4     | 57.7     |
| **w/ SDFA**              | **61.8** | **83.0** | **68.0** | **48.8** | **62.0** | **58.0** |
| **YOLO Series**          |          |          |          |          |          |          |
| YOLOv5s                  | 66.0     | 86.2     | 74.9     | 44.0     | 67.7     | 64.9     |
| **w/ SDFA**              | **67.1** | **86.8** | **75.7** | **44.9** | **68.4** | **66.5** |
| YOLOv8s                  | 67.8     | 86.0     | 75.5     | 43.5     | 69.8     | 66.9     |
| **w/ SDFA**              | **68.8** | **86.7** | **76.0** | **45.8** | **70.5** | **67.8** |
| **Underwater Detectors** |          |          |          |          |          |          |
| GCCNet                   | 61.9     | 81.7     | 69.8     | 53.2     | 64.3     | 59.6     |
| **w/ SDFA**              | **62.6** | **82.3** | **70.8** | **53.6** | **65.0** | **60.4** |
| Dynamic YOLO             | 68.2     | 86.1     | 75.2     | 52.1     | 70.0     | 66.8     |
| **w/ SDFA**              | **68.7** | **86.9** | **75.5** | **53.0** | **70.4** | **67.5** |

## 🚀 Quick Start 

```python
import torch
from sdfa import SDFA

# Assuming the input feature map from FPN has the shape [Batch, Channels, Height, Width]
x = torch.randn(2, 256, 64, 64)

# Instantiate the SDFA module
sdfa = SDFA_Module(in_channels=256)

# Enhance the features using SDFA
out = sdfa(x)

print(out.shape) # Output: [2, 256, 64, 64]
```