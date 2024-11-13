# QC-PICSL

**QC-PICSL** is a collection of publishable machine learning tools developed by me at the [PICSL (Penn Image Computing and Science Lab)]([http://www.cis.upenn.edu/~pennimage/](https://picsl.upenn.edu/)) for automating and enhancing the quality control (QC) process in medical image analysis, particularly for Alzheimer’s disease segmentation. These tools leverage state-of-the-art deep learning models and workflows to assist researchers and practitioners in processing and evaluating medical imaging data efficiently.

## Features

- **2D and 3D Training Pipelines:**
  - Includes scripts for training models on 2D slices (`2dtraining.py`) and 3D volumes (`3dtraining.py`).
  
- **Data Preparation:**
  - Tools such as `DataPrepare.py` and `build_dataset.py` for preprocessing and dataset creation.

- **Accuracy Assessment:**
  - Comprehensive evaluation scripts (`3dAccuracy.py`, `modelTesting.py`) for testing and validating model performance.

- **Graphical User Interface (GUI):**
  - An interactive GUI (`gui.py`) for user-friendly interaction with the QC tools.

- **ResNet-based Model:**
  - Implements transfer learning using ResNet-18 for high-accuracy image segmentation (`resnet18TL3Dfunction.mlx`).

- **Utility Functions:**
  - Additional utilities (`utilities.py`) for data augmentation, logging, and helper functions.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AdamEGorka/QC-PICSL.git
   cd QC-PICSL
## Note
Repository does not include segmentation data, which is required to run.
