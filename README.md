# MonReader: Video Segment Classification

## Project Overview

MonReader is an intelligent video segment classification system designed to automatically identify whether a video segment contains "flipping" content or not. This binary classification task is crucial for content moderation and video analysis in various applications.

The project follows a progressive development strategy, starting from traditional HOG (Histogram of Oriented Gradients) feature extraction, then introducing deep learning-based CNN features (ResNet-50), and finally exploring sequence modeling with LSTM to capture temporal patterns in video frames.

This project was developed as part of the Apziva program, demonstrating the evolution from classical computer vision approaches to modern deep learning techniques.

Due to license, I can not upload the full size of the training/testing files. Instead, I will only use one or two sequences of pictures as example.

---

## Core Features

### 1. Multi-Approach Classification

The system implements three different approaches to solve the binary classification problem:
- **HOG + Random Forest**: Traditional computer vision with hand-crafted features
- **CNN (ResNet-50) + Random Forest**: Transfer learning with pre-trained deep features
- **LSTM Sequential Model**: Temporal sequence modeling for frame-level analysis

### 2. Data Leakage Prevention

The system includes rigorous data leakage detection to ensure training and testing sets use completely different images. By checking file names between splits, we guarantee that the model learns to generalize rather than memorize specific samples.

### 3. Flexible Feature Extraction

Multiple feature extraction methods are supported:
- HOG features capturing edge and gradient information
- Color histograms for color distribution analysis
- ResNet-50 deep features for semantic understanding
- Frame sequences for temporal pattern recognition

---

## Technical Architecture

### Dataset Summary

| Metric | Value |
|--------|-------|
| Total Segments | 194 |
| Total Images | 2,804 |
| Training Segments | 117 |
| Testing Segments | 77 |
| Flip Segments | 90 |
| NotFlip Segments | 104 |

### Data Split

The original dataset is split into training and testing sets:
- **Training Set**: 65 Flip + 52 NotFlip segments (1,162 + 1,230 images)
- **Testing Set**: 25 Flip + 52 NotFlip segments (105 + 307 images)

### Generation 1: HOG Feature Extraction

HOG (Histogram of Oriented Gradients) is a classic computer vision feature descriptor that captures edge and gradient structure in images. The algorithm divides an image into small cells, computes a histogram of gradient directions within each cell, and optionally normalizes across blocks.

For each video segment, we extract:
- HOG features capturing local shape and edge information
- Color histograms representing color distribution

These hand-crafted features are then fed into a Random Forest classifier for binary classification.

**Key Parameters:**
- Feature Dimension: 1,872
- Model: Random Forest (200 trees, max depth 20)

### Generation 2: CNN Transfer Learning

Instead of designing features manually, we leverage transfer learning using ResNet-50, a deep convolutional neural network pre-trained on ImageNet (1.4 million images, 1,000 categories).

ResNet-50 extracts high-level semantic features from each frame:
- Input: 224×224 RGB image
- Output: 2,048-dimensional feature vector per frame

For each segment, we apply **average pooling** across all frames to get a single feature vector representing the entire segment.

**Key Parameters:**
- Feature Dimension: 2,048
- Pre-trained Model: ResNet-50 (ImageNet weights)
- Classifier: Random Forest

### Generation 3: LSTM Sequential Modeling

For video understanding, temporal information matters. We preserve the frame-level sequence and use LSTM (Long Short-Term Memory) to model the temporal dynamics.

Each segment is represented as:
- Sequence Length: 50 frames (shorter sequences are zero-padded)
- Feature per Frame: 2,048 dimensions (ResNet-50 features)

The LSTM model learns to recognize patterns across frames that indicate flipping vs. non-flipping content.

**Key Parameters:**
- Sequence Length: 50
- Feature Dimension: 2,048 per frame
- LSTM Units: 128
- Dropout: 0.5 (first), 0.3 (second)
- Optimizer: Adam
- Loss: Binary Cross-Entropy

---

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Apziva-Project-D

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
Apziva-Project-D/
├── images/                    # Video frame images
│   ├── training/
│   │   ├── flip/            # Training Flip segments
│   │   └── notflip/         # Training NotFlip segments
│   └── testing/
│       ├── flip/            # Testing Flip segments
│       └── notflip/         # Testing NotFlip segments
├── outputs/                  # Generated features and models
│   └── Visualization/        # Visualization of the EDA, LSTM Curve and Result Comparison
├── matlab/                   # MATLAB visualization scripts
├── 01_monreader_eda.py       # Exploratory Data Analysis
├── 02_monreader_preprocess.py  # HOG feature extraction
├── 02b_cnn_preprocess.py     # CNN feature extraction
├── 02c_sequential_preprocess.py  # Sequential data preparation
├── 03_monreader_train.py     # HOG + RF training
├── 03b_cnn_train.py          # CNN + RF training
├── 03c_lstm_train.py         # LSTM training
└── README.md                 # This file
```

### Running the Pipeline

```bash
# Step 1: Data Exploration and Leakage Check
python 01_monreader_eda.py

# Step 2a: HOG Feature Extraction
python 02_monreader_preprocess.py

# Step 3a: Train HOG + Random Forest
python 03_monreader_train.py

# Step 2b: CNN Feature Extraction (Average Pooling)
python 02b_cnn_preprocess.py

# Step 3b: Train CNN + Random Forest
python 03b_cnn_train.py

# Step 2c: Sequential Feature Extraction (for LSTM)
python 02c_sequential_preprocess.py

# Step 3c: Train LSTM
python 03c_lstm_train.py
```

---

## Performance Evaluation

### Model Comparison

| Method | Feature Type | Feature Dimension | Accuracy | F1 Score |
|--------|-------------|-------------------|----------|----------|
| HOG + RF | HOG + Color Histogram | 1,872 | 97.40% | 95.83% |
| CNN + RF | ResNet-50 (Avg Pool) | 2,048 | 94.81% | 91.30% |
| **LSTM** | ResNet-50 Sequence | 2,048 × 50 | **98.70%** | **97.96%** |

### Key Findings

1. **LSTM Achieves Best Results**: With proper hyperparameter tuning (learning rate, dropout), LSTM can outperform traditional methods by capturing temporal patterns in video sequences. The flipping motion contains sequential information that HOG and average-pooling CNN cannot capture.

2. **HOG is Stable and Efficient**: Traditional HOG features with Random Forest achieve excellent results (97.40%) with less computational cost and more stable training on small datasets.

3. **CNN Transfer Learning Limitations**: Despite ResNet-50's powerful ImageNet pre-training, average pooling loses temporal information. The small dataset size also limits fine-tuning effectiveness.

4. **Training Stability**: LSTM training on small datasets can be unstable. Key factors include:
   - Learning rate selection (too high causes instability, too low leads to underfitting)
   - Dropout regularization to prevent overfitting
   - Early stopping to capture the best validation performance

5. **No Data Leakage**: The rigorous leakage check confirms zero overlap between training and testing sets, ensuring fair evaluation.

---

## Challenges and Solutions

### Challenge 1: Small Dataset Size

With only 194 video segments, deep learning models easily overfit. Our solutions included:
- Using transfer learning (ResNet-50 pre-trained on ImageNet)
- Applying dropout regularization in LSTM
- Careful learning rate tuning for LSTM
- Using HOG + RF as a stable baseline

### Challenge 2: LSTM Training Instability

LSTM training on small datasets can be highly unstable—accuracy may fluctuate significantly between runs. Solutions:
- Adjust learning rate (lower values often work better)
- Use dropout layers (0.5 for LSTM, 0.3 for Dense)
- Apply early stopping to capture best validation performance
- Consider using HOG + RF for more predictable results

### Challenge 3: Class Imbalance

The dataset has slightly more NotFlip (104) than Flip (90) segments. We used:
- Stratified sampling in train/test split
- Random Forest's built-in handling of class weights

### Challenge 4: Feature Selection

Choosing the right feature representation is crucial. We explored:
- Hand-crafted features (HOG, Color)
- Deep learning features (ResNet-50)
- Sequential representations (Frame sequences with LSTM)

---

## Future Improvements

1. **Data Augmentation**: Apply augmentation techniques to increase effective dataset size

2. **Advanced Sequence Models**: Try Transformer-based models (TimeSformer, Video Swin Transformer) for better temporal modeling

3. **Ensemble Methods**: Combine predictions from multiple models for robust results

4. **More Data**: Collect more video segments to enable deep learning approaches

5. **Multi-modal Features**: Add audio features if available for richer representation

---

## Tech Stack

- **Data Processing**: numpy, pandas, PIL
- **Traditional ML**: scikit-learn (Random Forest)
- **Deep Learning**: TensorFlow / Keras
- **Feature Extraction**: 
  - OpenCV (HOG)
  - TensorFlow Keras Applications (ResNet-50)
- **Sequence Models**: Keras LSTM
- **Visualization**: MATLAB (scripts provided)

---


## Contributors

Thanks to the Apziva team for their support and guidance.

---

## License

This project is for internal use only.

---
