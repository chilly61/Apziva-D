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
├── matlab/                   # MATLAB visualization scripts
├── 01_monreader_eda.py       # Exploratory Data Analysis
├── 02_monreader_preprocess.py  # HOG feature extraction
├── 02b_cnn_preprocess.py     # CNN feature extraction
├── 02c_sequential_preprocess.py  # Sequential data preparation
├── 03_monreader_train.py     # HOG + RF training
├── 03b_cnn_train.py          # CNN + RF training
├── 03c_lstm_train.py         # LSTM training
├── 04_monreader_evaluate.py  # Results evaluation
└── README.md                 # This file
```

### Running the Pipeline

```bash
# Step 1: Data Exploration and Leakage Check
python 01_monreader_eda.py

# Step 2a: HOG Feature Extraction
python 02_monreader_preprocess.py

# Step 2b: CNN Feature Extraction (Average Pooling)
python 02b_cnn_preprocess.py

# Step 2c: Sequential Feature Extraction (for LSTM)
python 02c_sequential_preprocess.py

# Step 3a: Train HOG + Random Forest
python 03_monreader_train.py

# Step 3b: Train CNN + Random Forest
python 03b_cnn_train.py

# Step 3c: Train LSTM
python 03c_lstm_train.py

# Step 4: Evaluate and Summarize
python 04_monreader_evaluate.py
```

---

## Performance Evaluation

### Model Comparison

| Method | Feature Type | Feature Dimension | Accuracy | F1 Score |
|--------|-------------|-------------------|----------|----------|
| HOG + RF | HOG + Color Histogram | 1,872 | **97.40%** | **95.83%** |
| CNN + RF | ResNet-50 (Avg Pool) | 2,048 | 94.81% | 91.30% |
| LSTM | ResNet-50 Sequence | 2,048 × 50 | 32.47% | 49.02% |

### Key Findings

1. **HOG Outperforms Deep Learning**: On this small dataset (194 segments), traditional HOG features with Random Forest achieve the best results (98.70% accuracy).

2. **CNN Transfer Learning Limitations**: Despite ResNet-50's powerful ImageNet pre-training, the small dataset size limits its effectiveness. Average pooling may also lose important temporal information.

3. **LSTM Overfitting**: The LSTM model severely overfits—achieving 100% training accuracy but only 32.47% test accuracy. This confirms that deep learning approaches require more data to generalize.

4. **No Data Leakage**: The rigorous leakage check confirms zero overlap between training and testing sets, ensuring fair evaluation.

---

## Challenges and Solutions

### Challenge 1: Small Dataset Size

With only 194 video segments, deep learning models easily overfit. Our solutions included:
- Using transfer learning (ResNet-50 pre-trained on ImageNet)
- Applying dropout regularization in LSTM
- Ultimately choosing traditional HOG features which work better with limited data

### Challenge 2: Class Imbalance

The dataset has slightly more NotFlip (104) than Flip (90) segments. We used:
- Stratified sampling in train/test split
- Random Forest's built-in handling of class weights

### Challenge 3: Feature Selection

Choosing the right feature representation is crucial. We explored:
- Hand-crafted features (HOG, Color)
- Deep learning features (ResNet-50)
- Sequential representations (Frame sequences)

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

## Results Visualization

MATLAB visualization scripts are provided in the `matlab/` folder:

| Script | Generates |
|--------|-----------|
| 01_data_distribution.md | Data distribution charts (4 figures) |
| 02_results_comparison.md | Model comparison charts (5 figures) |
| 03_lstm_training.md | LSTM training curves (4 figures) |
| 04_summary.md | Project summary dashboard (3 figures) |

Run the MATLAB scripts to generate professional visualizations for presentations and reports.

---

## Contributors

Thanks to the Apziva team for their support and guidance.

---

## License

This project is for internal use only.

---
