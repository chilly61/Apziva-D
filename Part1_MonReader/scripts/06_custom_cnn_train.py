#!/usr/bin/env python3
"""
MonReader - 06_Custom_CNN_Train
Custom Convolutional Neural Network from scratch - Segment Classification

This file implements a CNN architecture designed from scratch for classifying
video SEGMENTS (multiple frames) as flip or notflip.
"""

import json
import numpy as np
import os
from pathlib import Path
from PIL import Image
import time
from collections import defaultdict

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============ CONFIGURATION - MODIFY THESE PARAMETERS ============
# CNN Architecture Parameters
CONV_FILTERS_1 = 32       # Number of filters in first conv layer (try: 16, 32, 64)
CONV_FILTERS_2 = 64        # Number of filters in second conv layer (try: 32, 64, 128)
CONV_FILTERS_3 = 128      # Number of filters in third conv layer (try: 64, 128, 256)
KERNEL_SIZE = (3, 3)      # Convolution kernel size (try: (3,3), (5,5), (7,7))
POOL_SIZE = (2, 2)        # Max pooling window (try: (2,2), (3,3))
PADDING = 'same'          # Padding mode: 'same' (keep size) or 'valid' (shrink) (try: 'same', 'valid')
ACTIVATION = 'leaky_relu'       # Activation function (try: 'relu', 'leaky_relu', 'elu', 'tanh', 'swish')
L2_REG = 0           # L2 regularization strength (try: 0, 0.001, 0.01)

# Dense layer
DENSE_UNITS = 128         # Number of units in dense layer (try: 64, 128, 256)
DROPOUT_RATE = 0.3       # Dropout rate for regularization (try: 0.3, 0.5, 0.7)

# Training Paramete1rs
IMAGE_SIZE = (64, 64)    # Input image size for EACH FRAME (try: (32,32), (64,64), (128,128))
BATCH_SIZE = 16           # Batch size for training (try: 8, 16, 32)
EPOCHS = 50              # Number of training epochs (try: 10, 20, 30, 50)
LEARNING_RATE = 0.01    # Learning rate (try: 0.01, 0.001, 0.0001)
OPTIMIZER = 'adam'       # Optimizer type (try: 'adam', 'sgd', 'rmsprop', 'adamw')
LOSS = 'binary_crossentropy'  # Loss function (try: 'binary_crossentropy', 'focal_loss')

# SEGMENT PROCESSING - IMPORTANT!
MAX_FRAMES_PER_SEG = None  # Max frames to use per segment (try: None, 5, 10, 15, 20)
                         # None = use all frames
                         # 10 = sample 10 evenly spaced frames per segment
SEGMENT_POOLING = 'mean' # How to combine frame features: 'mean', 'max', 'last'
                         # 'mean': average all frame features (RECOMMENDED)
                         # 'max': max pooling across frames
                         # 'last': use only last frame

# Data Augmentation
USE_AUGMENTATION = True  # Use data augmentation (try: True, False)
AUGMENTATION_FACTOR = 2 # How many augmented images per original (try: 1, 2, 3)

# Callbacks
PATIENCE = 10             # Early stopping patience (try: 3, 5, 7, 10)
REDUCE_LR_FACTOR = 0.5   # Reduce LR by this factor on plateau (try: 0.5, 0.3)
REDUCE_LR_PATIENCE = 3   # Reduce LR after N epochs (try: 2, 3, 5)
MIN_LR = 1e-6           # Minimum learning rate (try: 1e-5, 1e-6)

# =================================================================

# Paths
DATA_PATH = Path("C:\\Users\\75346\\Desktop\\Apziva Project D\\images")
OUTPUT_PATH = Path("C:\\Users\\75346\\Desktop\\Apziva Project D\\outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

print("=" * 70)
print("🔧 06_Custom_CNN_Train - Segment Classification from Scratch")
print("=" * 70)

# ============ DATA LOADING ============
def load_segment_data(segment_dict, label, max_frames=None):
    """
    Load all frames from each segment.
    
    Args:
        segment_dict: {seg_id: [image_path1, image_path2, ...]}
        label: 1 for flip, 0 for notflip
        max_frames: max frames to use per segment (None = all)
    
    Returns:
        X: list of segment arrays, each is (num_frames, height, width, channels)
        y: list of labels
    """
    X, y = [], []
    
    for seg_id, image_paths in sorted(segment_dict.items()):
        # Sort by frame number
        image_paths = sorted(image_paths)
        
        # Sample frames if max_frames specified
        if max_frames is not None and len(image_paths) > max_frames:
            indices = np.linspace(0, len(image_paths)-1, max_frames, dtype=int)
            image_paths = [image_paths[i] for i in indices]
        
        # Load all frames
        frames = []
        for img_path in image_paths:
            try:
                img = Image.open(str(img_path)).convert('RGB')
                img = img.resize(IMAGE_SIZE)
                arr = np.array(img, dtype=np.float32) / 255.0
                frames.append(arr)
            except:
                continue
        
        if len(frames) > 0:
            X.append(np.array(frames))
            y.append(label)
    
    return X, y

def load_dataset():
    """Load dataset organized by segments (flip/notflip video segments)"""
    print("\n📂 Loading dataset by segments...")
    
    # Collect segments: {seg_id: [frame1.jpg, frame2.jpg, ...]}
    train_flip_segs = defaultdict(list)
    train_notflip_segs = defaultdict(list)
    test_flip_segs = defaultdict(list)
    test_notflip_segs = defaultdict(list)
    
    # Training segments
    for label, segs in [("flip", train_flip_segs), ("notflip", train_notflip_segs)]:
        folder = DATA_PATH / "training" / label
        if folder.exists():
            for f in sorted(folder.glob("*.jpg")):
                seg_id = f.name.split('_')[0]  # seg001 from seg001_001.jpg
                segs[seg_id].append(f)
    
    # Testing segments
    for label, segs in [("flip", test_flip_segs), ("notflip", test_notflip_segs)]:
        folder = DATA_PATH / "testing" / label
        if folder.exists():
            for f in sorted(folder.glob("*.jpg")):
                seg_id = f.name.split('_')[0]
                segs[seg_id].append(f)
    
    print(f"  Found {len(train_flip_segs)} flip + {len(train_notflip_segs)} notflip training segments")
    print(f"  Found {len(test_flip_segs)} flip + {len(test_notflip_segs)} notflip testing segments")
    
    # Load segments with frame sampling
    X_train_flip, y_train_flip = load_segment_data(train_flip_segs, 1, MAX_FRAMES_PER_SEG)
    X_train_notflip, y_train_notflip = load_segment_data(train_notflip_segs, 0, MAX_FRAMES_PER_SEG)
    X_test_flip, y_test_flip = load_segment_data(test_flip_segs, 1, MAX_FRAMES_PER_SEG)
    X_test_notflip, y_test_notflip = load_segment_data(test_notflip_segs, 0, MAX_FRAMES_PER_SEG)
    
    X_train = X_train_flip + X_train_notflip
    y_train = np.array(y_train_flip + y_train_notflip)
    X_test = X_test_flip + X_test_notflip
    y_test = np.array(y_test_flip + y_test_notflip)
    
    # Get max frames for padding
    max_train_frames = max(len(x) for x in X_train) if X_train else 0
    max_test_frames = max(len(x) for x in X_test) if X_test else 0
    max_frames = max(max_train_frames, max_test_frames)
    
    # Pad sequences to same length
    def pad_sequences(segments, max_len):
        padded = []
        for seg in segments:
            if len(seg) < max_len:
                # Pad with zeros
                pad = np.zeros((max_len - len(seg),) + seg.shape[1:], dtype=np.float32)
                seg = np.concatenate([seg, pad], axis=0)
            padded.append(seg)
        return np.array(padded)
    
    X_train = pad_sequences(X_train, max_frames)
    X_test = pad_sequences(X_test, max_frames)
    
    print(f"\n  Training: {len(X_train)} segments")
    print(f"    Shape: {X_train.shape} (segments, frames, H, W, C)")
    print(f"    Flip: {sum(y_train)}, NotFlip: {len(y_train)-sum(y_train)}")
    print(f"  Testing: {len(X_test)} segments")
    print(f"    Shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

# ============ CUSTOM CNN MODEL ============
def build_segment_cnn(input_shape, num_frames): 
    """
    Build Custom CNN for segment classification.
    
    Architecture:
    1. CNN backbone (process each frame) -> TimeDistributed
    2. Temporal pooling across frames (mean/max/last)
    3. Dense layers for classification
    
    Args:
        input_shape: (height, width, channels) - single frame shape
        num_frames: number of frames per segment
    
    Returns:
        model: Keras model
        
    Try Lion Model
    """
    from tensorflow import keras
    from keras import layers
    from keras.regularizers import l2
    
    # ===== CNN BACKBONE (applied to each frame) =====
    # Input: single frame
    frame_input = layers.Input(shape=input_shape, name='frame_input')
    
    # Conv Block 1
    x = layers.Conv2D(CONV_FILTERS_1, KERNEL_SIZE, padding=PADDING,
                      kernel_regularizer=l2(L2_REG), name='conv1')(frame_input)
    x = layers.Activation(ACTIVATION, name='relu1')(x)
    x = layers.MaxPooling2D(POOL_SIZE, name='pool1')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout1')(x)
    
    # Conv Block 2
    x = layers.Conv2D(CONV_FILTERS_2, KERNEL_SIZE, padding=PADDING,
                      kernel_regularizer=l2(L2_REG), name='conv2')(x)
    x = layers.Activation(ACTIVATION, name='relu2')(x)
    x = layers.MaxPooling2D(POOL_SIZE, name='pool2')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout2')(x)
    
    # Conv Block 3
    x = layers.Conv2D(CONV_FILTERS_3, KERNEL_SIZE, padding=PADDING,
                      kernel_regularizer=l2(L2_REG), name='conv3')(x)
    x = layers.Activation(ACTIVATION, name='relu3')(x)
    x = layers.MaxPooling2D(POOL_SIZE, name='pool3')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout3')(x)
    
    # Global pooling per frame
    frame_features = layers.GlobalAveragePooling2D(name='frame_features')(x)
    
    # Create CNN backbone model
    cnn_backbone = keras.Model(frame_input, frame_features, name='CNN_Backbone')
    
    # ===== SEGMENT MODEL =====
    # Input: multiple frames (segment)
    segment_input = layers.Input(shape=(num_frames,) + input_shape, name='segment_input')
    
    # Apply CNN to each frame (TimeDistributed)
    frame_features = layers.TimeDistributed(cnn_backbone, name='frame_processing')(segment_input)
    # Output: (batch, num_frames, feature_dim)
    
    # ===== TEMPORAL POOLING =====
    if SEGMENT_POOLING == 'mean':
        # Mean pooling: average features across all frames
        pooled = layers.GlobalAveragePooling1D(name='temporal_pool')(frame_features)
    elif SEGMENT_POOLING == 'max':
        # Max pooling: max features across all frames
        pooled = layers.GlobalMaxPooling1D(name='temporal_pool')(frame_features)
    elif SEGMENT_POOLING == 'last':
        # Use only the last frame
        pooled = layers.Lambda(lambda x: x[:, -1, :], name='temporal_last')(frame_features)
    else:
        # Default to mean
        pooled = layers.GlobalAveragePooling1D(name='temporal_pool')(frame_features)
    
    # ===== CLASSIFICATION HEAD =====
    x = layers.Dense(DENSE_UNITS, kernel_regularizer=l2(L2_REG), name='dense1')(pooled)
    x = layers.Activation(ACTIVATION, name='relu_dense')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout_dense')(x)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(segment_input, outputs, name='SegmentCNN')
    
    return model

# ============ TRAINING ============
def main():
    start_time = time.time()
    
    # Load data
    X_train, y_train, X_test, y_test = load_dataset()
    
    # Get shapes
    frame_shape = X_train.shape[2:]  # (H, W, C)
    num_frames = X_train.shape[1]    # number of frames per segment
    
    print(f"\n🏗️ Building Custom Segment CNN...")
    print(f"  Frame shape: {frame_shape}")
    print(f"  Frames per segment: {num_frames}")
    print(f"  Segment pooling: {SEGMENT_POOLING}")
    
    # Build model
    model = build_segment_cnn(frame_shape, num_frames)
    from tensorflow import keras
    # Compile
    if OPTIMIZER == 'adam':
        opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    elif OPTIMIZER == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    else:
        opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(optimizer=opt, loss=LOSS, metrics=['accuracy'])
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                              restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR,
                                  patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=1)
    
    # Class weights
    from sklearn.utils import class_weight
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))
    
    # Train
    print(f"\n🚀 Training...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Dropout: {DROPOUT_RATE}")
    print(f"  L2 Reg: {L2_REG}")
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print(f"\n📊 Evaluating...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    Record the best check point
    
    
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n🎯 Test Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['NotFlip', 'Flip']))
    
    # Save
    model.save(OUTPUT_PATH / "06_custom_cnn_model.keras")
    
    results = {
        "model_type": "Custom CNN (from scratch) - Segment Classification",
        "architecture": {
            "conv_filters": [CONV_FILTERS_1, CONV_FILTERS_2, CONV_FILTERS_3],
            "kernel_size": KERNEL_SIZE,
            "pool_size": POOL_SIZE,
            "activation": ACTIVATION,
            "dense_units": DENSE_UNITS,
            "dropout": DROPOUT_RATE,
            "l2_reg": L2_REG,
            "segment_pooling": SEGMENT_POOLING
        },
        "training_params": {
            "image_size": IMAGE_SIZE,
            "max_frames_per_segment": MAX_FRAMES_PER_SEG,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER,
            "loss": LOSS,
            "use_augmentation": USE_AUGMENTATION
        },
        "accuracy": float(acc),
        "f1_score": float(f1),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "training_time_seconds": time.time() - start_time
    }
    
    optimizer : lion
    
    with open(OUTPUT_PATH / "06_train_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    import matplotlib.pyplot as plt

    # 绘制训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(20, 4))

    # 准确率曲线
    axes[0].plot(history.history['accuracy'], label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss曲线
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '06_training_curves.png', dpi=150)
    plt.show()
        
        
    
    print(f"\n✅ Training complete! Time: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    main()


Photos from databases  own images, grab a book stay still. texts.
Eazy OCR; paddle OCR - industry usage;
baseline: 
    
AI section:
1. full ownership of models: make changes to the model if you want to. Updated open source models (fine tune the model)
2. privacy, RAG, a private model/network, permissioned network. 
    cryptophy; patient data (anonmous data); data leakage problem
3. if non-sensitive data, apis from open source models;