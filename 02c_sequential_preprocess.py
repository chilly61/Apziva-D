#!/usr/bin/env python3
"""
MonReader - 02c_时序预处理
保存帧序列（用于LSTM），而不是池化后的特征
"""

import json
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_PATH = Path("yourpath/images")
OUTPUT_PATH = Path("yourpath/outputs")

print("=" * 70)
print("🔧 02c_时序预处理 (保存帧序列)")
print("=" * 70)

# 加载ResNet-50
print("\n📥 加载ResNet-50模型...")
from tensorflow import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = keras.Model(inputs=base_model.input, outputs=base_model.output)
print(f"✅ ResNet-50加载完成，输出维度: 2048")

def extract_cnn_features(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path).convert('RGB').resize(target_size)
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        features = feature_extractor.predict(arr, verbose=0)
        return features[0]
    except Exception as e:
        return None

def load_segment_sequences(segment_dict, label, max_frames=50):
    """加载片段的帧序列"""
    X, y, lengths = [], [], []
    total = len(segment_dict)
    
    for idx, (seg_id, images) in enumerate(segment_dict.items()):
        if (idx + 1) % 10 == 0:
            print(f"  处理: {idx+1}/{total}")
        
        features = []
        for img_path in sorted(images):
            feat = extract_cnn_features(str(img_path))
            if feat is not None:
                features.append(feat)
        
        if len(features) > 0:
            # 截断或填充到固定长度
            if len(features) > max_frames:
                features = features[:max_frames]
            else:
                # 填充
                while len(features) < max_frames:
                    features.append(np.zeros(2048))
            
            X.append(np.array(features))
            lengths.append(min(len(features), max_frames))
            y.append(1 if label == "flip" else 0)
    
    return X, y, lengths

def main():
    # 收集数据
    train_flip_segs, train_notflip_segs = defaultdict(list), defaultdict(list)
    test_flip_segs, test_notflip_segs = defaultdict(list), defaultdict(list)
    
    for label, segs_dict in [("flip", train_flip_segs), ("notflip", train_notflip_segs)]:
        folder = DATA_PATH / "training" / label
        if folder.exists():
            for f in sorted(folder.glob("*.jpg")):
                seg_id = f.name.split('_')[0]
                segs_dict[seg_id].append(f)
    
    for label, segs_dict in [("flip", test_flip_segs), ("notflip", test_notflip_segs)]:
        folder = DATA_PATH / "testing" / label
        if folder.exists():
            for f in sorted(folder.glob("*.jpg")):
                seg_id = f.name.split('_')[0]
                segs_dict[seg_id].append(f)
    
    print(f"\n📁 数据统计:")
    print(f"  Training: {len(train_flip_segs)+len(train_notflip_segs)}个片段")
    print(f"  Testing: {len(test_flip_segs)+len(test_notflip_segs)}个片段")
    
    # 提取序列
    print(f"\n📂 提取Training序列...")
    X_train_flip, y_train_flip, l_train_flip = load_segment_sequences(train_flip_segs, "flip")
    X_train_notflip, y_train_notflip, l_train_notflip = load_segment_sequences(train_notflip_segs, "notflip")
    
    print(f"\n📂 提取Testing序列...")
    X_test_flip, y_test_flip, l_test_flip = load_segment_sequences(test_flip_segs, "flip")
    X_test_notflip, y_test_notflip, l_test_notflip = load_segment_sequences(test_notflip_segs, "notflip")
    
    # 合并
    X_train = np.array(X_train_flip + X_train_notflip)
    y_train = np.array(y_train_flip + y_train_notflip)
    lengths_train = np.array(l_train_flip + l_train_notflip)
    
    X_test = np.array(X_test_flip + X_test_notflip)
    y_test = np.array(y_test_flip + y_test_notflip)
    lengths_test = np.array(l_test_flip + l_test_notflip)
    
    print(f"\n📊 序列统计:")
    print(f"  序列长度: 50帧 (不足则填充0)")
    print(f"  特征维度: 2048维/帧")
    print(f"  Training: {len(X_train)}个序列")
    print(f"  Testing: {len(X_test)}个序列")
    
    # 保存
    print(f"\n💾 保存序列数据...")
    np.savez(
        OUTPUT_PATH / "02c_sequences.npz",
        X_train=X_train,
        y_train=y_train,
        lengths_train=lengths_train,
        X_test=X_test,
        y_test=y_test,
        lengths_test=lengths_test
    )
    
    split_info = {
        "feature_extractor": "ResNet-50 (ImageNet预训练)",
        "feature_dim": 2048,
        "sequence_length": 50,
        "padding": "zero-padding",
        "split_method": "使用原始training/testing划分",
    }
    
    with open(OUTPUT_PATH / "02c_split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ 时序预处理完成！")

if __name__ == "__main__":
    main()
