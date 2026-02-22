#!/usr/bin/env python3
"""
MonReader - 02_数据预处理
使用CNN (ResNet-50) 提取每帧特征
"""

import json
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image

# 减少TensorFlow日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============ 配置 ============
DATA_PATH = Path("/mnt/c/Users/75346/Desktop/Apziva Project D/images")
OUTPUT_PATH = Path("/mnt/c/Users/75346/Desktop/Apziva Project D/outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

print("=" * 70)
print("🔧 02_数据预处理 (ResNet-50特征提取)")
print("=" * 70)

# 加载ResNet-50
print("\n📥 加载ResNet-50模型...")
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = keras.Model(inputs=base_model.input, outputs=base_model.output)
print(f"✅ ResNet-50加载完成，输出维度: 2048")

def extract_cnn_features(image_path, target_size=(224, 224)):
    """使用ResNet-50提取CNN特征"""
    try:
        img = Image.open(image_path).convert('RGB').resize(target_size)
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        features = feature_extractor.predict(arr, verbose=0)
        return features[0]
    except Exception as e:
        return None

def load_segment_cnn_features(segment_dict, label):
    """加载片段的CNN特征"""
    X, y, lengths = [], [], []
    total = len(segment_dict)
    
    for idx, (seg_id, images) in enumerate(segment_dict.items()):
        if (idx + 1) % 10 == 0:
            print(f"  处理: {idx+1}/{total}")
        
        features = []
        for img_path in images:
            feat = extract_cnn_features(str(img_path))
            if feat is not None:
                features.append(feat)
        
        if len(features) > 0:
            X.append(np.array(features))
            lengths.append(len(features))
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
    
    # 提取特征
    print(f"\n📂 提取Training CNN特征...")
    X_train_flip, y_train_flip, l_train_flip = load_segment_cnn_features(train_flip_segs, "flip")
    X_train_notflip, y_train_notflip, l_train_notflip = load_segment_cnn_features(train_notflip_segs, "notflip")
    
    print(f"\n📂 提取Testing CNN特征...")
    X_test_flip, y_test_flip, l_test_flip = load_segment_cnn_features(test_flip_segs, "flip")
    X_test_notflip, y_test_notflip, l_test_notflip = load_segment_cnn_features(test_notflip_segs, "notflip")
    
    train_seg = len(X_train_flip) + len(X_train_notflip)
    test_seg = len(X_test_flip) + len(X_test_notflip)
    
    print(f"\n📊 特征统计:")
    print(f"  特征维度: 2048维/帧")
    print(f"  Training: {train_seg}个片段")
    print(f"  Testing: {test_seg}个片段")
    
    # 保存
    print(f"\n💾 保存特征数据...")
    np.savez(
        OUTPUT_PATH / "02_cnn_features.npz",
        X_train_flip=X_train_flip,
        y_train_flip=np.array(y_train_flip),
        X_train_notflip=X_train_notflip,
        y_train_notflip=np.array(y_train_notflip),
        X_test_flip=X_test_flip,
        y_test_flip=np.array(y_test_flip),
        X_test_notflip=X_test_notflip,
        y_test_notflip=np.array(y_test_notflip)
    )
    
    split_info = {
        "feature_extractor": "ResNet-50 (ImageNet预训练)",
        "feature_dim": 2048,
        "split_method": "使用原始training/testing划分",
        "training": {
            "flip_segments": len(X_train_flip),
            "notflip_segments": len(X_train_notflip),
            "total_segments": train_seg
        },
        "testing": {
            "flip_segments": len(X_test_flip),
            "notflip_segments": len(X_test_notflip),
            "total_segments": test_seg
        }
    }
    
    with open(OUTPUT_PATH / "02_split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ 预处理完成！")
    print(f"  特征: 2048维 ResNet-50特征/帧")
    print(f"  保存: {OUTPUT_PATH / '02_cnn_features.npz'}")

if __name__ == "__main__":
    main()
