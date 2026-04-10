#!/usr/bin/env python3
"""
MonReader - 05_MobileNet模型训练
使用MobileNetV2提取特征 + Random Forest分类
"""

import json
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image
import time

# 减少TensorFlow日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============ 配置 ============
DATA_PATH = Path("C:\\Users\\75346\\Desktop\\Apziva Project D\\images")
OUTPUT_PATH = Path("C:\\Users\\75346\\Desktop\\Apziva Project D\\outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

print("=" * 70)
print("🧠 05_MobileNet模型训练")
print("=" * 70)

# 加载MobileNetV2
print("\n📥 加载MobileNetV2模型...")
start = time.time()
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = keras.Model(inputs=base_model.input, outputs=base_model.output)
print(f"✅ MobileNetV2加载完成 ({time.time()-start:.1f}s)，输出维度: 1280")


def extract_mobilenet_features(image_path, target_size=(224, 224)):
    """使用MobileNetV2提取CNN特征"""
    try:
        img = Image.open(image_path).convert('RGB').resize(target_size)
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        features = feature_extractor.predict(arr, verbose=0)
        return features[0]
    except Exception as e:
        print(f"Error: {e}")
        return None


def load_segment_features(segment_dict, label):
    """加载片段的CNN特征 - 每个片段用多帧特征的平均"""
    X, y = [], []
    total = len(segment_dict)
    
    for idx, (seg_id, images) in enumerate(segment_dict.items()):
        if (idx + 1) % 10 == 0:
            print(f"  处理: {idx+1}/{total}")
        
        features = []
        for img_path in images:
            feat = extract_mobilenet_features(str(img_path))
            if feat is not None:
                features.append(feat)
        
        if len(features) > 0:
            # 平均池化
            avg_features = np.mean(features, axis=0)
            X.append(avg_features)
            y.append(1 if label == "flip" else 0)
    
    return X, y


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
    print(f"\n📂 提取Training MobileNet特征...")
    X_train_flip, y_train_flip = load_segment_features(train_flip_segs, "flip")
    X_train_notflip, y_train_notflip = load_segment_features(train_notflip_segs, "notflip")
    
    print(f"\n📂 提取Testing MobileNet特征...")
    X_test_flip, y_test_flip = load_segment_features(test_flip_segs, "flip")
    X_test_notflip, y_test_notflip = load_segment_features(test_notflip_segs, "notflip")
    
    # 合并
    X_train = np.array(X_train_flip + X_train_notflip)
    y_train = np.array(y_train_flip + y_train_notflip)
    X_test = np.array(X_test_flip + X_test_notflip)
    y_test = np.array(y_test_flip + y_test_notflip)
    
    print(f"\n📊 特征统计:")
    print(f"  特征维度: 1280维/片段 (平均池化)")
    print(f"  Training: {len(X_train)}个片段")
    print(f"  Testing: {len(X_test)}个片段")
    
    # 保存特征
    print(f"\n💾 保存MobileNet特征...")
    np.savez(
        OUTPUT_PATH / "05_mobilenet_features.npz",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    # 训练Random Forest
    print("\n🌲 训练Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 预测
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"\n📊 测试集结果:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("\n📋 分类报告:")
    print(classification_report(y_test, y_pred, target_names=['NotFlip', 'Flip']))
    
    # 保存结果
    results = {
        "feature_type": "MobileNetV2 (ImageNet预训练)",
        "feature_dim": 1280,
        "model": "RandomForest",
        "accuracy": float(acc),
        "f1_score": float(f1),
        "n_estimators": 200,
        "max_depth": 20
    }
    
    with open(OUTPUT_PATH / "05_train_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ MobileNet训练完成！Accuracy = {acc:.4f}, F1 = {f1:.4f}")


if __name__ == "__main__":
    main()
