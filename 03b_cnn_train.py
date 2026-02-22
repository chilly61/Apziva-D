#!/usr/bin/env python3
"""
MonReader - 03b_CNN模型训练
使用CNN特征训练Random Forest
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

OUTPUT_PATH = Path("/mnt/c/Users/75346/Desktop/Apziva Project D/outputs")

print("=" * 70)
print("🧠 03b_CNN模型训练")
print("=" * 70)

# 加载特征
print("\n📂 加载CNN特征...")
data = np.load(OUTPUT_PATH / "02b_cnn_features.npz")
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

print(f"\n数据集:")
print(f"  训练集: {len(X_train)}个片段 (Flip: {sum(y_train)}, NotFlip: {len(y_train)-sum(y_train)})")
print(f"  测试集: {len(X_test)}个片段 (Flip: {sum(y_test)}, NotFlip: {len(y_test)-sum(y_test)})")
print(f"  特征维度: {X_train.shape[1]}")

# 训练
print("\n🌲 训练Random Forest...")
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

# 保存
results = {
    "feature_type": "ResNet-50 CNN特征 (平均池化)",
    "feature_dim": int(X_train.shape[1]),
    "model": "RandomForest",
    "accuracy": float(acc),
    "f1_score": float(f1),
    "n_estimators": 200,
    "max_depth": 20
}

with open(OUTPUT_PATH / "03b_train_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ 训练完成！Accuracy = {acc:.4f}, F1 = {f1:.4f}")
