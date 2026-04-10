#!/usr/bin/env python3
"""
MonReader - 03c_LSTM训练
使用LSTM对片段进行时序分类
"""

import json
import numpy as np
import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

OUTPUT_PATH = Path("C:\\Users\\75346\\Desktop\\Apziva Project D\\outputs")

print("=" * 70)
print("🧠 03c_LSTM模型训练")
print("=" * 70)

# 加载数据
print("\n📂 加载序列数据...")
data = np.load(OUTPUT_PATH / "02c_sequences.npz")
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
lengths_train = data['lengths_train']
lengths_test = data['lengths_test']

print(f"\n数据集:")
print(f"  训练集: {len(X_train)}个序列")
print(f"  测试集: {len(X_test)}个序列")
print(f"  序列形状: {X_train.shape} (样本, 帧数, 特征)")

# 构建LSTM模型
print("\n📐 构建LSTM模型...")
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

model = Sequential([
    LSTM(128, input_shape=(50, 2048), return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

from keras.optimizers import Adam

model.compile(
    optimizer=Adam (learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 训练
print("\n🌡️ 训练LSTM...")
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 评估
print("\n📊 测试集评估...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

from sklearn.metrics import accuracy_score, f1_score, classification_report
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n  Accuracy: {acc:.4f}")
print(f"  F1 Score: {f1:.4f}")
print("\n📋 分类报告:")
print(classification_report(y_test, y_pred, target_names=['NotFlip', 'Flip']))

# 保存
results = {
    "feature_type": "ResNet-50 CNN特征 + LSTM时序",
    "feature_dim": 2048,
    "sequence_length": 50,
    "model": "LSTM",
    "lstm_units": 128,
    "accuracy": float(acc),
    "f1_score": float(f1),
    "epochs_trained": len(history.history['accuracy']),
    "final_train_acc": float(history.history['accuracy'][-1]),
    "final_val_acc": float(history.history['val_accuracy'][-1])
}

with open(OUTPUT_PATH / "03c_lstm_results.json", "w") as f:
    json.dump(results, f, indent=2)

# 保存模型
model.save(OUTPUT_PATH / "03c_lstm_model.keras")

print(f"\n✅ LSTM训练完成！Accuracy = {acc:.4f}, F1 = {f1:.4f}")


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
plt.savefig(OUTPUT_PATH / '03c_training_curves.png', dpi=150)
plt.show()
print(f"\n📈 训练曲线已保存: {OUTPUT_PATH / '03c_training_curves.png'}")