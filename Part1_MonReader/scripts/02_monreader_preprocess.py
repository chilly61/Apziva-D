#!/usr/bin/env python3
"""
MonReader - 02_数据预处理
使用HOG + 颜色直方图提取特征
"""

import json
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image
from skimage.feature import hog
from skimage import color

# ============ 配置 ============
# WSL路径格式
DATA_PATH = Path("C:\\Users\\75346\\Desktop\\Apziva Project D\\images")
OUTPUT_PATH = Path("C:\\Users\\75346\\Desktop\\Apziva Project D\\outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

print("=" * 70)
print("🔧 02_数据预处理 (HOG + 颜色直方图)")
print("=" * 70)


def extract_hog_features(image_path, target_size=(64, 64)):
    """提取HOG + 颜色直方图特征 (匹配之前的1872维)"""
    try:
        img = Image.open(image_path).convert('RGB').resize(target_size)
        arr = np.array(img, dtype=np.float64) / 255.0

        # HOG特征 (64x64图像, 8x8 cells, 2x2 blocks, 9 orientations)
        # 特征数: 7*7 * 4 * 9 = 1764
        gray = color.rgb2gray(arr)
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), feature_vector=True)

        # 颜色直方图 (36 bins x 3 channels = 108)
        color_hist = []
        for i in range(3):
            hist, _ = np.histogram(arr[:, :, i], bins=36, range=(0, 1))
            color_hist.extend(hist / hist.sum())

        # 合并特征: 1764 + 108 = 1872
        features = np.concatenate([hog_feat, color_hist])
        return features
    except Exception as e:
        print(f"Error: {e}")
        return None


def load_segment_features(segment_dict, label):
    """加载片段特征"""
    X, y = [], []
    total = len(segment_dict)

    for idx, (seg_id, images) in enumerate(segment_dict.items()):
        if (idx + 1) % 10 == 0:
            print(f"  处理: {idx+1}/{total}")

        features = []
        for img_path in images:
            feat = extract_hog_features(str(img_path))
            if feat is not None:
                features.append(feat)

        if len(features) > 0:
            # 平均池化
            X.append(np.mean(features, axis=0))
            y.append(1 if label == "flip" else 0)

    return np.array(X), np.array(y)


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
    print(f"\n📂 提取Training特征...")
    X_train_flip, y_train_flip = load_segment_features(train_flip_segs, "flip")
    X_train_notflip, y_train_notflip = load_segment_features(train_notflip_segs, "notflip")
    X_train = np.vstack([X_train_flip, X_train_notflip])
    y_train = np.concatenate([y_train_flip, y_train_notflip])

    print(f"\n📂 提取Testing特征...")
    X_test_flip, y_test_flip = load_segment_features(test_flip_segs, "flip")
    X_test_notflip, y_test_notflip = load_segment_features(test_notflip_segs, "notflip")
    X_test = np.vstack([X_test_flip, X_test_notflip])
    y_test = np.concatenate([y_test_flip, y_test_notflip])

    print(f"\n📊 特征统计:")
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  Training: {len(X_train)}个片段")
    print(f"  Testing: {len(X_test)}个片段")

    # 保存
    print(f"\n💾 保存特征数据...")
    np.savez(
        OUTPUT_PATH / "02_hog_features.npz",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    split_info = {
        "feature_extractor": "HOG + 颜色直方图",
        "feature_dim": int(X_train.shape[1]),
        "split_method": "使用原始training/testing划分",
        "training": {
            "flip_segments": len(X_train_flip),
            "notflip_segments": len(X_train_notflip),
            "total_segments": len(X_train)
        },
        "testing": {
            "flip_segments": len(X_test_flip),
            "notflip_segments": len(X_test_notflip),
            "total_segments": len(X_test)
        }
    }

    with open(OUTPUT_PATH / "02_split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n✅ 预处理完成！")
    print(f"  特征: {X_train.shape[1]}维 HOG + 颜色直方图")
    print(f"  保存: {OUTPUT_PATH / '02_hog_features.npz'}")


if __name__ == "__main__":
    main()
