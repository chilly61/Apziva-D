#!/usr/bin/env python3
"""
MonReader - 01_EDA（探索性数据分析）
正确区分training/testing，检查真正的数据泄露（同一文件名）
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_PATH = Path("C:\\Users\\75346\\Desktop\\Apziva Project D\\images")
OUTPUT_PATH = Path("C:\\Users\\75346\\Desktop\\Apziva Project D\\outputs")
OUTPUT_PATH.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("📊 01_EDA - 探索性数据分析")
    print("=" * 70)

    # 收集数据
    train_flip_files, train_notflip_files = set(), set()
    test_flip_files, test_notflip_files = set(), set()
    train_flip_segs, train_notflip_segs = defaultdict(list), defaultdict(list)
    test_flip_segs, test_notflip_segs = defaultdict(list), defaultdict(list)

    for split, files_set, segs_dict in [
        ("training/flip", train_flip_files, train_flip_segs),
        ("training/notflip", train_notflip_files, train_notflip_segs),
        ("testing/flip", test_flip_files, test_flip_segs),
        ("testing/notflip", test_notflip_files, test_notflip_segs),
    ]:
        folder = DATA_PATH / split
        if folder.exists():
            for f in sorted(folder.glob("*.jpg")):
                files_set.add(f.name)
                seg_id = f.name.split('_')[0]
                segs_dict[seg_id].append(f.name)

    # 统计
    def stats(segs_dict):
        return len(segs_dict), sum(len(v) for v in segs_dict.values())

    tfs, tfi = stats(train_flip_segs)
    tns, tni = stats(train_notflip_segs)
    tefs, tefi = stats(test_flip_segs)
    tens, teni = stats(test_notflip_segs)

    print(f"\n📁 TRAINING: {tfs+tns}个片段, {tfi+tni}张图片")
    print(f"  Flip: {tfs}个片段, {tfi}张")
    print(f"  NotFlip: {tns}个片段, {tni}张")

    print(f"\n📁 TESTING: {tefs+tens}个片段, {tefi+teni}张图片")
    print(f"  Flip: {tefs}个片段, {tefi}张")
    print(f"  NotFlip: {tens}个片段, {teni}张")

    print(f"\n📈 总计: {tfs+tns+tefs+tens}个片段, {tfi+tni+tefi+teni}张图片")

    # 检查数据泄露（同一文件名=同一图片）
    print(f"\n{'='*50}")
    print("⚠️ 数据泄露检查（同一文件名）")
    print(f"{'='*50}")

    flip_overlap = train_flip_files & test_flip_files
    notflip_overlap = train_notflip_files & test_notflip_files

    print(f"Flip同一图片重叠: {len(flip_overlap)}")
    print(f"NotFlip同一图片重叠: {len(notflip_overlap)}")

    if len(flip_overlap) == 0 and len(notflip_overlap) == 0:
        print("\n✅ 无数据泄露！训练集和测试集使用不同的图片。")
    else:
        print(f"\n⚠️ 发现数据泄露！")

    # 保存
    output = {
        "training": {"flip_segs": tfs, "flip_imgs": tfi, "notflip_segs": tns, "notflip_imgs": tni},
        "testing": {"flip_segs": tefs, "flip_imgs": tefi, "notflip_segs": tens, "notflip_imgs": teni},
        "total_segs": tfs+tns+tefs+tens,
        "total_imgs": tfi+tni+tefi+teni,
        "leakage": {"flip": len(flip_overlap), "notflip": len(notflip_overlap), "no_leakage": len(flip_overlap) == 0 and len(notflip_overlap) == 0}
    }

    with open(OUTPUT_PATH / "01_eda_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ 已保存: {OUTPUT_PATH / '01_eda_results.json'}")


if __name__ == "__main__":
    main()
