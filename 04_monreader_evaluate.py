#!/usr/bin/env python3
"""
MonReader - 04_评估与结果汇总
"""

import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# ============ 配置 ============
OUTPUT_PATH = Path("/mnt/c/Users/75346/Desktop/Apziva Project D/outputs")

def main():
    print("=" * 70)
    print("📊 04_评估与结果汇总")
    print("=" * 70)
    
    # 读取训练结果
    with open(OUTPUT_PATH / "03_train_results.json", "r") as f:
        train_results = json.load(f)
    
    # 读取划分信息
    with open(OUTPUT_PATH / "02_split_info.json", "r") as f:
        split_info = json.load(f)
    
    # 读取EDA结果
    with open(OUTPUT_PATH / "01_eda_results.json", "r") as f:
        eda_results = json.load(f)
    
    # 汇总结果
    summary = {
        "project": "MonReader - 视频片段分类",
        "date": "2026-02-20",
        
        "dataset": {
            "total_segments": eda_results["flip_segments"] + eda_results["notflip_segments"],
            "total_images": eda_results["total_images"],
            "flip_segments": eda_results["flip_segments"],
            "notflip_segments": eda_results["notflip_segments"],
            "train_segments": split_info["total"]["train_segments"],
            "test_segments": split_info["total"]["test_segments"],
            "train_images": split_info["total"]["train_images"],
            "test_images": split_info["total"]["test_images"]
        },
        
        "data_leakage_check": split_info["data_leakage_check"],
        
        "model": train_results["model"],
        
        "final_results": {
            "accuracy": train_results["results"]["accuracy"],
            "f1_score": train_results["results"]["f1_score"]
        },
        
        "method": train_results["method"],
        
        "conclusion": "使用完整片段特征进行分类，按片段ID划分确保无数据泄露"
    }
    
    # 打印汇总
    print(f"\n📋 项目汇总:")
    print(f"  项目: {summary['project']}")
    print(f"  日期: {summary['date']}")
    
    print(f"\n📁 数据集:")
    print(f"  总片段数: {summary['dataset']['total_segments']}")
    print(f"  总图片数: {summary['dataset']['total_images']}")
    print(f"  训练片段: {summary['dataset']['train_segments']}")
    print(f"  测试片段: {summary['dataset']['test_segments']}")
    
    print(f"\n⚠️ 数据泄露检查:")
    print(f"  Flip重叠: {summary['data_leakage_check']['flip_overlap']}")
    print(f"  NotFlip重叠: {summary['data_leakage_check']['notflip_overlap']}")
    print(f"  无泄露: {summary['data_leakage_check']['no_leakage']}")
    
    print(f"\n🧠 模型:")
    print(f"  类型: {summary['model']['type']}")
    print(f"  参数: n_estimators={summary['model']['n_estimators']}, max_depth={summary['model']['max_depth']}")
    
    print(f"\n📊 最终结果:")
    print(f"  Accuracy: {summary['final_results']['accuracy']:.4f}")
    print(f"  F1 Score: {summary['final_results']['f1_score']:.4f}")
    
    print(f"\n💡 结论:")
    print(f"  {summary['conclusion']}")
    
    # 保存汇总
    with open(OUTPUT_PATH / "04_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ 汇总完成！结果已保存到: {OUTPUT_PATH / '04_summary.json'}")
    
    return summary

if __name__ == "__main__":
    main()
