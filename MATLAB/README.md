# MonReader MATLAB 可视化脚本说明

本文件夹包含用于可视化 MonReader 项目数据和结果的 MATLAB 代码。

## 文件列表

| 文件 | 说明 |
|------|------|
| `01_data_distribution.md` | 数据分布可视化 |
| `02_results_comparison.md` | 模型结果对比可视化 |
| `03_lstm_training.md` | LSTM训练过程可视化 |
| `04_summary.md` | 综合结果汇总可视化 |

## 生成的可视化图表

### 01_data_distribution.md
- `fig01_segment_distribution.png` - 片段数量分布
- `fig02_image_distribution.png` - 图片数量分布
- `fig03_train_test_split.png` - 训练/测试集划分
- `fig04_class_distribution.png` - 类别分布（Flip/NotFlip）

### 02_results_comparison.md
- `fig05_accuracy_comparison.png` - 准确率对比
- `fig06_f1_comparison.png` - F1分数对比
- `fig07_combined_metrics.png` - 准确率与F1组合对比
- `fig08_radar_chart.png` - 雷达图综合性能
- `fig09_feature_dimension.png` - 特征维度对比

### 03_lstm_training.md
- `fig10_lstm_accuracy.png` - LSTM训练准确率曲线
- `fig11_lstm_loss.png` - LSTM训练损失曲线
- `fig12_overfitting_analysis.png` - 过拟合分析
- `fig13_lstm_training_report.png` - LSTM训练综合报告

### 04_summary.md
- `fig14_project_dashboard.png` - 项目仪表板
- `fig15_method_ranking.png` - 方法排名
- `fig16_performance_complexity.png` - 性能与复杂度权衡

## 运行方法

1. 打开 MATLAB
2. 将当前目录设置为 `matlab` 文件夹
3. 运行相应的脚本

```matlab
% 示例：运行数据分布可视化
cd('matlab')
% 复制并运行 01_data_distribution.md 中的代码
```

## 依赖

- MATLAB R2016b 或更高版本
- Statistics and Machine Learning Toolbox（用于部分图表）

## 注意事项

1. LSTM 训练历史可视化需要先从 Python 导出历史数据
2. 生成的图片默认保存在 `outputs/` 文件夹
3. 部分脚本会自动创建输出目录
