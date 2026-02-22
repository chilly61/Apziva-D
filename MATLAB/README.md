# MonReader MATLAB Instructions
This folder contains MATLAB code for visualizing MonReader project data and results.
## File List
| File | Description |
|------|-------------|
| `01_data_distribution.md` | Data distribution visualization |
| `02_results_comparison.md` | Model results comparison visualization |
| `03_lstm_training.md` | LSTM training process visualization |
| `04_summary.md` | Comprehensive results summary visualization |
## Generated Visualization Charts
### 01_data_distribution.md
- `fig01_segment_distribution.png` - Segment count distribution
- `fig02_image_distribution.png` - Image count distribution
- `fig03_train_test_split.png` - Train/Test split
- `fig04_class_distribution.png` - Class distribution (Flip/NotFlip)
### 02_results_comparison.md
- `fig05_accuracy_comparison.png` - Accuracy comparison
- `fig06_f1_comparison.png` - F1 score comparison
- `fig07_combined_metrics.png` - Accuracy and F1 combined comparison
- `fig08_radar_chart.png` - Radar chart for comprehensive performance
- `fig09_feature_dimension.png` - Feature dimension comparison
### 03_lstm_training.md
- `fig10_lstm_accuracy.png` - LSTM training accuracy curve
- `fig11_lstm_loss.png` - LSTM training loss curve
- `fig12_overfitting_analysis.png` - Overfitting analysis
- `fig13_lstm_training_report.png` - LSTM training comprehensive report
### 04_summary.md
- `fig14_project_dashboard.png` - Project dashboard
- `fig15_method_ranking.png` - Method ranking
- `fig16_performance_complexity.png` - Performance vs. complexity tradeoff
## How to Run
1. Open MATLAB
2. Set current directory to `matlab` folder
3. Run the corresponding script
```matlab
% Example: Run data distribution visualization
cd('matlab')
% Copy and run the code from 01_data_distribution.md
```
## Dependencies
- MATLAB R2016b or higher
- Statistics and Machine Learning Toolbox (for some charts)
## Notes
1. LSTM training history visualization requires exporting history data from Python first
2. Generated images are saved in `outputs/` folder by default
3. Some scripts will automatically create output directories
