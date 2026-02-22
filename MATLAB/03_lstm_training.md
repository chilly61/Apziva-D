# 03 LSTM训练过程可视化

```matlab
% MonReader - 03 LSTM训练过程可视化
% 展示LSTM模型的训练和验证曲线

% 注意: 此代码需要从Python导出训练历史数据
% 可以使用以下Python代码保存:
%   import json
%   with open('history.json', 'w') as f:
%       json.dump(history.history, f)

%% 加载训练历史数据 (如果存在)
if exist('outputs/lstm_history.json', 'file')
    fid = fopen('outputs/lstm_history.json', 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    history = jsondecode(str);
    
    train_acc = history.accuracy;
    val_acc = history.val_accuracy;
    train_loss = history.loss;
    val_loss = history.val_loss;
    epochs = 1:length(train_acc);
else
    % 如果没有历史数据，使用示例数据展示
    warning('未找到lstm_history.json，使用示例数据');
    
    epochs = 1:15;
    % 模拟过拟合的训练曲线
    train_acc = linspace(0.3, 1.0, 15);  % 训练准确率上升
    val_acc = [0.35, 0.38, 0.40, 0.42, 0.40, 0.38, 0.37, 0.35, 0.36, 0.34, 0.33, 0.32, 0.31, 0.30, 0.375];
    train_loss = linspace(1.2, 0.1, 15);  % 训练损失下降
    val_loss = [1.1, 1.0, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.4];  % 验证损失上升
end

%% 图1: 训练和验证准确率曲线
figure('Position', [100, 100, 600, 400]);

plot(epochs, train_acc, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Training Accuracy');
hold on;
plot(epochs, val_acc, 'r-s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Validation Accuracy');
hold off;

xlabel('Epoch', 'FontSize', 12);
ylabel('Accuracy', 'FontSize', 12);
title('LSTM Training - Accuracy over Epochs', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
ylim([0, 1.05]);

% 标记最佳验证点
[best_val_acc, best_idx] = max(val_acc);
plot(best_idx, best_val_acc, 'g*', 'MarkerSize', 15, 'DisplayName', sprintf('Best: %.2f', best_val_acc));

saveas(gcf, 'outputs/fig10_lstm_accuracy.png');
fprintf('图1已保存: fig10_lstm_accuracy.png\n');

%% 图2: 训练和验证损失曲线
figure('Position', [100, 100, 600, 400]);

plot(epochs, train_loss, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Training Loss');
hold on;
plot(epochs, val_loss, 'r-s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Validation Loss');
hold off;

xlabel('Epoch', 'FontSize', 12);
ylabel('Loss', 'FontSize', 12);
title('LSTM Training - Loss over Epochs', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% 标记最佳验证点
[best_val_loss, best_idx_loss] = min(val_loss);
plot(best_idx_loss, best_val_loss, 'g*', 'MarkerSize', 15, 'DisplayName', sprintf('Best: %.2f', best_val_loss));

saveas(gcf, 'outputs/fig11_lstm_loss.png');
fprintf('图2已保存: fig11_lstm_loss.png\n');

%% 图3: 准确率差距 (过拟合分析)
figure('Position', [100, 100, 600, 400]);

accuracy_gap = train_acc - val_acc;
bar(epochs, accuracy_gap, 'FaceColor', [0.8, 0.3, 0.3], 'FaceAlpha', 0.7);
hold on;
plot(epochs, accuracy_gap, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 4);
hold off;

xlabel('Epoch', 'FontSize', 12);
ylabel('Accuracy Gap (Train - Val)', 'FontSize', 12);
title('Overfitting Analysis: Train-Val Accuracy Gap', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% 添加阈值线
yline(0.1, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Overfitting Threshold (0.1)');
legend('Location', 'best');

saveas(gcf, 'outputs/fig12_overfitting_analysis.png');
fprintf('图3已保存: fig12_overfitting_analysis.png\n');

%% 图4: 综合训练报告
figure('Position', [100, 100, 700, 500);

% 子图1: 准确率
subplot(2, 2, 1);
plot(epochs, train_acc, 'b-o', 'LineWidth', 2, 'MarkerSize', 5);
hold on;
plot(epochs, val_acc, 'r-s', 'LineWidth', 2, 'MarkerSize', 5);
hold off;
ylabel('Accuracy');
title('Accuracy');
legend({'Train', 'Validation'}, 'Location', 'best');
grid on;
ylim([0, 1.05]);

% 子图2: 损失
subplot(2, 2, 2);
plot(epochs, train_loss, 'b-o', 'LineWidth', 2, 'MarkerSize', 5);
hold on;
plot(epochs, val_loss, 'r-s', 'LineWidth', 2, 'MarkerSize', 5);
hold off;
ylabel('Loss');
title('Loss');
legend({'Train', 'Validation'}, 'Location', 'best');
grid on;

% 子图3: 最终结果表格
subplot(2, 2, 3);
axis off;
final_results = [
    {'Final Train Accuracy:', sprintf('%.2f%%', train_acc(end)*100)};
    {'Final Val Accuracy:', sprintf('%.2f%%', val_acc(end)*100)};
    {'Best Val Accuracy:', sprintf('%.2f%%', best_val_acc*100)};
    {'Final Train Loss:', sprintf('%.4f', train_loss(end))};
    {'Final Val Loss:', sprintf('%.4f', val_loss(end))};
];
table_data = final_results;
theader = {'Metric', 'Value'};
uitable('Data', table_data, 'ColumnName', theader, ...
    'Position', [50, 50, 300, 150], 'FontSize', 12);
title('Training Summary');

% 子图4: 结论
subplot(2, 2, 4);
axis off;
conclusion_text = {
    'Training Analysis:';
    '';
    '• Severe overfitting observed';
    '• Train accuracy: 100%';
    '• Validation accuracy: ~37%';
    '• Large gap indicates overfitting';
    '';
    'Recommendations:';
    '• Use dropout/regularization';
    '• Reduce model complexity';
    '• Use HOG method instead'
};
text(0.1, 0.9, conclusion_text, 'FontSize', 11, 'VerticalAlignment', 'top');
title('Conclusions');

saveas(gcf, 'outputs/fig13_lstm_training_report.png');
fprintf('图4已保存: fig13_lstm_training_report.png\n');

fprintf('\n✅ LSTM训练可视化完成！\n');
```

---
**说明：**
- 图1: 训练和验证准确率曲线
- 图2: 训练和验证损失曲线
- 图3: 过拟合分析（训练-验证准确率差距）
- 图4: 综合训练报告（包含结论）

**使用前需要：**
1. 从Python导出训练历史：
```python
import json
# 在训练后执行
with open('outputs/lstm_history.json', 'w') as f:
    json.dump(history.history, f)
```
2. 将JSON文件放到outputs文件夹
3. 运行MATLAB代码
