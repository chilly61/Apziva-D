# 02 Comparison of Results

```matlab
% MonReader - 02 模型结果对比可视化
% 展示三种方法的准确率和F1分数对比

%% 结果数据
% 从训练结果JSON中提取
methods = {'HOG + RF', 'CNN + RF', 'LSTM'};
accuracy = [0.9870, 0.9481, 0.3247];    % Accuracy values
f1_score = [0.9796, 0.9130, 0.4902];     % F1 Score values

%% 图1: 准确率对比 (Accuracy Comparison)
figure('Position', [100, 100, 600, 400]);

colors = [0.298, 0.686, 0.313; 0.204, 0.596, 0.859; 0.803, 0.145, 0.133];
bar(methods, accuracy, 0.5);
ylabel('Accuracy', 'FontSize', 12);
title('Model Accuracy Comparison', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, 1.1]);
grid on;

% 添加数值标签
for i = 1:length(accuracy)
    text(i, accuracy(i) + 0.02, sprintf('%.2f%%', accuracy(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end

% 添加颜色
for i = 1:length(methods)
    bar(i, accuracy(i), 'FaceColor', colors(i,:), 'Parent', gca);
end

saveas(gcf, 'outputs/fig05_accuracy_comparison.png');
fprintf('图1已保存: fig05_accuracy_comparison.png\n');

%% 图2: F1分数对比 (F1 Score Comparison)
figure('Position', [100, 100, 600, 400]);

bar(methods, f1_score, 0.5);
ylabel('F1 Score', 'FontSize', 12);
title('Model F1 Score Comparison', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, 1.1]);
grid on;

% 添加数值标签
for i = 1:length(f1_score)
    text(i, f1_score(i) + 0.02, sprintf('%.2f%%', f1_score(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end

for i = 1:length(methods)
    bar(i, f1_score(i), 'FaceColor', colors(i,:), 'Parent', gca);
end

saveas(gcf, 'outputs/fig06_f1_comparison.png');
fprintf('图2已保存: fig06_f1_comparison.png\n');

%% 图3: 准确率与F1分数组合对比 (Combined Metrics)
figure('Position', [100, 100, 700, 400]);

x = 1:length(methods);
width = 0.35;

bar(x - width/2, accuracy, width, 'DisplayName', 'Accuracy', 'FaceColor', [0.298, 0.686, 0.313]);
hold on;
bar(x + width/2, f1_score, width, 'DisplayName', 'F1 Score', 'FaceColor', [0.204, 0.596, 0.859]);
hold off;

ylabel('Score', 'FontSize', 12);
title('Accuracy vs F1 Score Comparison', 'FontSize', 14, 'FontWeight', 'bold');
xticks(x);
xticklabels(methods);
legend('Location', 'best');
ylim([0, 1.15]);
grid on;

% 添加数值标签
for i = 1:length(methods)
    text(i - width/2, accuracy(i) + 0.03, sprintf('%.2f', accuracy(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
    text(i + width/2, f1_score(i) + 0.03, sprintf('%.2f', f1_score(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

saveas(gcf, 'outputs/fig07_combined_metrics.png');
fprintf('图3已保存: fig07_combined_metrics.png\n');

%% 图4: 雷达图对比 (Radar Chart)
figure('Position', [100, 100, 500, 500]);

% 准备雷达图数据
categories = {'Accuracy', 'F1 Score', 'Simplicity', 'Speed'};
scores = [
    0.987, 0.980, 0.9, 0.9;   % HOG + RF (简单快速)
    0.948, 0.913, 0.7, 0.6;   % CNN + RF (中等)
    0.325, 0.490, 0.5, 0.4    % LSTM (复杂慢)
];

% 归一化到0-1范围（已经是0-1范围）
angles = linspace(0, 2*pi, length(categories)+1);
angles(end) = [];

% 绘制雷达图
colors_radar = [0.298, 0.686, 0.313; 0.204, 0.596, 0.859; 0.803, 0.145, 0.133];

for method_idx = 1:3
    polarplot(angles, scores(method_idx,:), '-o', 'LineWidth', 2, ...
        'Color', colors_radar(method_idx,:), 'MarkerSize', 8, ...
        'DisplayName', methods{method_idx});
    hold on;
end

ax = gca;
ax.ThetaTickLabel = categories;
ax.ThetaLim = [0, 2*pi];
title('Model Performance Radar Chart', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');

saveas(gcf, 'outputs/fig08_radar_chart.png');
fprintf('图4已保存: fig08_radar_chart.png\n');

%% 图5: 特征维度对比 (Feature Dimension Comparison)
figure('Position', [100, 100, 500, 400]);

feature_dims = [1872, 2048, 2048];  % HOG, CNN, LSTM (per frame)
bar(methods, feature_dims, 0.5);
ylabel('Feature Dimension', 'FontSize', 12);
title('Feature Dimension by Method', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

for i = 1:length(feature_dims)
    text(i, feature_dims(i) + 30, num2str(feature_dims(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11);
end

ylim([0, max(feature_dims) * 1.2]);
saveas(gcf, 'outputs/fig09_feature_dimension.png');
fprintf('图5已保存: fig09_feature_dimension.png\n');

fprintf('\n✅ 结果对比可视化完成！\n');
```

---
**说明：**
- 图1: 准确率柱状图对比
- 图2: F1分数柱状图对比
- 图3: 准确率与F1分数并排对比
- 图4: 雷达图综合性能对比（包含自定义的Simplicity和Speed评分）
- 图5: 特征维度对比

**注意：** 雷达图中的"Simplicity"和"Speed"是主观评分，用于综合展示各方法的特性。
