# 04 综合结果汇总可视化

```matlab
% MonReader - 04 综合结果汇总
% 创建一个完整的项目总结图

%% 项目信息
project_name = 'MonReader - Video Segment Classification';
methods = {'HOG + RF', 'CNN + RF', 'LSTM'};
accuracy = [0.9870, 0.9481, 0.3247];
f1_score = [0.9796, 0.9130, 0.4902];

% 数据集信息
total_segments = 194;
total_images = 2804;
train_segments = 117;
test_segments = 77;
leakage_check = 'No Data Leakage';

%% 图1: 项目概述仪表板 (Project Overview Dashboard)
figure('Position', [50, 50, 900, 600]);
clf;

% 标题
annotation('textbox', [0.3, 0.92, 0.4, 0.05], 'String', project_name, ...
    'FontSize', 18, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'LineStyle', 'none');

% 1. 数据集统计 (左上)
subplot(2, 3, 1);
pie([train_segments, test_segments], {'Train', 'Test'});
title('Data Split', 'FontSize', 12, 'FontWeight', 'bold');

% 2. 类别分布 (右上)
subplot(2, 3, 2);
flip_segs = 65 + 25;
notflip_segs = 52 + 52;
pie([flip_segs, notflip_segs], {'Flip', 'NotFlip'});
title('Class Distribution', 'FontSize', 12, 'FontWeight', 'bold');

% 3. 准确率对比 (中左)
subplot(2, 3, 3);
colors = [0.298, 0.686, 0.313; 0.204, 0.596, 0.859; 0.803, 0.145, 0.133];
bar(accuracy, 'FaceColor', [0.4, 0.6, 0.8]);
set(gca, 'XTickLabel', methods);
ylabel('Accuracy');
title('Accuracy Comparison', 'FontSize', 12, 'FontWeight', 'bold');
ylim([0, 1.1]);
grid on;
for i = 1:length(accuracy)
    text(i, accuracy(i)+0.03, sprintf('%.1f%%', accuracy(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

% 4. F1分数对比 (中右)
subplot(2, 3, 4);
bar(f1_score, 'FaceColor', [0.6, 0.4, 0.8]);
set(gca, 'XTickLabel', methods);
ylabel('F1 Score');
title('F1 Score Comparison', 'FontSize', 12, 'FontWeight', 'bold');
ylim([0, 1.1]);
grid on;
for i = 1:length(f1_score)
    text(i, f1_score(i)+0.03, sprintf('%.1f%%', f1_score(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

% 5. 关键指标 (下左)
subplot(2, 3, 5);
axis off;
stats_text = {
    '📊 Dataset Statistics:'
    sprintf('  • Total Segments: %d', total_segments)
    sprintf('  • Total Images: %d', total_images)
    sprintf('  • Training: %d segments', train_segments)
    sprintf('  • Testing: %d segments', test_segments)
    sprintf('  • Data Leakage: %s', leakage_check)
};
text(0.1, 0.9, stats_text, 'FontSize', 11, 'VerticalAlignment', 'top');

% 6. 结论 (下右)
subplot(2, 3, 6);
axis off;
conclusion = {
    '💡 Key Findings:'
    '  • HOG + RF achieves best result'
    '  • 98.7% Accuracy, 98.0% F1'
    '  • CNN overfits on small dataset'
    '  • LSTM shows severe overfitting'
    ''
    '✓ Recommended: HOG + RF'
};
text(0.1, 0.9, conclusion, 'FontSize', 11, 'VerticalAlignment', 'top', ...
    'Color', [0.1, 0.1, 0.1]);

saveas(gcf, 'outputs/fig14_project_dashboard.png');
fprintf('图1已保存: fig14_project_dashboard.png\n');

%% 图2: 方法排名可视化 (Method Ranking)
figure('Position', [100, 100, 600, 500]);

% 计算综合得分 (准确率 * 0.5 + F1 * 0.5)
composite_score = (accuracy + f1_score) / 2;

% 排序
[sorted_scores, idx] = sort(composite_score, 'descend');
sorted_methods = methods(idx);
sorted_accuracy = accuracy(idx);
sorted_f1 = f1_score(idx);

% 绘制水平条形图
y_pos = 1:length(sorted_methods);
barh(y_pos, sorted_scores, 0.6);
set(gca, 'YTickLabel', sorted_methods, 'YTick', y_pos);
xlabel('Composite Score (Accuracy + F1) / 2', 'FontSize', 12);
title('Method Ranking by Performance', 'FontSize', 14, 'FontWeight', 'bold');
xlim([0, 1.1]);
grid on;

% 添加标签
for i = 1:length(sorted_scores)
    text(sorted_scores(i)+0.02, y_pos(i), ...
        sprintf('Acc: %.1f%% | F1: %.1f%%', sorted_accuracy(i)*100, sorted_f1(i)*100), ...
        'VerticalAlignment', 'middle', 'FontSize', 10);
end

% 标记最佳方法
plot(sorted_scores(1), y_pos(1), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'gold');
text(0.5, y_pos(1), ' 🏆 BEST', 'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'bold');

saveas(gcf, 'outputs/fig15_method_ranking.png');
fprintf('图2已保存: fig15_method_ranking.png\n');

%% 图3: 性能与复杂度权衡 (Performance vs Complexity)
figure('Position', [100, 100, 600, 450]);

% X轴: 复杂度 (1=简单, 2=中等, 3=复杂)
complexity = [1, 2, 3];
% Y轴: 准确率
performance = accuracy;

% 气泡大小基于F1分数
bubble_size = f1_score * 500;

scatter(complexity, performance, bubble_size, colors, 'filled', 'Alpha', 0.7);
xlabel('Model Complexity (1=Low, 2=Medium, 3=High)', 'FontSize', 12);
ylabel('Accuracy', 'FontSize', 12);
title('Performance vs Complexity Trade-off', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'XTick', [1, 2, 3], 'XTickLabel', {'Simple (HOG)', 'Medium (CNN)', 'Complex (LSTM)'});
ylim([0, 1.1]);
grid on;

% 添加方法标签
for i = 1:length(methods)
    text(complexity(i)+0.15, performance(i), methods{i}, ...
        'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold');
end

% 添加注释框
annotation('textbox', [0.7, 0.8, 0.25, 0.12], 'String', ...
    'Bubble size = F1 Score', 'FontSize', 10, 'LineStyle', '--');

saveas(gcf, 'outputs/fig16_performance_complexity.png');
fprintf('图3已保存: fig16_performance_complexity.png\n');

fprintf('\n✅ 综合可视化完成！\n');
```

---
**说明：**
- 图1: 项目仪表板（数据集统计、准确率/F1对比、结论）
- 图2: 方法排名（基于综合得分）
- 图3: 性能与复杂度权衡（气泡图）

**使用方法：**
1. 确保outputs文件夹存在
2. 在MATLAB中运行脚本
3. 生成的图片将保存在outputs文件夹中
