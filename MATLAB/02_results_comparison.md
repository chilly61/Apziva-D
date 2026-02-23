# 02 Comparison of Results

```matlab
% MonReader - 02 结果对比可视化
%% 结果数据
methods = categorical({'HOG + RF', 'CNN + RF', 'LSTM'});
accuracy = [0.9740, 0.9481, 0.9870];
f1_score = [0.9583, 0.9130, 0.9796];
colors = [0.298, 0.686, 0.313; 0.204, 0.596, 0.859; 0.803, 0.145, 0.133];
%% 图: 准确率和F1分数对比
figure('Position', [100, 100, 700, 400]);
x = 1:3;
width = 0.35;
bar(x-width/2, accuracy, width, 'FaceColor', [0.298, 0.686, 0.313], 'DisplayName', 'Accuracy');
hold on;
bar(x+width/2, f1_score, width, 'FaceColor', [0.204, 0.596, 0.859], 'DisplayName', 'F1 Score');
hold off;
ylabel('Score', 'FontSize', 12);
title('Model Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
xticks(x);
xticklabels(methods);
legend('Location', 'best');
ylim([0, 1.15]);
grid on;
% 数值标注
for i = 1:3
    text(i-width/2, accuracy(i)+0.02, sprintf('%.2f', accuracy(i)), 'HorizontalAlignment', 'center', 'FontSize', 10);
    text(i+width/2, f1_score(i)+0.02, sprintf('%.2f', f1_score(i)), 'HorizontalAlignment', 'center', 'FontSize', 10);
end
fprintf('✅ 结果对比可视化完成！\n');
```
