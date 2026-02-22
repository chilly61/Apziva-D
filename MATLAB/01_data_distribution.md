%% 数据准备
% 从EDA结果中获取
training_flip_segs = 65;
training_notflip_segs = 52;
testing_flip_segs = 25;
testing_notflip_segs = 52;

training_flip_imgs = 1162;
training_notflip_imgs = 1230;
testing_flip_imgs = 105;
testing_notflip_imgs = 307;

%% 图1: 片段数量分布 (Segment Distribution)
figure('Position', [100, 100, 600, 400]);

categories = categorical(...
    {'Training Flip', 'Training NotFlip', 'Testing Flip', 'Testing NotFlip'}, ...
    {'Training Flip', 'Training NotFlip', 'Testing Flip', 'Testing NotFlip'});
seg_counts = [training_flip_segs, training_notflip_segs, testing_flip_segs, testing_notflip_segs];

colors = [0.2 0.6 0.8; 0.2 0.8 0.4; 0.8 0.4 0.2; 0.8 0.6 0.2];
bar(categories, seg_counts, 0.6);
ylabel('Segment Count', 'FontSize', 12);
title('Segment Distribution by Category', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% 添加数值标签
for i = 1:length(seg_counts)
    text(i, seg_counts(i) + max(seg_counts)*0.02, num2str(seg_counts(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11);
end

ylim([0, max(seg_counts) * 1.15]);
saveas(gcf, 'C:\\Users\\75346\\Desktop\\Apziva Project D\\outputs\\fig01_segment_distribution.png');
fprintf('图1已保存: fig01_segment_distribution.png\n');

%% 图2: 图片数量分布 (Image Distribution)
figure('Position', [100, 100, 600, 400]);
categories_img = categorical(...
    {'Training Flip', 'Training NotFlip', 'Testing Flip', 'Testing NotFlip'}, ...
    {'Training Flip', 'Training NotFlip', 'Testing Flip', 'Testing NotFlip'});
img_counts = [training_flip_imgs, training_notflip_imgs, testing_flip_imgs, testing_notflip_imgs];

bar(categories_img, img_counts, 0.6);
ylabel('Image Count', 'FontSize', 12);
title('Image Distribution by Category', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% 添加数值标签
for i = 1:length(img_counts)
    text(i, img_counts(i) + 30, num2str(img_counts(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11);
end

ylim([0, max(img_counts) * 1.15]);
saveas(gcf, 'C:\\Users\\75346\\Desktop\\Apziva Project D\\outputs\\fig02_image_distribution.png');
fprintf('图2已保存: fig02_image_distribution.png\n');

%% 图3: 训练集 vs 测试集 比例 (Train/Test Split)
figure('Position', [100, 100, 500, 400]);

train_total = training_flip_segs + training_notflip_segs;
test_total = testing_flip_segs + testing_notflip_segs;

labels = {'Training', 'Testing'};
sizes = [train_total, test_total];
pie(sizes, labels);
title('Train/Test Split Ratio', 'FontSize', 14, 'FontWeight', 'bold');

% 添加百分比标签
percentages = sizes / sum(sizes) * 100;
legend_labels = cellfun(@(x,y,z) sprintf('%s: %d (%.1f%%)', x,y,z), labels, num2cell(sizes), num2cell(percentages), 'UniformOutput', false);
legend(legend_labels, 'Location', 'northoutside');
saveas(gcf, 'C:\\Users\\75346\\Desktop\\Apziva Project D\\outputs\\fig03_train_test_split.png');
fprintf('图3已保存: fig03_train_test_split.png\n');

%% 图4: Flip vs NotFlip 分布 (Class Distribution)
figure('Position', [100, 100, 500, 400]);

flip_total = training_flip_segs + testing_flip_segs;
notflip_total = training_notflip_segs + testing_notflip_segs;

labels_class = {'Flip', 'NotFlip'};
class_sizes = [flip_total, notflip_total];
pie(class_sizes, labels_class);
title('Class Distribution (All Data)', 'FontSize', 14, 'FontWeight', 'bold');

percentages_class = class_sizes / sum(class_sizes) * 100;
legend_labels_class = cellfun(@(x,y,z) sprintf('%s: %d (%.1f%%)', x,y,z), labels_class, num2cell(class_sizes), num2cell(percentages_class), 'UniformOutput', false);
legend(legend_labels_class, 'Location', 'northoutside');
saveas(gcf, 'C:\\Users\\75346\\Desktop\\Apziva Project D\\outputs\\fig04_class_distribution.png');
fprintf('图4已保存: fig04_class_distribution.png\n');

fprintf('\n✅ 所有可视化完成！\n');
