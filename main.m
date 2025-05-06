%% 混叠信号识别系统（基于多BP神经网络）
% 该系统可识别不同调制方式（2ASK、BPSK、QPSK、16QAM）的两两混叠信号
% 使用多个神经网络处理不同信噪比范围的样本
clear;
clc;
close all;

% 记录程序开始时间
total_start_time = tic;

fprintf('============================================================\n');
fprintf('       混叠信号识别系统 (基于多BP神经网络) 启动\n');
fprintf('============================================================\n');
fprintf('开始时间: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

%% 参数设置
SNR_range = -5:5:20;          % 信噪比范围(dB)
samples_per_class = 500;      % 每类混叠信号的样本数
sample_length = 1024;         % 每个样本的采样点数
fs = 44100;                   % 采样频率
fc1 = 5000;                   % 第一个载波频率
fc2 = 10000;                  % 第二个载波频率
mod_types = {'2ASK', 'BPSK', 'QPSK', '16QAM'};  % 调制方式类型

fprintf('系统参数设置:\n');
fprintf('- 信噪比范围: %s dB\n', mat2str(SNR_range));
fprintf('- 每类样本数: %d\n', samples_per_class);
fprintf('- 采样点数: %d\n', sample_length);
fprintf('- 采样频率: %d Hz\n', fs);
fprintf('- 载波频率: %d Hz, %d Hz\n\n', fc1, fc2);

% 确保data文件夹存在
if ~exist('data', 'dir')
    mkdir('data');
    fprintf('创建data文件夹...\n');
end

% 生成混叠组合标签（仅包含不同调制方式的混叠）
mod_combinations = {};
combination_labels = [];
count = 1;
for i = 1:length(mod_types)
    for j = (i+1):length(mod_types)  % 从i+1开始，避免自己和自己混叠
        mod_combinations{count} = [mod_types{i}, '+', mod_types{j}];
        combination_labels(count,:) = [i, j];
        count = count + 1;
    end
end

fprintf('混叠信号组合 (%d种):\n', length(mod_combinations));
for i = 1:length(mod_combinations)
    fprintf('- %s\n', mod_combinations{i});
end
fprintf('\n');

%% 生成数据集
fprintf('============================================================\n');
fprintf('                    数据集生成阶段\n');
fprintf('============================================================\n');

dataset_start_time = tic;
fprintf('正在生成训练数据集...\n');
[X_train, Y_train] = generate_dataset(mod_combinations, combination_labels, samples_per_class, sample_length, SNR_range, fs, fc1, fc2, 'train');

fprintf('正在生成测试数据集...\n');
test_samples_per_class = round(samples_per_class * 0.3);
[X_test, Y_test] = generate_dataset(mod_combinations, combination_labels, test_samples_per_class, sample_length, SNR_range, fs, fc1, fc2, 'test');

dataset_time = toc(dataset_start_time);
fprintf('数据集生成完成，用时: %.2f秒\n\n', dataset_time);

%% 特征提取
fprintf('============================================================\n');
fprintf('                    特征提取阶段\n');
fprintf('============================================================\n');

feature_start_time = tic;
fprintf('正在提取训练集特征...\n');
features_train = extract_features(X_train);

fprintf('正在提取测试集特征...\n');
features_test = extract_features(X_test);

feature_time = toc(feature_start_time);
fprintf('特征提取完成，用时: %.2f秒\n', feature_time);
fprintf('训练集特征维度: %d x %d\n', size(features_train, 1), size(features_train, 2));
fprintf('测试集特征维度: %d x %d\n\n', size(features_test, 1), size(features_test, 2));

%% 训练多个BP神经网络
fprintf('============================================================\n');
fprintf('                多BP神经网络训练阶段\n');
fprintf('============================================================\n');

train_start_time = tic;
fprintf('正在按信噪比范围训练多个BP神经网络...\n');
fprintf('网络结构: 输入层(%d) - 隐藏层1(%d) - 隐藏层2(%d) - 输出层(%d)\n', ...
    size(features_train, 2), 32, 24, 8);
fprintf('训练算法: Levenberg-Marquardt\n');
fprintf('最大训练轮数: 1000\n');
fprintf('信噪比分组: 低(<5dB), 中(5-15dB), 高(15dB), 超高(20dB)\n');
fprintf('高信噪比优化: 已禁用非线性失真\n\n');

% 训练多个神经网络
multi_networks = train_multiple_networks(features_train, Y_train, SNR_range);

train_time = toc(train_start_time);
fprintf('多神经网络训练完成，用时: %.2f秒\n\n', train_time);

%% 测试网络性能
fprintf('============================================================\n');
fprintf('                    性能评估阶段\n');
fprintf('============================================================\n');

test_start_time = tic;
fprintf('正在评估多网络系统性能...\n');
accuracy = test_multiple_networks(multi_networks, features_test, Y_test, mod_combinations);

test_time = toc(test_start_time);
fprintf('性能评估完成，用时: %.2f秒\n\n', test_time);

%% 保存结果
save('data/multi_network_recognition_model.mat', 'multi_networks', 'mod_combinations', 'combination_labels');
fprintf('模型已保存到 data/multi_network_recognition_model.mat\n\n');

%% 可视化组合准确率
fprintf('============================================================\n');
fprintf('                    结果可视化阶段\n');
fprintf('============================================================\n');

fprintf('正在生成组合准确率图...\n');
figure(1);
bar(accuracy);
title('多网络系统混叠信号组合识别准确率');
xlabel('混叠信号类型');
ylabel('准确率 (%)');
set(gca, 'XTick', 1:length(mod_combinations));
set(gca, 'XTickLabel', mod_combinations);
grid on;

% 保存组合准确率图像
saveas(gcf, 'data/multi_network_combination_accuracy.fig');
saveas(gcf, 'data/multi_network_combination_accuracy.png');
fprintf('组合准确率图像已保存到 data 文件夹\n');

% 注意：信噪比准确率图像已在test_multiple_networks函数中生成并保存

%% 程序结束
total_time = toc(total_start_time);
fprintf('\n============================================================\n');
fprintf('                    程序执行完成\n');
fprintf('============================================================\n');
fprintf('结束时间: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('总运行时间: %.2f秒 (%.2f分钟)\n', total_time, total_time/60);
fprintf('- 数据集生成: %.2f秒 (%.1f%%)\n', dataset_time, dataset_time/total_time*100);
fprintf('- 特征提取: %.2f秒 (%.1f%%)\n', feature_time, feature_time/total_time*100);
fprintf('- 网络训练: %.2f秒 (%.1f%%)\n', train_time, train_time/total_time*100);
fprintf('- 性能评估: %.2f秒 (%.1f%%)\n', test_time, test_time/total_time*100);
fprintf('============================================================\n'); 