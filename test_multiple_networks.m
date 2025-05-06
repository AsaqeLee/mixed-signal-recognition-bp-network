function accuracy = test_multiple_networks(networks, features, labels, mod_combinations)
% TEST_MULTIPLE_NETWORKS 测试多个神经网络的性能
%   accuracy = TEST_MULTIPLE_NETWORKS(networks, features, labels, mod_combinations)
%   测试训练好的多个BP神经网络在混叠信号识别上的性能
%
%   输入:
%   - networks: 多个训练好的BP神经网络及其信噪比范围信息
%   - features: 测试集特征矩阵
%   - labels: 测试集标签
%   - mod_combinations: 混叠信号组合标签
%
%   输出:
%   - accuracy: 每种混叠信号组合的识别准确率

% 记录测试开始时间
test_start = tic;

% 获取样本数和调制方式数
[num_samples, ~] = size(features);
num_mod_types = 4;  % 2ASK、BPSK、QPSK、16QAM
num_combinations = length(mod_combinations);

fprintf('开始评估多网络模型性能...\n');
fprintf('- 测试样本数: %d\n', num_samples);
fprintf('- 混叠组合数: %d\n', num_combinations);
fprintf('- 调制方式数: %d\n', num_mod_types);
fprintf('- 神经网络数: %d\n', length(networks.nets));

% 加载测试数据集以获取SNR信息
fprintf('加载SNR信息...\n');
test_data = load('data/mixed_signal_test.mat');
SNR_range = test_data.SNR_range;
num_snr = length(SNR_range);
fprintf('- 信噪比范围: %s dB\n', mat2str(SNR_range));

% 初始化预测结果数组
predictions = zeros(num_samples, 2);

% 计算每个信噪比每个组合的样本数
samples_per_combo_per_snr = num_samples / (num_combinations * num_snr);

% 对每个信噪比，使用对应的网络进行预测
fprintf('开始进行模型预测...\n');
pred_start = tic;

for snr_idx = 1:length(SNR_range)
    current_snr = SNR_range(snr_idx);
    fprintf('- 预测信噪比 %d dB 的样本...\n', current_snr);
    
    % 确定当前信噪比应使用的网络
    net_idx = -1;
    for i = 1:length(networks.snr_groups)
        group = networks.snr_groups{i};
        if ismember(current_snr, group.range)
            net_idx = i;
            fprintf('  使用%s网络\n', group.name);
            break;
        end
    end
    
    if net_idx == -1
        error('未找到适用于信噪比 %d dB 的网络', current_snr);
    end
    
    % 当前SNR下的样本范围
    snr_start = (snr_idx-1) * samples_per_combo_per_snr * num_combinations + 1;
    snr_end = snr_idx * samples_per_combo_per_snr * num_combinations;
    current_samples = snr_start:snr_end;
    
    % 提取当前信噪比的特征
    current_features = features(current_samples, :);
    
    % 使用对应网络进行预测
    current_net = networks.nets{net_idx};
    output = current_net(current_features');
    output = output';
    
    % 解码网络输出
    for i = 1:length(current_samples)
        % 预测第一个调制方式（前 num_mod_types 个输出）
        [~, pred_mod1] = max(output(i, 1:num_mod_types));
        
        % 预测第二个调制方式（后 num_mod_types 个输出）
        [~, pred_mod2] = max(output(i, (num_mod_types+1):(2*num_mod_types)));
        
        predictions(current_samples(i), :) = [pred_mod1, pred_mod2];
    end
end

pred_time = toc(pred_start);
fprintf('- 预测完成, 用时: %.2f秒\n', pred_time);

% 初始化准确率数组
accuracy = zeros(1, num_combinations);
% 不同信噪比下的准确率矩阵 [SNR, combination]
snr_accuracy = zeros(num_snr, num_combinations);

% 计算每种组合在不同SNR下的准确率
fprintf('计算不同信噪比下的准确率...\n');
for snr_idx = 1:num_snr
    % 当前SNR下的样本范围
    snr_start = (snr_idx-1) * samples_per_combo_per_snr * num_combinations + 1;
    snr_end = snr_idx * samples_per_combo_per_snr * num_combinations;
    
    % 计算每种组合在当前SNR下的准确率
    fprintf('  分析信噪比 %d dB 的性能...\n', SNR_range(snr_idx));
    for combo_idx = 1:num_combinations
        % 当前组合在当前SNR下的样本范围
        combo_start = snr_start + (combo_idx-1) * samples_per_combo_per_snr;
        combo_end = snr_start + combo_idx * samples_per_combo_per_snr - 1;
        
        % 获取当前范围内的样本
        curr_labels = labels(combo_start:combo_end, :);
        curr_preds = predictions(combo_start:combo_end, :);
        
        % 计算准确率
        correct = 0;
        for i = 1:size(curr_labels, 1)
            if isequal(sort(curr_preds(i,:)), sort(curr_labels(i,:)))
                correct = correct + 1;
            end
        end
        
        snr_accuracy(snr_idx, combo_idx) = correct / size(curr_labels, 1) * 100;
    end
end

% 计算每种混叠信号组合的总体准确率
for combo_idx = 1:num_combinations
    accuracy(combo_idx) = mean(snr_accuracy(:, combo_idx));
end

% 计算总体准确率
total_accuracy = mean(accuracy);

% 输出结果
fprintf('\n============================================================\n');
fprintf('             多网络混叠信号识别系统性能评估结果\n');
fprintf('============================================================\n');
fprintf('总体识别准确率: %.2f%%\n', total_accuracy);
fprintf('\n各混叠信号组合的识别准确率:\n');
for j = 1:num_combinations
    fprintf('%s: %.2f%%\n', mod_combinations{j}, accuracy(j));
end

% 显示各信噪比下的识别准确率
fprintf('\n============================================================\n');
fprintf('             各信噪比下的识别准确率\n');
fprintf('============================================================\n');
fprintf('%-10s', 'SNR(dB)');
for j = 1:num_combinations
    fprintf('%-15s', mod_combinations{j});
end
fprintf('%-10s\n', '平均');

% 打印结果表格
for i = 1:num_snr
    fprintf('%-10d', SNR_range(i));
    for j = 1:num_combinations
        fprintf('%-15.2f', snr_accuracy(i, j));
    end
    fprintf('%-10.2f\n', mean(snr_accuracy(i, :)));
end

% 高信噪比性能分析
high_snr_indices = find(SNR_range >= 15);
if ~isempty(high_snr_indices)
    high_snr_acc = mean(mean(snr_accuracy(high_snr_indices, :)));
    fprintf('\n高信噪比(≥15dB)平均准确率: %.2f%%\n', high_snr_acc);
end

% 中等信噪比性能分析
mid_snr_indices = find(SNR_range >= 5 & SNR_range < 15);
if ~isempty(mid_snr_indices)
    mid_snr_acc = mean(mean(snr_accuracy(mid_snr_indices, :)));
    fprintf('中等信噪比(5-15dB)平均准确率: %.2f%%\n', mid_snr_acc);
end

% 低信噪比性能分析
low_snr_indices = find(SNR_range < 5);
if ~isempty(low_snr_indices)
    low_snr_acc = mean(mean(snr_accuracy(low_snr_indices, :)));
    fprintf('低信噪比(<5dB)平均准确率: %.2f%%\n', low_snr_acc);
end

% 可视化不同信噪比下的识别准确率
fprintf('\n生成信噪比-准确率曲线图...\n');
figure;
plot(SNR_range, snr_accuracy, 'LineWidth', 2, 'Marker', 'o');
hold on;
plot(SNR_range, mean(snr_accuracy, 2), 'k--', 'LineWidth', 2, 'Marker', '*');
hold off;
title('多网络系统在不同信噪比下的混叠信号识别准确率');
xlabel('SNR (dB)');
ylabel('准确率 (%)');
legend([mod_combinations, {'平均准确率'}], 'Location', 'northwest');
grid on;

% 保存结果图像
saveas(gcf, 'data/multi_network_snr_accuracy.fig');
saveas(gcf, 'data/multi_network_snr_accuracy.png');
fprintf('不同信噪比下的准确率图像已保存到 data/multi_network_snr_accuracy.png\n');

% 计算总测试时间
test_time = toc(test_start);
fprintf('\n性能评估完成，总用时: %.2f秒 (%.2f分钟)\n', test_time, test_time/60);
fprintf('============================================================\n');

end 