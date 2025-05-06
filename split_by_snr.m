function [features_by_snr, labels_by_snr, snr_groups] = split_by_snr(features, labels, SNR_range)
% SPLIT_BY_SNR 根据信噪比范围拆分数据集
%   [features_by_snr, labels_by_snr, snr_groups] = SPLIT_BY_SNR(features, labels, SNR_range)
%   将特征和标签按信噪比范围分组
%
%   输入:
%   - features: 特征矩阵，每行代表一个样本的特征向量
%   - labels: 对应的标签矩阵，每行代表一个样本的标签 [调制方式1, 调制方式2]
%   - SNR_range: 信噪比范围数组
%
%   输出:
%   - features_by_snr: 按信噪比分组的特征元胞数组
%   - labels_by_snr: 按信噪比分组的标签元胞数组
%   - snr_groups: 信噪比分组信息

% 信噪比分组设置（可根据需要调整）
% 低信噪比：< 5dB
% 中等信噪比：5dB - 10dB
% 高信噪比：15dB
% 超高信噪比：20dB (单独训练，避免与其他信噪比混合)
low_snr_threshold = 5;
mid_snr_threshold = 15;
high_snr_threshold = 20;

% 定义信噪比分组
low_snr_indices = find(SNR_range < low_snr_threshold);
mid_snr_indices = find(SNR_range >= low_snr_threshold & SNR_range < mid_snr_threshold);
high_snr_indices = find(SNR_range >= mid_snr_threshold & SNR_range < high_snr_threshold);
very_high_snr_indices = find(SNR_range >= high_snr_threshold);

% 创建信噪比分组信息
snr_groups = {
    struct('name', 'low_snr', 'range', SNR_range(low_snr_indices), 'indices', low_snr_indices),
    struct('name', 'mid_snr', 'range', SNR_range(mid_snr_indices), 'indices', mid_snr_indices),
    struct('name', 'high_snr', 'range', SNR_range(high_snr_indices), 'indices', high_snr_indices)
};

% 如果有超高信噪比数据，也添加到分组中
if ~isempty(very_high_snr_indices)
    snr_groups{end+1} = struct('name', 'very_high_snr', 'range', SNR_range(very_high_snr_indices), 'indices', very_high_snr_indices);
end

% 加载数据集以获取每个SNR的样本数量信息
fprintf('加载数据集信息...\n');
data_info = load('data/mixed_signal_train.mat', 'mod_combinations');
num_combinations = length(data_info.mod_combinations);

% 计算每个信噪比每个组合的样本数
[num_samples, ~] = size(features);
num_snr = length(SNR_range);
samples_per_combo_per_snr = num_samples / (num_combinations * num_snr);

fprintf('拆分数据集：\n');
% 初始化分组后的特征和标签元胞数组
features_by_snr = cell(length(snr_groups), 1);
labels_by_snr = cell(length(snr_groups), 1);

% 对每个信噪比分组进行处理
for group_idx = 1:length(snr_groups)
    group = snr_groups{group_idx};
    fprintf('- 处理%s组 (', group.name);
    
    % 打印该组的信噪比范围
    for i = 1:length(group.range)
        if i > 1
            fprintf(', ');
        end
        fprintf('%d dB', group.range(i));
    end
    fprintf(')...\n');
    
    % 收集该组的样本索引
    all_indices = [];
    for snr_idx = group.indices
        % 当前SNR下的样本范围
        snr_start = (snr_idx-1) * samples_per_combo_per_snr * num_combinations + 1;
        snr_end = snr_idx * samples_per_combo_per_snr * num_combinations;
        all_indices = [all_indices, snr_start:snr_end];
    end
    
    % 提取该组的特征和标签
    features_by_snr{group_idx} = features(all_indices, :);
    labels_by_snr{group_idx} = labels(all_indices, :);
    
    fprintf('  样本数量: %d\n', length(all_indices));
end

fprintf('数据集按信噪比拆分完成\n');
end 