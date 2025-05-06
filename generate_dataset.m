function [X, Y] = generate_dataset(mod_combinations, combination_labels, samples_per_class, sample_length, SNR_range, fs, fc1, fc2, dataset_type)
% GENERATE_DATASET 生成混叠信号数据集
%   [X, Y] = GENERATE_DATASET(mod_combinations, combination_labels, samples_per_class, sample_length, SNR_range, fs, fc1, fc2, dataset_type)
%   生成用于训练和测试的混叠信号数据集
%
%   输入:
%   - mod_combinations: 混叠调制方式组合的元胞数组
%   - combination_labels: 每种组合对应的标签矩阵
%   - samples_per_class: 每类混叠信号的样本数
%   - sample_length: 每个样本的采样点数
%   - SNR_range: 信噪比范围(dB)
%   - fs: 采样频率
%   - fc1: 第一个载波频率
%   - fc2: 第二个载波频率
%   - dataset_type: 数据集类型 ('train' 或 'test')
%
%   输出:
%   - X: 混叠信号样本数据
%   - Y: 对应的标签

% 初始化输出
num_combinations = length(mod_combinations);
total_samples = num_combinations * samples_per_class * length(SNR_range);
X = zeros(total_samples, sample_length);
Y = zeros(total_samples, 2);

% 随机数生成器设置 - 保证训练集和测试集不同
if strcmp(dataset_type, 'train')
    rng(42); % 训练集使用固定种子
    fprintf('使用固定随机种子(42)生成训练集\n');
else
    rng(100); % 测试集使用不同种子
    fprintf('使用固定随机种子(100)生成测试集\n');
end

% 确保data文件夹存在
if ~exist('data', 'dir')
    mkdir('data');
end

fprintf('开始生成%s数据集：\n', dataset_type);
fprintf('- 混叠组合数: %d\n', num_combinations);
fprintf('- 每类样本数: %d\n', samples_per_class);
fprintf('- 信噪比范围: %s dB\n', mat2str(SNR_range));
fprintf('- 总样本数: %d\n', total_samples);

% 计算总迭代次数用于进度显示
total_iterations = num_combinations * length(SNR_range) * samples_per_class;
iteration_count = 0;
last_percentage = 0;

% 显示进度条头部
fprintf('\n进度: [');
for i = 1:50
    fprintf(' ');
end
fprintf('] 0%%');

% 生成数据集
sample_idx = 1;
dataset_start = tic;

for combo_idx = 1:num_combinations
    fprintf('\n\n生成 %s 混叠信号...\n', mod_combinations{combo_idx});
    combo_start = tic;
    
    mod_type1_idx = combination_labels(combo_idx, 1);
    mod_type2_idx = combination_labels(combo_idx, 2);
    
    for snr_idx = 1:length(SNR_range)
        snr = SNR_range(snr_idx);
        fprintf('  - 信噪比: %d dB\n', snr);
        snr_start = tic;
        
        for sample = 1:samples_per_class
            % 更新进度
            iteration_count = iteration_count + 1;
            current_percentage = floor(iteration_count / total_iterations * 100);
            
            % 只在百分比变化时更新进度条
            if current_percentage > last_percentage
                % 回到进度条起始位置
                fprintf('\r进度: [');
                % 打印进度条
                filled = floor(current_percentage / 2);
                for i = 1:filled
                    fprintf('█');
                end
                for i = 1:50-filled
                    fprintf(' ');
                end
                fprintf('] %d%%', current_percentage);
                last_percentage = current_percentage;
            end
            
            % 生成两个不同调制方式的信号 - 传递信噪比参数
            s1 = generate_modulated_signal(mod_type1_idx, sample_length, fs, fc1, snr);
            
            % 如果是相同调制方式，生成不同数据
            if mod_type1_idx == mod_type2_idx
                % 等待一小段时间保证随机数不同
                pause(0.01);
            end
            
            s2 = generate_modulated_signal(mod_type2_idx, sample_length, fs, fc2, snr);
            
            % 混叠信号（简单叠加）
            mixed_signal = s1 + s2;
            
            % 添加噪声
            mixed_signal_with_noise = awgn(mixed_signal, snr, 'measured');
            
            % 保存样本和标签
            X(sample_idx, :) = mixed_signal_with_noise;
            Y(sample_idx, :) = [mod_type1_idx, mod_type2_idx];
            
            sample_idx = sample_idx + 1;
        end
        
        snr_time = toc(snr_start);
        fprintf('      完成 %d 个样本，用时: %.2f秒\n', samples_per_class, snr_time);
    end
    
    combo_time = toc(combo_start);
    fprintf('  完成 %s 组合的所有样本，用时: %.2f秒\n', mod_combinations{combo_idx}, combo_time);
end

% 打印完整进度条
fprintf('\r进度: [');
for i = 1:50
    fprintf('█');
end
fprintf('] 100%%\n');

% 保存数据集到data文件夹
if strcmp(dataset_type, 'train')
    save('data/mixed_signal_train.mat', 'X', 'Y', 'mod_combinations', 'combination_labels', 'SNR_range');
    fprintf('\n训练数据集已保存到: data/mixed_signal_train.mat\n');
else
    save('data/mixed_signal_test.mat', 'X', 'Y', 'mod_combinations', 'combination_labels', 'SNR_range');
    fprintf('\n测试数据集已保存到: data/mixed_signal_test.mat\n');
end

dataset_time = toc(dataset_start);
fprintf('数据集生成完成: %d 个样本，总用时: %.2f秒 (%.2f分钟)\n', sample_idx-1, dataset_time, dataset_time/60);
end 