function features = extract_features(X)
% EXTRACT_FEATURES 从混叠信号中提取统计特征
%   features = EXTRACT_FEATURES(X)
%   从混叠信号中提取用于识别的统计特征
%
%   输入:
%   - X: 混叠信号样本数据，每行代表一个样本
%
%   输出:
%   - features: 提取的特征矩阵，每行代表一个样本的特征向量

[num_samples, ~] = size(X);
features = zeros(num_samples, 26);  % 预分配26个特征(新增一个特征)

% 显示特征提取信息
fprintf('开始提取特征:\n');
fprintf('- 样本数量: %d\n', num_samples);
fprintf('- 特征数量: 26\n');  % 更新特征数量

% 初始化进度显示
fprintf('\n进度: [');
for i = 1:50
    fprintf(' ');
end
fprintf('] 0%%');

feature_start = tic;
update_interval = max(1, round(num_samples / 100)); % 每完成1%的样本更新一次进度

for i = 1:num_samples
    % 更新进度条
    if mod(i, update_interval) == 0 || i == 1 || i == num_samples
        percentage = floor(i / num_samples * 100);
        fprintf('\r进度: [');
        filled = floor(percentage / 2);
        for j = 1:filled
            fprintf('█');
        end
        for j = 1:50-filled
            fprintf(' ');
        end
        fprintf('] %d%% (样本 %d/%d)', percentage, i, num_samples);
    end
    
    % 获取当前信号
    signal = X(i, :);
    
    % 特征1-5: 时域统计特征
    % 1. 均值
    features(i, 1) = mean(signal);
    
    % 2. 方差
    features(i, 2) = var(signal);
    
    % 3. 偏度 - 分布偏斜度
    features(i, 3) = skewness(signal);
    
    % 4. 峰度 - 分布尖锐度
    features(i, 4) = kurtosis(signal);
    
    % 5. 峰值因子 - 峰值与均方根比值
    features(i, 5) = max(abs(signal)) / sqrt(mean(signal.^2));
    
    % 特征6-10: 高阶矩
    % 6-10. 2-6阶矩
    for j = 2:6
        features(i, 4+j) = moment(signal, j);
    end
    
    % 特征11: 零交叉率
    zero_crossings = sum(diff(sign(signal)) ~= 0);
    features(i, 11) = zero_crossings / length(signal);
    
    % 频域特征
    % 计算信号的功率谱密度
    [psd, f] = pwelch(signal, [], [], [], 1);
    
    % 归一化功率谱
    psd_norm = psd / sum(psd);
    
    % 特征12-14: 频域统计特征
    % 12. 频谱中心
    features(i, 12) = sum(f .* psd_norm);
    
    % 13. 频谱标准差
    features(i, 13) = sqrt(sum((f - features(i, 12)).^2 .* psd_norm));
    
    % 14. 频谱偏度
    features(i, 14) = sum(((f - features(i, 12)).^3) .* psd_norm) / (features(i, 13)^3);
    
    % 特征15: 频谱峭度
    features(i, 15) = sum(((f - features(i, 12)).^4) .* psd_norm) / (features(i, 13)^4);
    
    % 特征16-17: 频谱能量集中度
    % 16. 频谱能量25%点
    features(i, 16) = find(cumsum(psd_norm) >= 0.25, 1) / length(f);
    
    % 17. 频谱能量75%点
    features(i, 17) = find(cumsum(psd_norm) >= 0.75, 1) / length(f);
    
    % 包络特征
    % 计算信号包络
    analytic_signal = hilbert(signal);
    envelope = abs(analytic_signal);
    
    % 特征18-19: 包络统计特征
    % 18. 包络均值
    features(i, 18) = mean(envelope);
    
    % 19. 包络标准差
    features(i, 19) = std(envelope);
    
    % 特征20: 调制频谱相关性
    envelope_fft = abs(fft(envelope));
    signal_fft = abs(fft(signal));
    
    % 修复：使用corrcoef函数计算相关系数，并获取非对角线元素
    half_len = floor(length(envelope_fft)/2);
    corr_matrix = corrcoef(envelope_fft(1:half_len), signal_fft(1:half_len));
    features(i, 20) = corr_matrix(1, 2); % 获取非对角线元素作为相关系数
    
    % 特征21-22: 瞬时相位特征
    % 计算瞬时相位
    inst_phase = unwrap(angle(analytic_signal));
    
    % 21. 相位变化率标准差
    phase_diff = diff(inst_phase);
    features(i, 21) = std(phase_diff);
    
    % 22. 相位分布集中度
    features(i, 22) = kurtosis(inst_phase);
    
    % 23. 高信噪比下有效的非线性特征
    signal_squared = signal.^2;
    % 修复：使用corrcoef计算相关系数
    corr_matrix = corrcoef(signal(1:end-1), signal_squared(2:end));
    features(i, 23) = corr_matrix(1, 2);
    
    % 24. 包络与相位相关性
    % 修复：使用corrcoef计算相关系数
    corr_matrix = corrcoef(envelope, inst_phase);
    features(i, 24) = corr_matrix(1, 2);
    
    % 25. 信号零交叉间隔的标准差
    zero_crossings_idx = find(diff(sign(signal)) ~= 0);
    if length(zero_crossings_idx) > 1
        features(i, 25) = std(diff(zero_crossings_idx));
    else
        features(i, 25) = 0;
    end
    
    % 26. 新增特征: 针对2ASK-BPSK混叠信号的幅度-相位相关波动特征
    % 将信号分成多个段，计算每段的幅度-相位相关系数，然后计算这些相关系数的标准差
    segment_length = floor(length(signal) / 8); % 将信号分为8段
    amp_phase_corrs = zeros(8, 1);
    
    for seg = 1:8
        start_idx = (seg-1) * segment_length + 1;
        end_idx = min(seg * segment_length, length(signal));
        
        if end_idx - start_idx < 5  % 确保有足够的样本点
            continue;
        end
        
        seg_envelope = envelope(start_idx:end_idx);
        seg_phase = inst_phase(start_idx:end_idx);
        
        corr_matrix = corrcoef(seg_envelope, seg_phase);
        if size(corr_matrix, 1) > 1
            amp_phase_corrs(seg) = corr_matrix(1, 2);
        end
    end
    
    % 计算各段幅度-相位相关系数的标准差
    % 2ASK-BPSK混叠信号的特点是幅度-相位相关系数在不同信号段之间波动较大
    features(i, 26) = std(amp_phase_corrs);
end

% 完成进度条
fprintf('\r进度: [');
for i = 1:50
    fprintf('█');
end
fprintf('] 100%%\n');

% 特征归一化
fprintf('进行特征归一化...\n');
features = (features - mean(features)) ./ std(features);

% 移除NaN和Inf
nan_count = sum(sum(isnan(features)));
inf_count = sum(sum(isinf(features)));
features(isnan(features)) = 0;
features(isinf(features)) = 0;

if nan_count > 0 || inf_count > 0
    fprintf('警告: 检测到 %d 个NaN值和 %d 个Inf值，已替换为0\n', nan_count, inf_count);
end

feature_time = toc(feature_start);
fprintf('特征提取完成，用时: %.2f秒 (%.2f分钟)\n', feature_time, feature_time/60);
fprintf('特征尺寸: %d x %d\n\n', size(features, 1), size(features, 2));

end 