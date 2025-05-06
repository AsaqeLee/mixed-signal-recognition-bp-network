function networks = train_multiple_networks(features, labels, SNR_range)
% TRAIN_MULTIPLE_NETWORKS 训练多个BP神经网络
%   networks = TRAIN_MULTIPLE_NETWORKS(features, labels, SNR_range)
%   针对不同信噪比范围训练多个BP神经网络用于混叠信号识别
%
%   输入:
%   - features: 特征矩阵，每行代表一个样本的特征向量
%   - labels: 对应的标签矩阵，每行代表一个样本的标签 [调制方式1, 调制方式2]
%   - SNR_range: 信噪比范围数组
%
%   输出:
%   - networks: 包含多个训练好的神经网络的结构体

% 记录训练开始时间
train_start = tic;

fprintf('============================================================\n');
fprintf('               多BP神经网络训练阶段\n');
fprintf('============================================================\n');

% 按信噪比范围拆分数据集
fprintf('按信噪比范围拆分数据集...\n');
[features_by_snr, labels_by_snr, snr_groups] = split_by_snr(features, labels, SNR_range);

% 初始化神经网络元胞数组
networks = struct();
networks.snr_groups = snr_groups;
networks.nets = cell(length(snr_groups), 1);
networks.group_names = cell(length(snr_groups), 1);
networks.snr_ranges = cell(length(snr_groups), 1);

% 调制方式数量
num_mod_types = 4;  % 2ASK、BPSK、QPSK、16QAM

% 对每个信噪比范围训练一个神经网络
for i = 1:length(snr_groups)
    group = snr_groups{i};
    fprintf('\n============================================================\n');
    fprintf('   训练%s神经网络 (%s)\n', group.name, strjoin(arrayfun(@(x) sprintf('%d dB', x), group.range, 'UniformOutput', false), ', '));
    fprintf('============================================================\n');
    
    % 获取当前信噪比组的特征和标签
    curr_features = features_by_snr{i};
    curr_labels = labels_by_snr{i};
    [num_samples, num_features] = size(curr_features);
    
    fprintf('开始训练BP神经网络...\n');
    fprintf('- 样本数量: %d\n', num_samples);
    fprintf('- 特征维度: %d\n', num_features);
    fprintf('- 调制方式: %d种\n', num_mod_types);
    
    % 将标签转换为one-hot编码
    fprintf('转换为one-hot编码...\n');
    targets = zeros(num_samples, num_mod_types*2);
    
    for j = 1:num_samples
        % 第一个调制方式的one-hot编码
        targets(j, curr_labels(j, 1)) = 1;
        
        % 第二个调制方式的one-hot编码
        targets(j, num_mod_types + curr_labels(j, 2)) = 1;
    end
    
    % 创建BP神经网络 - 简化版
    fprintf('创建简化神经网络...\n');
    fprintf('- 网络结构: %d-%d-%d-%d\n', num_features, 32, 24, num_mod_types*2);
    fprintf('- 隐藏层1激活函数: ReLU (poslin)\n');
    fprintf('- 隐藏层2激活函数: Tanh (tansig)\n');
    fprintf('- 输出层激活函数: Sigmoid (logsig)\n');
    
    % 使用三层结构：输入层 - 隐藏层 - 隐藏层 - 输出层
    net = feedforwardnet([32, 24], 'trainscg');  % 使用比例共轭梯度算法训练
    
    % 设置神经网络训练参数
    fprintf('设置训练参数...\n');
    net.trainParam.epochs = 1000;     % 增加最大训练轮数为1000
    net.trainParam.goal = 1e-4;      % 性能目标
    net.trainParam.min_grad = 1e-7;  % 最小梯度
    net.trainParam.max_fail = 20;    % 最大验证失败次数
    net.divideParam.trainRatio = 0.7;  % 训练集比例
    net.divideParam.valRatio = 0.15;   % 验证集比例
    net.divideParam.testRatio = 0.15;  % 测试集比例
    
    fprintf('- 最大训练轮数: %d\n', net.trainParam.epochs);
    fprintf('- 性能目标: %g\n', net.trainParam.goal);
    fprintf('- 最小梯度: %g\n', net.trainParam.min_grad);
    fprintf('- 最大验证失败次数: %d\n', net.trainParam.max_fail);
    fprintf('- 数据分割比例: 训练%.0f%%, 验证%.0f%%, 测试%.0f%%\n', ...
        net.divideParam.trainRatio*100, ...
        net.divideParam.valRatio*100, ...
        net.divideParam.testRatio*100);
    
    % 设置每个层的传递函数
    net.layers{1}.transferFcn = 'poslin';  % 隐藏层1使用ReLU
    net.layers{2}.transferFcn = 'tansig';  % 隐藏层2使用tansig
    
    % 显示训练进度
    net.trainParam.showWindow = true;
    net.trainParam.showCommandLine = false;
    
    % 准备训练数据
    X_train = curr_features';
    T_train = targets';
    
    % 使用Levenberg-Marquardt算法训练网络
    fprintf('使用Levenberg-Marquardt算法训练网络...\n');
    fprintf('训练即将开始，请等待训练窗口显示进度...\n\n');
    
    net.trainFcn = 'trainlm';  % 使用Levenberg-Marquardt算法
    net.trainParam.mu_dec = 0.1;
    net.trainParam.mu_inc = 10;
    
    % 训练网络 - 不使用并行计算
    net_train_start = tic;
    [net, tr] = train(net, X_train, T_train);
    net_train_time = toc(net_train_start);
    
    % 输出训练结果
    fprintf('\n============================================================\n');
    fprintf('%s神经网络训练完成\n', group.name);
    fprintf('- 训练用时: %.2f秒 (%.2f分钟)\n', net_train_time, net_train_time/60);
    fprintf('- 训练集性能: %f\n', tr.best_perf);
    fprintf('- 验证集性能: %f\n', tr.best_vperf);
    fprintf('- 测试集性能: %f\n', tr.best_tperf);
    fprintf('- 训练轮数: %d/%d\n', length(tr.epoch), net.trainParam.epochs);
    fprintf('============================================================\n');
    
    % 保存当前信噪比范围的神经网络
    networks.nets{i} = net;
    networks.group_names{i} = group.name;
    networks.snr_ranges{i} = group.range;
end

% 计算总训练时间
train_time = toc(train_start);
fprintf('\n============================================================\n');
fprintf('          多BP神经网络训练完成\n');
fprintf('============================================================\n');
fprintf('- 总训练时间: %.2f秒 (%.2f分钟)\n', train_time, train_time/60);
fprintf('- 训练的神经网络数量: %d\n', length(snr_groups));

for i = 1:length(snr_groups)
    fprintf('- %s网络: 用于信噪比 ', networks.group_names{i});
    fprintf('%s\n', strjoin(arrayfun(@(x) sprintf('%d dB', x), networks.snr_ranges{i}, 'UniformOutput', false), ', '));
end
fprintf('============================================================\n');

end 