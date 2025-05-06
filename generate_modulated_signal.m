function signal = generate_modulated_signal(mod_type_idx, signal_length, fs, fc, snr)
% GENERATE_MODULATED_SIGNAL 生成指定调制方式的信号
%   signal = GENERATE_MODULATED_SIGNAL(mod_type_idx, signal_length, fs, fc, snr)
%   生成指定调制方式（2ASK、BPSK、QPSK、16QAM）的信号
%
%   输入:
%   - mod_type_idx: 调制方式索引 (1=2ASK, 2=BPSK, 3=QPSK, 4=16QAM)
%   - signal_length: 信号长度（采样点数）
%   - fs: 采样频率
%   - fc: 载波频率
%   - snr: 信噪比(dB)，可选参数
%
%   输出:
%   - signal: 生成的调制信号

% 检查snr参数是否存在
if nargin < 5
    snr = 0; % 默认值
end

% 调制方式映射
mod_types = {'2ASK', 'BPSK', 'QPSK', '16QAM'};

% 参数计算
t = (0:signal_length-1)/fs;
Ts = 1/(fs/100);  % 符号周期 (每100个采样点一个符号)
num_symbols = ceil(signal_length/(fs*Ts));  % 符号数量
symbol_samples = round(fs*Ts);  % 每个符号的采样点数

% 显示详细信息（仅在前几次调用时显示，避免日志过多）
persistent call_count;
if isempty(call_count)
    call_count = 0;
end

if call_count < 10
    fprintf('生成 %s 信号 [fc=%d Hz, 长度=%d, SNR=%d dB]\n', ...
        mod_types{mod_type_idx}, fc, signal_length, snr);
    call_count = call_count + 1;
    
    % 当达到一定次数后，提示后续信息将被抑制
    if call_count == 10
        fprintf('...... 后续信号生成信息将被抑制以减少输出量 ......\n');
    end
end

% 生成随机二进制数据
switch mod_type_idx
    case 1  % 2ASK
        bit_stream = randi([0, 1], num_symbols, 1);
    case 2  % BPSK
        bit_stream = randi([0, 1], num_symbols, 1);
    case 3  % QPSK
        bit_stream = randi([0, 3], num_symbols, 1);
    case 4  % 16QAM
        bit_stream = randi([0, 15], num_symbols, 1);
    otherwise
        error('不支持的调制方式索引');
end

% 生成调制信号
signal = zeros(1, signal_length);
carrier = cos(2*pi*fc*t);

for i = 1:num_symbols
    % 确定当前符号的时间范围
    start_idx = (i-1)*symbol_samples + 1;
    end_idx = min(i*symbol_samples, signal_length);
    
    % 根据不同调制方式生成调制信号
    switch mod_type_idx
        case 1  % 2ASK
            % 2ASK调制：0映射到振幅为0.5，1映射到振幅为1
            amp = 0.5 + 0.5*bit_stream(i);
            signal(start_idx:end_idx) = amp * carrier(start_idx:end_idx);
            
        case 2  % BPSK
            % BPSK调制：0映射到相位0，1映射到相位π
            phase = pi * bit_stream(i);
            signal(start_idx:end_idx) = cos(2*pi*fc*t(start_idx:end_idx) + phase);
            
        case 3  % QPSK
            % QPSK调制：00,01,10,11映射到相位π/4,3π/4,5π/4,7π/4
            phase = (bit_stream(i) * pi/2) + pi/4;
            signal(start_idx:end_idx) = cos(2*pi*fc*t(start_idx:end_idx) + phase);
            
        case 4  % 16QAM
            % 16QAM调制：将0-15映射到16个复数点
            % 简化实现：将每个符号映射到一个幅度和相位组合
            amp_levels = [0.33, 0.67, 1.0, 1.33];
            phase_levels = [0, pi/2, pi, 3*pi/2];
            
            % 从0-15映射到幅度和相位的组合
            amp_idx = floor(bit_stream(i)/4) + 1;
            phase_idx = mod(bit_stream(i), 4) + 1;
            
            amp = amp_levels(amp_idx);
            phase = phase_levels(phase_idx);
            
            signal(start_idx:end_idx) = amp * cos(2*pi*fc*t(start_idx:end_idx) + phase);
    end
end

% 归一化信号能量
signal = signal / sqrt(mean(signal.^2));

% 对高信噪比信号进行额外处理（已移除非线性失真）
if snr >= 15 && call_count <= 10
    fprintf('注意: 高信噪比信号处理已禁用 [SNR=%d dB]\n', snr);
end
end 