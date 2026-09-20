% 25新平衡

%% 注释
%机体=车体，腿=摆杆
%i = (l , r)
%d_    : 一阶导                        dd_ : 二阶导

%theta_w_i  :驱动轮转角                theta_l_i  :腿倾斜角
%theta_b    :机体倾斜角                phi        :偏航角
%s          :两轮水平方向移动距离       s_b        :机体质心水平方向移动距离
%h_b        :机体质心竖直方向移动距离   s_l_i      :腿质心水平方向移动距离
%h_l_i      :腿质心竖直方向移动距离     T_lw_i      :驱动轮转矩（腿-轮）
%T_bl_i      :腿部转矩（机体-腿）        f_i        :地面对驱动轮摩擦力
%F_ws_i     :驱动轮对腿水平方向作用力   F_wh_i     :驱动轮对腿竖直方向作用力
%F_bs_i     :腿对机体水平方向作用力     F_bh_i     :腿对机体竖直方向作用力

%R_w        :驱动轮半径                R_l        :驱动轮轮距/2
%l_i        :腿长*1                   l_w_i      :驱动轮到腿部质心距离*1
%l_b_i      :腿部质心到机体距离*1       l_c        :机体质心到腿部髋关节距离
%m_w        :驱动轮质量                m_l        :腿部质量
%m_b        :机体质量                  I_w        :驱动轮转动惯量
%I_l        :腿部转动惯量*1            I_b        :机体转动惯量
%I_z        :机器人z轴转动惯量*2

if exist('cal_done', 'var') == 0
    %% 变量设定及初始值设置
    syms theta_w_l theta_w_r theta_l_l theta_l_r theta_b phi s s_b h_b
    syms s_l_l s_l_r h_l_l h_l_r T_lw_l T_lw_r T_bl_l T_bl_r f_l f_r
    syms F_ws_l F_ws_r F_wh_l F_wh_r F_bs_l F_bs_r F_bh_l F_bh_r
    syms d_s dd_s d_phi dd_phi d_theta_l_l dd_theta_l_l d_theta_l_r dd_theta_l_r d_theta_b dd_theta_b
    syms d_theta_w_l dd_theta_w_l d_theta_w_r dd_theta_w_r

    syms g R_w R_l l_l l_r l_w_l l_w_r l_b_l l_b_r l_c m_w m_l m_b I_w I_l_l I_l_r I_b I_z

    eqn1 = ((I_w * l_l / R_w) + m_w * R_w * l_l + m_l * R_w * l_b_l) * dd_theta_w_l + (m_l * l_w_l * l_b_l - I_l_l) * dd_theta_l_l + (m_l * l_w_l + 0.5 * m_b * l_l) * g * theta_l_l + T_bl_l - T_lw_l * (1 + l_l / R_w) == 0;
    eqn2 = ((I_w * l_r / R_w) + m_w * R_w * l_r + m_l * R_w * l_b_r) * dd_theta_w_r + (m_l * l_w_r * l_b_r - I_l_r) * dd_theta_l_r + (m_l * l_w_r + 0.5 * m_b * l_r) * g * theta_l_r + T_bl_r - T_lw_r * (1 + l_r / R_w) == 0;
    % eqn3 =- (m_w * R_w ^ 2 + I_w + m_l * R_w ^ 2 + 0.5 * m_b * R_w ^ 2) * (dd_theta_w_l + dd_theta_w_r) - (m_l * R_w * l_w_l + 0.5 * m_b * R_w * l_l) * dd_theta_l_l - (m_l * R_w * l_w_r + 0.5 * m_b * R_w * l_r) * dd_theta_l_r + T_lw_l + T_lw_r - m_b * l_c * R_w * dd_theta_b == 0;
    eqn3 =- (m_w * R_w ^ 2 + I_w + m_l * R_w ^ 2 + 0.5 * m_b * R_w ^ 2) * (dd_theta_w_l + dd_theta_w_r) - (m_l * R_w * l_w_l + 0.5 * m_b * R_w * l_l) * dd_theta_l_l - (m_l * R_w * l_w_r + 0.5 * m_b * R_w * l_r) * dd_theta_l_r + T_lw_l + T_lw_r == 0;
    eqn4 = (m_w * R_w * l_c + I_w * l_c / R_w + m_l * R_w * l_c) * (dd_theta_w_l + dd_theta_w_r) + m_l * l_w_l * l_c * dd_theta_l_l + m_l * l_w_r * l_c * dd_theta_l_r - I_b * dd_theta_b + m_b * g * l_c * theta_b - (T_lw_l + T_lw_r) * l_c / R_w - (T_bl_l + T_bl_r) == 0;
    eqn5 = (0.5 * I_z * R_w / R_l + I_w * R_l / R_w) * (dd_theta_w_l - dd_theta_w_r) + (0.5 * I_z * l_l / R_l) * dd_theta_l_l - (0.5 * I_z * l_r / R_l) * dd_theta_l_r - T_lw_l * R_l / R_w + T_lw_r * R_l / R_w == 0;
    eqn6 = dd_s - 0.5 * R_w * (dd_theta_w_l + dd_theta_w_r) == 0;
    % eqn7 = dd_phi * R_l - (0.5 * R_w * (-dd_theta_w_l + dd_theta_w_r) - 0.5 * l_l * dd_theta_l_l + 0.5 * l_r * dd_theta_l_r) == 0;
    eqn7 = dd_phi * R_l - (0.5 * R_w * (-dd_theta_w_l + dd_theta_w_r) - 0.5 * l_l * dd_theta_l_l + 0.5 * l_r * dd_theta_l_r) == 0;
    [dd_s, dd_phi, dd_theta_w_l, dd_theta_w_r, dd_theta_l_l, dd_theta_l_r, dd_theta_b] = solve(eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, dd_s, dd_phi, dd_theta_w_l, dd_theta_w_r, dd_theta_l_l, dd_theta_l_r, dd_theta_b);

    cal_done = 1;
end

min_L = 0.10;
d_L = 0.01;
max_L = 0.40;

[X, Y] = meshgrid(0.06:0.02:0.4, 0.06:0.02:0.4);

L_l0 = X(:)';
L_r0 = Y(:)';

% 预分配结果数组
num_iterations = length(0.06:0.02:0.4) ^ 2;
x_length = 10;
u_length = 4;

K_all = zeros(u_length, x_length, num_iterations);

%% [s,d_s,phi,d_phi,theta_l_l,d_theta_l_l,theta_l_r,d_theta_l_r,theta_b,d_theta_b]
% Q = 1.0 .* diag([10 800 400 20 2000 100 2000 100 6000 60]);
%Q = diag([10 800 1200 10 50 10 50 10 10 10 ]);
% Q = 1.0 .* diag([50 200 200 100 2000 100 2000 100 10000 100 ]);
% Q = diag([100 200 100 200 2000 100 2000 100 40000 1000]);
% Q = diag([150 350 500 400 400 100 400 100 40000 200 ]);
%% [T_lw_l,T_lw_r,T_bl_l,T_bl_r]
% R = diag([10.0 10.0 1.0 1.0]);
%R = diag([15.0 15.0 1.0 1.0]);
Q = diag([10 300 400 20 300 10 300 10 6000 60 ])
R = diag([1 1 1 1 ])

%% 全工况K矩阵计算

g0 = 9.80;
R_w0 = 0.06;
R_l0 = 0.18; %0.4128/2
l_c0 = 0.0;
m_w0 = 1.1;
m_l0 = 0.8; %0.6;
m_b0 = 20; %20-2*0.9-2*1.5
I_w0 = 0.5 * m_w0 * R_w0 ^ 2;
I_b0 = m_b0 * (0.48 ^ 2 + 0.145 ^ 2) / 12.0;
% I_b0 = 0.02;
I_z0 = 2 * m_b0 * (0.48 ^ 2 + 0.36 ^ 2) / 12.0;

A_jacobian = jacobian([d_s, dd_s, d_phi, dd_phi, d_theta_l_l, dd_theta_l_l, d_theta_l_r, dd_theta_l_r, d_theta_b, dd_theta_b], [s, d_s, phi, d_phi, theta_l_l, d_theta_l_l, theta_l_r, d_theta_l_r, theta_b, d_theta_b]);
B_jacobian = jacobian([d_s, dd_s, d_phi, dd_phi, d_theta_l_l, dd_theta_l_l, d_theta_l_r, dd_theta_l_r, d_theta_b, dd_theta_b], [T_lw_l, T_lw_r, T_bl_l, T_bl_r]);

A_jacobian = subs(A_jacobian, [g, R_w, R_l, l_c, m_w, m_l, m_b, I_w, I_b, I_z], [g0, R_w0, R_l0, l_c0, m_w0, m_l0, m_b0, I_w0, I_b0, I_z0]);
B_jacobian = subs(B_jacobian, [g, R_w, R_l, l_c, m_w, m_l, m_b, I_w, I_b, I_z], [g0, R_w0, R_l0, l_c0, m_w0, m_l0, m_b0, I_w0, I_b0, I_z0]);

use_discrete =false; % 控制是否使用离散系统
Ts = 0.01; % 采样时间（10ms）
C = eye(10); % 输出矩阵
D = zeros(10, 4); % 直接传递矩阵

for i = 1:num_iterations
    l_l0 = L_l0(i);
    l_r0 = L_r0(i);
    l_w_l0 = 0.5 * l_l0;
    l_w_r0 = 0.5 * l_r0;
    l_b_l0 = 0.5 * l_l0;
    l_b_r0 = 0.5 * l_r0;
    I_l_l0 = 0.2250*l_l0*l_l0+0.0341;
    I_l_r0 = 0.2250*l_r0*l_r0+0.0341;

    A = subs(A_jacobian, [l_l, l_r, l_w_l, l_w_r, l_b_l, l_b_r, I_l_l, I_l_r], [l_l0, l_r0, l_w_l0, l_w_r0, l_b_l0, l_b_r0, I_l_l0, I_l_r0]);
    A = double(A);
    B = subs(B_jacobian, [l_l, l_r, l_w_l, l_w_r, l_b_l, l_b_r, I_l_l, I_l_r], [l_l0, l_r0, l_w_l0, l_w_r0, l_b_l0, l_b_r0, I_l_l0, I_l_r0]);
    B = double(B);

    if use_discrete
        % 创建系统模型并离散化

        sys_continuous = ss(A, B, C, D);
        sys_discrete = c2d(sys_continuous, Ts);

        % 提取离散化后的系统矩阵
        A_d = sys_discrete.A;
        B_d = sys_discrete.B;

        % 计算离散LQR控制器
        K_all(:, :, i) = dlqr(A_d, B_d, Q, R);
    else
        % 计算连续系统LQR控制器
        K_all(:, :, i) = lqr(A, B, Q, R);
    end

end

[m, n, p] = size(K_all);
K_all = permute(K_all, [2, 1, 3]);
K_all = reshape(K_all, [m * n, p]);

% x1 = l_l0; x2 = l_r0;
% N = length(K11); % 数据点的数量
X = [ones(1, length(L_l0)); L_l0; L_l0 .^ 2; L_r0; L_r0 .^ 2; L_l0 .* L_r0];
theta = zeros(40, 6);
XX = (X * X'); % Pre-compute inverse once

% 最小二乘 
for j = 1:40
    theta(j, :) = (K_all(j, :) * X') / XX;
end

% 将theta矩阵分割成6个4x10的矩阵
matrices = cell(1, 6);

for i = 1:6
    matrices{i} = reshape(theta(:, i), [4, 10]);
end

% 将theta矩阵分割成6个float32_t[40]的数组
arrays = cell(1, 6);

for i = 1:6
    arrays{i} = reshape(theta(:, i), 1, []);
end

% 以C语言风格输出到txt文件中
filename = 'output.txt';
fileID = fopen(fullfile(pwd, filename), 'w');

fprintf(fileID, '// Q = diag([');
fprintf(fileID, '%g ', diag(Q));
fprintf(fileID, '])\n');
fprintf(fileID, '// R = diag([');
fprintf(fileID, '%g ', diag(R));
fprintf(fileID, '])\n\n');

for i = 1:6
    fprintf(fileID, 'static const float K_%d[40] {', i);
    fprintf(fileID, '%f, ', arrays{i}(1:end - 1));
    fprintf(fileID, '%f};\n\n', arrays{i}(end));
end
fprintf('%s \n', fullfile(pwd, filename));
fclose(fileID);
fprintf('complete');

%% 分析特定腿长下的系统特性（两腿长均为0.25m）
% 设定目标腿长
target_l_l = 0.4;
target_l_r = 0.4;

% 使用拟合结果计算对应的K矩阵
x = [1; target_l_l; target_l_l ^ 2; target_l_r; target_l_r ^ 2; target_l_l * target_l_r];
K_target = zeros(4, 10);

for j = 1:40
    row = ceil(j / 10);
    col = j - (row - 1) * 10;
    K_target(row, col) = sum(theta(j, :) .* x');
end

% 输出K矩阵
fprintf('两腿长均为%.2f米时的K矩阵：\n', target_l_l);
disp(K_target);

% 计算对应的A和B矩阵
l_l0 = target_l_l;
l_r0 = target_l_r;
l_w_l0 = 0.5 * l_l0;
l_w_r0 = 0.5 * l_r0;
l_b_l0 = 0.5 * l_l0;
l_b_r0 = 0.5 * l_r0;
I_l_l0 = m_l0 * ((l_w_l0 + l_b_l0) ^ 2 + 0.05 ^ 2) / 12.0;
I_l_r0 = m_l0 * ((l_w_r0 + l_b_r0) ^ 2 + 0.05 ^ 2) / 12.0;

A = subs(A_jacobian, [l_l, l_r, l_w_l, l_w_r, l_b_l, l_b_r, I_l_l, I_l_r], [l_l0, l_r0, l_w_l0, l_w_r0, l_b_l0, l_b_r0, I_l_l0, I_l_r0]);
A = double(A);
B = subs(B_jacobian, [l_l, l_r, l_w_l, l_w_r, l_b_l, l_b_r, I_l_l, I_l_r], [l_l0, l_r0, l_w_l0, l_w_r0, l_b_l0, l_b_r0, I_l_l0, I_l_r0]);
B = double(B);

if use_discrete
    % 创建系统模型并离散化
    sys_continuous = ss(A, B, C, D);
    sys_discrete = c2d(sys_continuous, Ts);

    % 提取离散化后的系统矩阵
    A_d = sys_discrete.A;
    B_d = sys_discrete.B;

    % 计算闭环离散系统矩阵
    A_cl_d = A_d - B_d * K_target;

    % 分析闭环离散系统特性
    eig_cl_d = eig(A_cl_d); % 计算闭环离散系统的特征值
    stable_d = all(abs(eig_cl_d) < 1); % 检查稳定性（离散系统特征值模值需<1）

    % 将离散系统特征值映射回连续域以计算时间常数和阻尼比
    eig_cl_c = log(eig_cl_d) / Ts; % 离散到连续域的映射

    time_constants = -1 ./ real(eig_cl_c(real(eig_cl_c) < 0)); % 时间常数(仅对稳定极点)
    damping_ratios = -real(eig_cl_c) ./ abs(eig_cl_c); % 阻尼比
    natural_freqs = abs(eig_cl_c); % 自然频率

    % 绘制离散系统闭环极点图
    figure;
    subplot(1, 2, 1);
    plot(real(eig_cl_d), imag(eig_cl_d), 'x', 'MarkerSize', 10);
    hold on;
    th = linspace(0, 2 * pi, 100);
    plot(cos(th), sin(th), '--k'); % 单位圆
    grid on;
    axis equal;
    title('离散系统闭环极点分布');
    xlabel('实部');
    ylabel('虚部');

    subplot(1, 2, 2);
    plot(real(eig_cl_c), imag(eig_cl_c), 'x', 'MarkerSize', 10);
    grid on;
    title('映射回连续域的闭环极点分布');
    xlabel('实部');
    ylabel('虚部');
    xline(0, '--r'); % 添加虚轴参考线

    % 显示系统特性
    fprintf('两腿长均为%.2f米的系统特性分析（离散系统, Ts=%.4fs）：\n', target_l_l, Ts);

    if stable_d
        fprintf('离散系统稳定\n');
    else
        fprintf('离散系统不稳定\n');
    end

    fprintf('闭环离散系统特征值：\n');

    for i = 1:length(eig_cl_d)
        fprintf('  λ%d_d = %.4f + %.4fi (|λ| = %.4f)\n', ...
            i, real(eig_cl_d(i)), imag(eig_cl_d(i)), abs(eig_cl_d(i)));
    end

    % 模拟闭环离散系统响应
    t_d = 0:Ts:5; % 时间范围
    % 初始状态：车体、左右腿有小幅度倾角扰动
    x0 = zeros(10, 1);
    x0(5) = 0.1; % theta_l_l - 左腿倾角
    x0(7) = 0.1; % theta_l_r - 右腿倾角
    x0(9) = 0.1; % theta_b - 车体倾角

    sys_cl_d = ss(A_cl_d, zeros(10, 4), eye(10), zeros(10, 4), Ts);
    [y_d, t_d] = initial(sys_cl_d, x0, t_d);

    % 绘制离散系统响应曲线
    figure;
    subplot(3, 1, 1);
    plot(t_d, y_d(:, 9), 'o-', 'LineWidth', 1.5, 'MarkerSize', 4); % theta_b - 车体倾角
    title('离散系统车体倾角响应');
    xlabel('时间 (s)');
    ylabel('角度 (rad)');
    grid on;

    subplot(3, 1, 2);
    plot(t_d, y_d(:, 5), 'o-', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
    plot(t_d, y_d(:, 7), 'x--', 'LineWidth', 1.5, 'MarkerSize', 4); % theta_l_l, theta_l_r - 左右腿倾角
    title('离散系统腿部倾角响应');
    xlabel('时间 (s)');
    ylabel('角度 (rad)');
    legend('左腿', '右腿');
    grid on;

    subplot(3, 1, 3);
    plot(t_d, y_d(:, 1), 'o-', 'LineWidth', 1.5, 'MarkerSize', 4); % s - 位置
    title('离散系统位置响应');
    xlabel('时间 (s)');
    ylabel('位置 (m)');
    grid on;

    % 计算离散系统的控制输入
    u_d = -K_target * y_d';
    figure;
    subplot(2, 1, 1);
    plot(t_d, u_d(1, :), 'o-', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
    plot(t_d, u_d(2, :), 'x--', 'LineWidth', 1.5, 'MarkerSize', 4);
    title('离散系统驱动轮转矩响应');
    xlabel('时间 (s)');
    ylabel('转矩 (N·m)');
    legend('左轮', '右轮');
    grid on;

    subplot(2, 1, 2);
    plot(t_d, u_d(3, :), 'o-', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
    plot(t_d, u_d(4, :), 'x--', 'LineWidth', 1.5, 'MarkerSize', 4);
    title('离散系统腿部转矩响应');
    xlabel('时间 (s)');
    ylabel('转矩 (N·m)');
    legend('左腿', '右腿');
    grid on;

    % 分析离散系统的性能指标
    settling_time = zeros(1, 10);
    overshoot = zeros(1, 10);

    for i = 1:10

        if any(y_d(:, i)) % 如果该状态变量有响应
            % 计算稳定时间（响应保持在最终值±2%范围内）
            final_val = y_d(end, i);
            settling_band = abs(0.02 * final_val);
            idx = find(abs(y_d(:, i) - final_val) <= settling_band, 1);

            if ~isempty(idx)
                settling_time(i) = t_d(idx);
            end

            % 计算超调量
            if final_val ~= 0
                peak = max(abs(y_d(:, i)));
                overshoot(i) = max(0, (peak - abs(final_val)) / abs(final_val) * 100);
            end

        end

    end

    fprintf('\n离散系统性能分析:\n');
    fprintf('车体倾角稳定时间: %.4f秒, 超调量: %.2f%%\n', settling_time(9), overshoot(9));
    fprintf('左腿倾角稳定时间: %.4f秒, 超调量: %.2f%%\n', settling_time(5), overshoot(5));
    fprintf('右腿倾角稳定时间: %.4f秒, 超调量: %.2f%%\n', settling_time(7), overshoot(7));

else

    % 计算闭环系统矩阵
    A_cl = A - B * K_target;
    % 分析闭环系统特性
    eig_cl = eig(A_cl); % 计算闭环系统的特征值
    stable = all(real(eig_cl) < 0); % 检查稳定性
    time_constants = -1 ./ real(eig_cl(real(eig_cl) < 0)); % 时间常数(仅对稳定极点)
    damping_ratios = -real(eig_cl) ./ abs(eig_cl); % 阻尼比
    natural_freqs = abs(eig_cl); % 自然频率

    % 绘制闭环极点图
    figure;
    plot(real(eig_cl), imag(eig_cl), 'x', 'MarkerSize', 10);
    grid on;
    title('闭环系统极点分布');
    xlabel('实部');
    ylabel('虚部');
    xline(0, '--r'); % 添加虚轴参考线

    % 显示系统特性
    fprintf('两腿长均为%.2f米的系统特性分析：\n', target_l_l);

    if stable
        fprintf('系统稳定\n');
    else
        fprintf('系统不稳定\n');
    end

    fprintf('闭环系统特征值：\n');

    for i = 1:length(eig_cl)
        fprintf('  λ%d = %.4f + %.4fi (|λ| = %.4f, ζ = %.4f)\n', ...
            i, real(eig_cl(i)), imag(eig_cl(i)), abs(eig_cl(i)), ...
            damping_ratios(i));
    end

    % 模拟闭环系统响应
    t = 0:0.01:5; % 时间范围
    % 初始状态：车体、左右腿有小幅度倾角扰动
    x0 = zeros(10, 1);
    x0(5) = 0.1; % theta_l_l - 左腿倾角
    x0(7) = 0.1; % theta_l_r - 右腿倾角
    x0(9) = 0.1; % theta_b - 车体倾角

    sys_cl = ss(A_cl, zeros(10, 4), eye(10), zeros(10, 4));
    [y, t] = initial(sys_cl, x0, t);

    % 绘制响应曲线
    figure;
    subplot(3, 1, 1);
    plot(t, y(:, 9), 'LineWidth', 1.5); % theta_b - 车体倾角
    title('车体倾角响应');
    xlabel('时间 (s)');
    ylabel('角度 (rad)');
    grid on;

    subplot(3, 1, 2);
    plot(t, y(:, 5), 'LineWidth', 1.5); hold on;
    plot(t, y(:, 7), '--', 'LineWidth', 1.5); % theta_l_l, theta_l_r - 左右腿倾角
    title('腿部倾角响应');
    xlabel('时间 (s)');
    ylabel('角度 (rad)');
    legend('左腿', '右腿');
    grid on;

    subplot(3, 1, 3);
    plot(t, y(:, 1), 'LineWidth', 1.5); % s - 位置
    title('位置响应');
    xlabel('时间 (s)');
    ylabel('位置 (m)');
    grid on;

    % 计算控制输入
    u = -K_target * y';
    figure;
    subplot(2, 1, 1);
    plot(t, u(1, :), 'LineWidth', 1.5); hold on;
    plot(t, u(2, :), '--', 'LineWidth', 1.5);
    title('驱动轮转矩响应');
    xlabel('时间 (s)');
    ylabel('转矩 (N·m)');
    legend('左轮', '右轮');
    grid on;

    subplot(2, 1, 2);
    plot(t, u(3, :), 'LineWidth', 1.5); hold on;
    plot(t, u(4, :), '--', 'LineWidth', 1.5);
    title('腿部转矩响应');
    xlabel('时间 (s)');
    ylabel('转矩 (N·m)');
    legend('左腿', '右腿');
    grid on;
end
