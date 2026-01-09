%% Discrete-time 2nd-order system: LQR + (Uniform vs K-means) quantization + cost (IMPROVED + 1D/2D K-means)
% Included improvements:
% 1) Train/Test split (avoid optimistic results)
% 2) Cost-aware geometry via Riccati shaping: z = L x, L = chol(S)
% 3) Better initial-condition sampling (mixture: mostly near origin + some wide coverage)
% 4) Uniform baseline: fixed full-scale range [-5,5] (simple ADC-like quantizer)
% 5) K-means vector quantizer trained in z-space (2D)
% 6) Separate (decoupled) 1D K-means quantizers for x1 and x2 (two independent ADC channels)
%
% NOTE: Requires Statistics and Machine Learning Toolbox for kmeans

clear; clc; close all;
rng(1); % reproducibility

Ts = 0.1;

%% Plant
A = [1.2  0.4;
    -0.3  0.9];
B = [1; 0.5];

%% LQR
Q = diag([10, 4]);
R = 0.5;
[K,S,e] = dlqr(A,B,Q,R);
Acl = A - B*K;

disp('K_LQR ='); disp(K);
disp('Closed-loop eigenvalues (no quantization) ='); disp(eig(Acl));

%% Cost horizon (truncated + terminal term)
Ncost = 300;
cost_fun = @(x,u) sum(arrayfun(@(k) x(:,k).'*Q*x(:,k) + u(k)*R*u(k), 1:size(u,2))) ...
                  + x(:,end).'*S*x(:,end);

%% Baseline initial condition to visualize
x0 = [1.0; -0.5];
J_star = x0.'*S*x0;

%% ======================= 1) Training data generation =====================
M_train = 10000;  % training trajectories
M_test  = 1000;   % test trajectories (fresh)
Ntrain  = 200;    % steps per training trajectory

% Mixture sampling for initial conditions:
p_local     = 0.85;
sigma_local = 0.35;
xmax_wide   = 1.5;

% Riccati shaping (z = Lx)
L     = chol(S,'upper');   % S = L'*L
Linvt = inv(L);            % small 2x2, OK

Xdata = zeros(M_train*(Ntrain+1), 2);
Zdata = zeros(M_train*(Ntrain+1), 2);
idx = 1;

disp('Generating training data...');
for m = 1:M_train
    if mod(m,1000)==0, fprintf('  train traj %d / %d\n', m, M_train); end

    x = zeros(2, Ntrain+1);
    x(:,1) = sample_x0(p_local, sigma_local, xmax_wide);

    for k = 1:Ntrain
        u = -K*x(:,k);
        x(:,k+1) = A*x(:,k) + B*u;
    end

    rows = idx:idx+Ntrain;
    Xdata(rows,:) = x.';        % x samples (rows)
    Zdata(rows,:) = (L*x).';    % z samples (rows)
    idx = idx + (Ntrain+1);
end

%% ======================= 2) Train K-means quantizers =====================
% --- 2D (vector) K-means in z-space ---
Kq2D = 256;  % 8-bit index (if you transmit one codeword index)

opts = statset('MaxIter', 300, 'Display', 'final');
disp('Training 2D K-means in z = Lx space...');
[~, Cz] = kmeans(Zdata, Kq2D, 'Replicates', 8, 'Options', opts);

% Quantize in z and map back to x
kmeans2D_quantize = @(x) Linvt * nearest_centroid_2d((L*x).', Cz).';  % returns 2x1

% --- 1D K-means per channel (decoupled ADCs) ---
% Choose K1 and K2 (product codebook size = K1*K2; bits total = log2(K1)+log2(K2))
% Example "fair-ish" match to 2D Kq2D=256 (8 bits total): K1=16, K2=16 -> 4+4=8 bits, product=256.
K1 = 64;
K2 = 64;

disp('Training 1D K-means for z1 and z2 (decoupled, still Riccati-shaped)...');
[~, C1z] = kmeans(Zdata(:,1), K1, 'Replicates', 8, 'Options', opts);
[~, C2z] = kmeans(Zdata(:,2), K2, 'Replicates', 8, 'Options', opts);

C1z = sort(C1z);
C2z = sort(C2z);

% % Quantize each shaped component independently, then map back: xhat = L^{-1} zhat
% kmeans1D_quantize = @(x) kmeans1D_quantize_impl(x, L, Linvt, C1z, C2z);

% NOTE: MATLAB doesn't allow (L*x)(1) syntax; use a helper wrapper:
% kmeans1D_quantize = @(x) kmeans1D_quantize_impl(x, L, Linvt, C1z, C2z);
kmeans1D_quantize = @(x) kmeans1D_quantize_impl(x, L, Linvt, C1z, C2z);

%% ======================= 3) Define uniform quantizer (fixed range [-5,5]) ===
% Simple ADC-like quantizer: uniform between -5 and 5 with saturation
xmin = -5;
xmax =  5;

% 8 bits per channel (separate ADCs)
N_bits_dim1 = 14;
N_bits_dim2 = 14;

Delta1 = (xmax - xmin)/pow2(N_bits_dim1);
Delta2 = (xmax - xmin)/pow2(N_bits_dim2);

q_uniform = @(xi,Delta) Delta * round( sat(xi, xmin, xmax) / Delta );

uniform_quantize = @(x) [ ...
    q_uniform(x(1), Delta1); ...
    q_uniform(x(2), Delta2)  ...
];

%% ======================= 4) Single-trajectory demo (same x0) =============
[x_nq,  u_nq]              = simulate_cl(A,B,K,x0,Ncost, @(x) x);
[x_uq,  u_uq,  xhat_uq]    = simulate_cl(A,B,K,x0,Ncost, uniform_quantize);
[x_kq2, u_kq2, xhat_kq2]   = simulate_cl(A,B,K,x0,Ncost, kmeans2D_quantize);
[x_kq1, u_kq1, xhat_kq1]   = simulate_cl(A,B,K,x0,Ncost, kmeans1D_quantize);

J_nq  = cost_fun(x_nq,  u_nq);
J_uq  = cost_fun(x_uq,  u_uq);
J_kq2 = cost_fun(x_kq2, u_kq2);
J_kq1 = cost_fun(x_kq1, u_kq1);

fprintf('\n==== Cost comparison (single x0) ====\n');
fprintf('J_star (analytic optimum)            = %.16f\n', J_star);
fprintf('J_nq   (sim non-quantized)           = %.16f   |err|=%.3e\n', J_nq, abs(J_nq-J_star));
fprintf('J_uq   (uniform quantized, [-5,5])   = %.16f\n', J_uq);
fprintf('J_kq1  (1D K-means in z, decoupled)  = %.16f\n', J_kq1);
fprintf('J_kq2  (2D K-means in z, vector)     = %.16f\n', J_kq2);

%% ======================= 5) Monte-Carlo TEST evaluation ==================
disp('Running test-set Monte Carlo evaluation...');

Jstar_test = zeros(M_test,1);
Juq_test   = zeros(M_test,1);
Jkq1_test  = zeros(M_test,1);
Jkq2_test  = zeros(M_test,1);

for m = 1:M_test
    x0m = sample_x0(p_local, sigma_local, xmax_wide);

    [~,~] = simulate_cl(A,B,K,x0m,Ncost, @(x) x); %#ok<ASGLU> (just for symmetry)
    [xu,uu]   = simulate_cl(A,B,K,x0m,Ncost, uniform_quantize);
    [x1,u1]   = simulate_cl(A,B,K,x0m,Ncost, kmeans1D_quantize);
    [x2,u2]   = simulate_cl(A,B,K,x0m,Ncost, kmeans2D_quantize);

    Jstar_test(m) = x0m.'*S*x0m;
    Juq_test(m)   = cost_fun(xu,uu);
    Jkq1_test(m)  = cost_fun(x1,u1);
    Jkq2_test(m)  = cost_fun(x2,u2);
end

dU  = Juq_test  - Jstar_test;
dK1 = Jkq1_test - Jstar_test;
dK2 = Jkq2_test - Jstar_test;

fprintf('\n==== Test-set summary (M_test=%d) ====\n', M_test);
fprintf('Uniform: mean(J-J*)=%.4g, median=%.4g, 95%%=%.4g\n', mean(dU),  median(dU),  prctile(dU,95));
fprintf('1D-KM : mean(J-J*)=%.4g, median=%.4g, 95%%=%.4g\n', mean(dK1), median(dK1), prctile(dK1,95));
fprintf('2D-KM : mean(J-J*)=%.4g, median=%.4g, 95%%=%.4g\n', mean(dK2), median(dK2), prctile(dK2,95));
fprintf('Win rate (1D < Uniform): %.2f %%\n', 100*mean(Jkq1_test < Juq_test));
fprintf('Win rate (2D < Uniform): %.2f %%\n', 100*mean(Jkq2_test < Juq_test));
fprintf('Win rate (2D < 1D):      %.2f %%\n', 100*mean(Jkq2_test < Jkq1_test));

%% ======================= 6) Plots =======================================
t = (0:Ncost)*Ts;

figure('Name','Time simulations: Non-quantized vs Uniform vs 1D/2D K-means');
subplot(2,1,1);
plot(t, x_nq(1,:),  'LineWidth', 1.6); hold on;
plot(t, x_uq(1,:),  'LineWidth', 1.6);
plot(t, x_kq1(1,:), 'LineWidth', 1.6);
plot(t, x_kq2(1,:), 'LineWidth', 1.6);
grid on; ylabel('x_1');
title('x_1 time response');
legend('Non-quantized','Uniform [-5,5]','1D K-means (z, decoupled)','2D K-means (z, vector)','Location','best');

subplot(2,1,2);
plot(t, x_nq(2,:),  'LineWidth', 1.6); hold on;
plot(t, x_uq(2,:),  'LineWidth', 1.6);
plot(t, x_kq1(2,:), 'LineWidth', 1.6);
plot(t, x_kq2(2,:), 'LineWidth', 1.6);
grid on; xlabel('Time (s)'); ylabel('x_2');
title('x_2 time response');
legend('Non-quantized','Uniform [-5,5]','1D K-means (z, decoupled)','2D K-means (z, vector)','Location','best');

figure('Name','Phase portrait (true state)');
plot(x_nq(1,:),  x_nq(2,:),  '-o', 'LineWidth', 1.0, 'MarkerSize', 3); hold on;
plot(x_uq(1,:),  x_uq(2,:),  '-s', 'LineWidth', 1.0, 'MarkerSize', 3);
plot(x_kq1(1,:), x_kq1(2,:), '-^', 'LineWidth', 1.0, 'MarkerSize', 3);
plot(x_kq2(1,:), x_kq2(2,:), '-d', 'LineWidth', 1.0, 'MarkerSize', 3);
grid on; axis equal;
xlabel('x_1'); ylabel('x_2');
title('Phase portrait: Non-quantized vs Uniform vs 1D/2D K-means');
legend('Non-quantized','Uniform','1D K-means','2D K-means','Location','best');

figure('Name','Quantized state seen by controller');
plot(xhat_uq(1,:),  xhat_uq(2,:),  '-s', 'LineWidth', 1.0, 'MarkerSize', 3); hold on;
plot(xhat_kq1(1,:), xhat_kq1(2,:), '-^', 'LineWidth', 1.0, 'MarkerSize', 3);
plot(xhat_kq2(1,:), xhat_kq2(2,:), '-d', 'LineWidth', 1.0, 'MarkerSize', 3);
grid on; axis equal;
xlabel('\hat x_1'); ylabel('\hat x_2');
title('Quantized trajectories (controller measurement)');
legend('Uniform','1D K-means','2D K-means','Location','best');

figure('Name','Excess cost histogram on test set');
histogram(dU,  'Normalization','pdf'); hold on;
histogram(dK1, 'Normalization','pdf');
histogram(dK2, 'Normalization','pdf');
grid on;
xlabel('J - J^*'); ylabel('PDF');
title('Test-set excess cost');
legend('Uniform','1D K-means','2D K-means','Location','best');

%% ======================= Local functions ===============================
function x0 = sample_x0(p_local, sigma_local, xmax_wide)
    if rand < p_local
        x0 = sigma_local * randn(2,1);
    else
        x0 = (2*rand(2,1)-1) * xmax_wide;
    end
end

function y = sat(x, xmin, xmax)
    y = min(max(x, xmin), xmax);
end

function c = nearest_centroid_2d(xrow2, C)
    % xrow2: 1x2 row; C: Kx2; return nearest centroid row (1x2)
    dif = C - xrow2;
    d2 = sum(dif.^2, 2);
    [~, i] = min(d2);
    c = C(i,:);
end

function xhat = nearest_centroid_1d(x, C)
    % C sorted vector (Kx1); return nearest centroid scalar
    [~, i] = min(abs(C - x));
    xhat = C(i);
end

% function xhat = kmeans1D_quantize_impl(x, L, Linvt, C1z, C2z)
%     z = L*x;
%     zhat = [nearest_centroid_1d(z(1), C1z); nearest_centroid_1d(z(2), C2z)];
%     xhat = Linvt * zhat;
% end

function [x, u, xhat_hist] = simulate_cl(A,B,K,x0,N,quantizer)
    x = zeros(2, N+1);
    u = zeros(1, N);
    xhat_hist = zeros(2, N+1);

    x(:,1) = x0;
    xhat_hist(:,1) = quantizer(x(:,1));

    for k = 1:N
        xhat = quantizer(x(:,k));
        xhat_hist(:,k) = xhat;
        u(k) = -K*xhat;
        x(:,k+1) = A*x(:,k) + B*u(k);
    end
    xhat_hist(:,N+1) = quantizer(x(:,N+1));
end

function xhat = kmeans1D_quantize_impl(x, L, Linvt, C1z, C2z)
    z = L*x;
    zhat = [nearest_centroid_1d(z(1), C1z); nearest_centroid_1d(z(2), C2z)];
    xhat = Linvt * zhat;
end
