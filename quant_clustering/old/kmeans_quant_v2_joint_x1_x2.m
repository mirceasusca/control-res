%% Discrete-time 2nd-order system: LQR + (Uniform vs K-means) quantization + cost (IMPROVED)
% Improvements included:
% 1) Train/Test split (avoid optimistic results)
% 2) Cost-aware geometry via Riccati shaping: z = L x, L = chol(S)
% 3) Better initial-condition sampling (mixture: mostly near origin + some wide coverage)
% 4) Fairer uniform baseline: full-scale range learned from training data percentiles + saturation
% 5) Optional cost-aware nearest-centroid metric (W = I in z-space already; also provided in x-space)
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
N_bits_dim1 = 8;
N_bits_dim2 = 8;

M_train = 10000;   % training trajectories
M_test  = 1000;   % test trajectories (fresh)
Ntrain  = 200;    % steps per training trajectory

% Mixture sampling for initial conditions:
% - with prob p_local: Gaussian near origin (focus near limit-cycle region)
% - otherwise: uniform box (coverage of larger transients)
p_local = 0.85;
sigma_local = 0.35;  % tune: smaller -> more mass near origin
xmax_wide   = 1.5;   % wide coverage box

% Riccati shaping: z = L x makes LQR value contours "round-ish"
% S is SPD => chol exists
L = chol(S,'upper');     % S = L'*L
% L = eye(size(S));
Linvt = inv(L);          % for mapping centroids back (small 2x2, OK)

% Collect training state cloud (in x and in z)
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
    Xdata(rows,:) = x.';           % rows are samples
    Zdata(rows,:) = (L*x).';       % shaped coordinates
    idx = idx + (Ntrain+1);
end

%%
% Learn an effective "working range" from data (fairer uniform baseline)
% Use percentiles (e.g., 99.5%) per dimension in x-space; then saturate outside.
pct = 99.5;
guard = 3.0;
xmax_eff = guard*prctile(abs(Xdata), pct, 1);   % 1x2
xmax_eff = max(xmax_eff, 1e-6);           % safety
xmin_eff = -xmax_eff;

fprintf('Effective uniform range from training (± percentile %.1f%%):\n', pct);
fprintf('  x1 in [%.3f, %.3f], x2 in [%.3f, %.3f]\n', xmin_eff(1), xmax_eff(1), xmin_eff(2), xmax_eff(2));

% % Uniform steps (per-dimension) over learned range
% Delta1 = (xmax_eff(1)-xmin_eff(1))/pow2(N_bits_dim1);
% Delta2 = (xmax_eff(2)-xmin_eff(2))/pow2(N_bits_dim2);

%% ======================= 2) Train K-means quantizer =====================
% K-means codebook size
% Kq = 1024;   % 1024 clusters = 10-bit index if you were transmitting it
% Kq = 512;   % 9-bit index if you were transmitting it
Kq = 256;   % 8-bit index if you were transmitting it
% Kq = 128;   % 8-bit index if you were transmitting it

% Train in shaped coordinates z (often better for LQR)
opts = statset('MaxIter', 300, 'Display', 'final');
disp('Training K-means in z = Lx space...');
[~, Cz] = kmeans(Zdata, Kq, 'Replicates', 8, 'Options', opts);

% Map centroids back to x space: x = L^{-1} z
Cx = (Linvt * Cz.').';  % Kq x 2

% Quantizer variants:
kmeans_quantize_x  = @(x) nearest_centroid_euclid(x, Cx);      % in x-space
kmeans_quantize_z  = @(x) Linvt * nearest_centroid_euclid(L*x, Cz).'; % quantize in z then map back

% Recommended: quantize in z, then map back
kmeans_quantize = @(x) Linvt * nearest_centroid_euclid(L*x, Cz).';

% %% ======================= 3) Define uniform quantizer (with saturation) ===
% qfun_scalar = @(x,Delta) Delta .* round(x./Delta);
% 
% uniform_quantize = @(x) [ ...
%     qfun_scalar(sat(x(1), xmin_eff(1), xmax_eff(1)), Delta1); ...
%     qfun_scalar(sat(x(2), xmin_eff(2), xmax_eff(2)), Delta2)  ...
% ];

%% ======================= 3) Define uniform quantizer (fixed range [-5,5]) ===
xmin = -5;
xmax =  5;

Delta1 = (xmax - xmin)/pow2(log2(Kq));
Delta2 = (xmax - xmin)/pow2(log2(Kq));

% Uniform mid-tread quantizer over [-5,5] with saturation
% (finite-bit quantizer -> saturation is inherent)
q_uniform = @(xi,Delta) Delta * round( sat(xi, xmin, xmax) / Delta );

uniform_quantize = @(x) [ ...
    q_uniform(x(1), Delta1); ...
    q_uniform(x(2), Delta2)  ...
];

%% ======================= 4) Single-trajectory demo (same x0) =============
[x_nq, u_nq] = simulate_cl(A,B,K,x0,Ncost, @(x) x);
[x_uq, u_uq, xhat_uq] = simulate_cl(A,B,K,x0,Ncost, uniform_quantize);
[x_kq, u_kq, xhat_kq] = simulate_cl(A,B,K,x0,Ncost, kmeans_quantize);

J_nq = cost_fun(x_nq, u_nq);
J_uq = cost_fun(x_uq, u_uq);
J_kq = cost_fun(x_kq, u_kq);

fprintf('\n==== Cost comparison (single x0) ====\n');
fprintf('J_star (analytic optimum)        = %.16f\n', J_star);
fprintf('J_nq   (sim non-quantized)       = %.16f   |err|=%.3e\n', J_nq, abs(J_nq-J_star));
fprintf('J_uq   (uniform quantized)       = %.16f\n', J_uq);
fprintf('J_kq   (K-means shaped quantized)= %.16f\n', J_kq);

%% ======================= 5) Monte-Carlo TEST evaluation ==================
disp('Running test-set Monte Carlo evaluation...');

Jstar_test = zeros(M_test,1);
Juq_test   = zeros(M_test,1);
Jkq_test   = zeros(M_test,1);

for m = 1:M_test
    x0m = sample_x0(p_local, sigma_local, xmax_wide);

    [x1,u1] = simulate_cl(A,B,K,x0m,Ncost, @(x) x);
    [x2,u2] = simulate_cl(A,B,K,x0m,Ncost, uniform_quantize);
    [x3,u3] = simulate_cl(A,B,K,x0m,Ncost, kmeans_quantize);

    Jstar_test(m) = x0m.'*S*x0m;      % true infinite-horizon optimum (non-quantized)
    Juq_test(m)   = cost_fun(x2,u2);
    Jkq_test(m)   = cost_fun(x3,u3);
end

dU = Juq_test - Jstar_test;
dK = Jkq_test - Jstar_test;

fprintf('\n==== Test-set summary (M_test=%d) ====\n', M_test);
fprintf('Uniform: mean(J-J*) = %.4g, median = %.4g, 95%% = %.4g\n', mean(dU), median(dU), prctile(dU,95));
fprintf('Kmeans : mean(J-J*) = %.4g, median = %.4g, 95%% = %.4g\n', mean(dK), median(dK), prctile(dK,95));
fprintf('Win rate (Kmeans < Uniform): %.2f %%\n', 100*mean(Jkq_test < Juq_test));

%% ======================= 6) Plots =======================================
t = (0:Ncost)*Ts;

% Time simulations of all three cases (true state)
figure('Name','Time simulations: Non-quantized vs Uniform vs K-means');
subplot(2,1,1);
plot(t, x_nq(1,:), 'LineWidth', 1.8); hold on;
plot(t, x_uq(1,:), 'LineWidth', 1.8);
plot(t, x_kq(1,:), 'LineWidth', 1.8);
grid on; ylabel('x_1');
title('State x_1 time response');
legend('Non-quantized','Uniform (sat+learned range)','K-means (Riccati-shaped)','Location','best');

subplot(2,1,2);
plot(t, x_nq(2,:), 'LineWidth', 1.8); hold on;
plot(t, x_uq(2,:), 'LineWidth', 1.8);
plot(t, x_kq(2,:), 'LineWidth', 1.8);
grid on; xlabel('Time (s)'); ylabel('x_2');
title('State x_2 time response');
legend('Non-quantized','Uniform (sat+learned range)','K-means (Riccati-shaped)','Location','best');

% Phase portrait (true state)
figure('Name','Phase portrait (true state) - compare');
plot(x_nq(1,:), x_nq(2,:), '-o', 'LineWidth', 1.2, 'MarkerSize', 3); hold on;
plot(x_uq(1,:), x_uq(2,:), '-s', 'LineWidth', 1.2, 'MarkerSize', 3);
plot(x_kq(1,:), x_kq(2,:), '-d', 'LineWidth', 1.2, 'MarkerSize', 3);
grid on; axis equal;
xlabel('x_1'); ylabel('x_2');
title('True-state phase portrait: Non-quantized vs Uniform vs K-means');
legend('Non-quantized','Uniform','K-means','Location','best');

% Quantized trajectories (what controller sees)
figure('Name','Quantized state seen by controller');
plot(xhat_uq(1,:), xhat_uq(2,:), '-s', 'LineWidth', 1.2, 'MarkerSize', 3); hold on;
plot(xhat_kq(1,:), xhat_kq(2,:), '-d', 'LineWidth', 1.2, 'MarkerSize', 3);
grid on; axis equal;
xlabel('\hat x_1'); ylabel('\hat x_2');
title('Quantized trajectories (controller measurement)');
legend('Uniform \hat x','K-means \hat x (shaped)','Location','best');

% Codebook visualization (optional; plot a subsample of training data for speed)
figure('Name','Codebook in shaped coordinates z = Lx');
sub = 1:50:size(Zdata,1); % thin points for plotting
plot(Zdata(sub,1), Zdata(sub,2), '.', 'MarkerSize', 4); hold on;
plot(Cz(:,1), Cz(:,2), 'kx', 'LineWidth', 1.8, 'MarkerSize', 9);
grid on; axis equal;
xlabel('z_1'); ylabel('z_2');
title('Training cloud (subsampled) and K-means centroids in z-space');
legend('Training samples','Centroids','Location','best');

% Histogram of excess cost on test set
figure('Name','Excess cost histogram on test set');
histogram(dU, 'Normalization','pdf'); hold on;
histogram(dK, 'Normalization','pdf');
grid on;
xlabel('J - J^*');
ylabel('PDF');
title('Test-set excess cost: Uniform vs K-means');
legend('Uniform','K-means','Location','best');

%% ======================= Local functions ===============================
function x0 = sample_x0(p_local, sigma_local, xmax_wide)
    if rand < p_local
        x0 = sigma_local * randn(2,1);           % near-origin Gaussian
    else
        x0 = (2*rand(2,1)-1) * xmax_wide;        % wide uniform box
    end
end

function y = sat(x, xmin, xmax)
    y = min(max(x, xmin), xmax);
end

function c = nearest_centroid_euclid(xrow2, C)
    % xrow2: 2x1 OR 1x2 acceptable; C: K x 2; returns 1x2 centroid row
    if iscolumn(xrow2), xrow2 = xrow2.'; end
    dif = C - xrow2;                 % K x 2
    d2  = sum(dif.^2, 2);            % K x 1
    [~, i] = min(d2);
    c = C(i,:);                      % 1x2
end

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
