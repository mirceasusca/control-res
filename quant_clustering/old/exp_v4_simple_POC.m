%% ============================================================
%  Minimal canonical demo: state-feedback regulation over a quantized link
%
%  Plant (discrete-time LTI):
%      x_{k+1} = A x_k + B u_k
%      u_k     = -K * xhat_k
%
%  Sensor-to-controller link:
%      controller receives xhat_k = Q(x_k) (state quantized at sensor)
%
%  Compared quantizers (same bit budget, per channel):
%    1) Ideal:      xhat = x
%    2) Uniform:    xhat_i = Delta_i * round( sat(x_i)/Delta_i )
%    3) mu-law:     compand -> uniform in [-1,1] -> expand (per channel)
%    4) K-means 1D: per-channel Lloyd/K-means codebooks learned OFFLINE
%
%  Training: offline closed-loop rollouts (ideal link) from random x0
%  Testing:  fresh Monte Carlo set
%
%  Requires Statistics and Machine Learning Toolbox (kmeans).
%% ============================================================

clear; clc; close all;
rng(1);

Ts = 0.1;

%% -------------------- Plant --------------------
A = [0.97061827, 0.16090772;
     0.16090772, 0.85348702];
B = [0.2;
     1.0];
C = [1 0];  % just for plotting y = x1

n = size(A,1);

%% -------------------- LQR controller --------------------
theta = deg2rad(35);
Rrot  = [cos(theta) -sin(theta); sin(theta) cos(theta)];

Qeig = diag([15, 1]);
Q = Rrot * Qeig * Rrot.';
R = 0.15;

[K,S,~] = dlqr(A,B,Q,R);
Acl = A - B*K;

disp('K ='); disp(K);
disp('eig(A-BK) ='); disp(eig(Acl).');

%% -------------------- Simulation horizon & cost --------------------
Nsim = 250;
t = (0:Nsim)*Ts;

cost_fun = @(x,u) ...
    sum(arrayfun(@(k) x(:,k).'*Q*x(:,k) + u(:,k).'*R*u(:,k), 1:size(u,2))) ...
    + x(:,end).'*S*x(:,end);

%% -------------------- Bit budget (per sample, per state-vector) --------------------
bits_total = 8;            % try 6..12 to see differences
b1 = floor(bits_total/2);
b2 = bits_total - b1;
K1 = 2^b1;
K2 = 2^b2;
fprintf('\nBit budget: bits_total=%d (b1=%d, b2=%d)\n', bits_total, b1, b2);

%% ======================= TRAINING (offline, ideal link) =======================
M_train = 2000;

% mixture for x0 (dense near origin + wide coverage)
p_local = 0.85;
sigma_local = 0.35;
xmax_wide = 2.0;

Xtrain = zeros(M_train*(Nsim+1), n);
idx = 1;

disp('Generating training data (ideal link rollouts)...');
for tr = 1:M_train
    if mod(tr,500)==0, fprintf('  train traj %d / %d\n', tr, M_train); end

    x = zeros(n, Nsim+1);
    u = zeros(1, Nsim);

    x(:,1) = sample_x0(p_local, sigma_local, xmax_wide, n);

    for k = 1:Nsim
        u(k) = -K*x(:,k);                 % ideal state available in training
        x(:,k+1) = A*x(:,k) + B*u(k);
    end

    rows = idx:idx+Nsim;
    Xtrain(rows,:) = x.';                 % store as rows
    idx = idx + (Nsim+1);
end

%% ======================= Train K-means 1D codebooks =======================
opts = statset('MaxIter', 300, 'Display', 'final');
[~, C1] = kmeans(Xtrain(:,1), K1, 'Replicates', 6, 'Options', opts);
[~, C2] = kmeans(Xtrain(:,2), K2, 'Replicates', 6, 'Options', opts);
C1 = sort(C1); C2 = sort(C2);
q_kmeans = @(x) [nearest_1d(x(1), C1); nearest_1d(x(2), C2)];

%% ======================= Uniform quantizer =======================
xmin = -5; xmax = 5;                         % fixed full-scale
Delta1 = (xmax-xmin)/2^b1;
Delta2 = (xmax-xmin)/2^b2;
q_uniform = @(x) [Delta1*round(sat(x(1),xmin,xmax)/Delta1);
                  Delta2*round(sat(x(2),xmin,xmax)/Delta2)];

%% ======================= mu-law quantizer =======================
mu1 = 255; mu2 = 255;
Xmax1 = xmax; Xmax2 = xmax;
DeltaY1 = 2 / 2^b1;                          % uniform in [-1,1]
DeltaY2 = 2 / 2^b2;
q_mulaw = @(x) [mulaw_q(x(1), mu1, Xmax1, DeltaY1);
                mulaw_q(x(2), mu2, Xmax2, DeltaY2)];

%% ======================= Single episode (visual) =======================
x0 = [1.0; -0.5];

[x_id, u_id] = simulate_reg(A,B,K, x0, Nsim, @(x) x);
[x_uq, u_uq] = simulate_reg(A,B,K, x0, Nsim, q_uniform);
[x_mq, u_mq] = simulate_reg(A,B,K, x0, Nsim, q_mulaw);
[x_kq, u_kq] = simulate_reg(A,B,K, x0, Nsim, q_kmeans);

J_id = cost_fun(x_id,u_id);
J_uq = cost_fun(x_uq,u_uq);
J_mq = cost_fun(x_mq,u_mq);
J_kq = cost_fun(x_kq,u_kq);

fprintf('\n==== Single-episode regulation cost ====\n');
fprintf('Ideal      J = %.6f\n', J_id);
fprintf('Uniform    J = %.6f\n', J_uq);
fprintf('mu-law     J = %.6f\n', J_mq);
fprintf('K-means1D  J = %.6f\n', J_kq);

%% ======================= Monte Carlo test =======================
M_test = 500;
J_id_t = zeros(M_test,1);
J_uq_t = zeros(M_test,1);
J_mq_t = zeros(M_test,1);
J_kq_t = zeros(M_test,1);

disp('Running Monte Carlo test set...');
for tr = 1:M_test
    x0 = sample_x0(p_local, sigma_local, xmax_wide, n);

    [x1,u1] = simulate_reg(A,B,K, x0, Nsim, @(x) x);
    [x2,u2] = simulate_reg(A,B,K, x0, Nsim, q_uniform);
    [x3,u3] = simulate_reg(A,B,K, x0, Nsim, q_mulaw);
    [x4,u4] = simulate_reg(A,B,K, x0, Nsim, q_kmeans);

    J_id_t(tr) = cost_fun(x1,u1);
    J_uq_t(tr) = cost_fun(x2,u2);
    J_mq_t(tr) = cost_fun(x3,u3);
    J_kq_t(tr) = cost_fun(x4,u4);
end

dU = J_uq_t - J_id_t;
dM = J_mq_t - J_id_t;
dK = J_kq_t - J_id_t;

fprintf('\n==== Test-set excess cost (relative to ideal) ====\n');
fprintf('Uniform:  mean=%.4g, median=%.4g, 95%%=%.4g\n', mean(dU), median(dU), prctile(dU,95));
fprintf('mu-law:   mean=%.4g, median=%.4g, 95%%=%.4g\n', mean(dM), median(dM), prctile(dM,95));
fprintf('K-means:  mean=%.4g, median=%.4g, 95%%=%.4g\n', mean(dK), median(dK), prctile(dK,95));
fprintf('Win rate (K-means < mu-law): %.2f %%\n', 100*mean(J_kq_t < J_mq_t));
fprintf('Win rate (mu-law < uniform): %.2f %%\n', 100*mean(J_mq_t < J_uq_t));

%% ======================= Plots =======================
y_id = C*x_id; y_uq = C*x_uq; y_mq = C*x_mq; y_kq = C*x_kq;

figure('Name','Regulation output y=x1');
plot(t, y_id, 'LineWidth', 1.6); hold on;
plot(t, y_uq, 'LineWidth', 1.6);
plot(t, y_mq, 'LineWidth', 1.6);
plot(t, y_kq, 'LineWidth', 1.6);
grid on;
xlabel('Time (s)'); ylabel('y=x_1');
title('Regulation with quantized state feedback');
legend('Ideal','Uniform','\mu-law','K-means 1D','Location','best');

figure('Name','State trajectories');
subplot(2,1,1);
plot(t, x_id(1,:), 'LineWidth', 1.4); hold on;
plot(t, x_uq(1,:), 'LineWidth', 1.4);
plot(t, x_mq(1,:), 'LineWidth', 1.4);
plot(t, x_kq(1,:), 'LineWidth', 1.4);
grid on; ylabel('x_1');
legend('Ideal','Uniform','\mu-law','K-means','Location','best');
title('State x_1');

subplot(2,1,2);
plot(t, x_id(2,:), 'LineWidth', 1.4); hold on;
plot(t, x_uq(2,:), 'LineWidth', 1.4);
plot(t, x_mq(2,:), 'LineWidth', 1.4);
plot(t, x_kq(2,:), 'LineWidth', 1.4);
grid on; ylabel('x_2'); xlabel('Time (s)');
title('State x_2');

figure('Name','Control input');
stairs(t(1:end-1), u_id, 'LineWidth', 1.4); hold on;
stairs(t(1:end-1), u_uq, 'LineWidth', 1.4);
stairs(t(1:end-1), u_mq, 'LineWidth', 1.4);
stairs(t(1:end-1), u_kq, 'LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('u');
title('Control inputs');
legend('Ideal','Uniform','\mu-law','K-means','Location','best');

figure('Name','Excess cost histogram (test set)');
histogram(dU, 'Normalization','pdf'); hold on;
histogram(dM, 'Normalization','pdf');
histogram(dK, 'Normalization','pdf');
grid on;
xlabel('J - J_{ideal}'); ylabel('PDF');
title('Excess cost distribution');
legend('Uniform','\mu-law','K-means','Location','best');

%% ============================================================
% Local functions
%% ============================================================

function [x, u] = simulate_reg(A,B,K, x0, N, quantizer)
    n = size(A,1);
    x = zeros(n, N+1);
    u = zeros(1, N);
    x(:,1) = x0;
    for k = 1:N
        xhat = quantizer(x(:,k));
        u(k) = -K*xhat;
        x(:,k+1) = A*x(:,k) + B*u(k);
    end
end

function x0 = sample_x0(p_local, sigma_local, xmax_wide, n)
    if rand < p_local
        x0 = sigma_local * randn(n,1);
    else
        x0 = (2*rand(n,1)-1) * xmax_wide;
    end
end

function y = sat(x, xmin, xmax)
    y = min(max(x, xmin), xmax);
end

function q = mulaw_q(x, mu, Xmax, Dy)
    % mu-law compand -> uniform quantize in [-1,1] -> expand
    x = sat(x, -Xmax, Xmax);
    y = sign(x) .* log(1 + mu*abs(x)/Xmax) ./ log(1 + mu);   % in [-1,1]
    yq = Dy * round( sat(y,-1,1) / Dy );
    q  = sign(yq) .* (Xmax/mu) .* ((1 + mu).^abs(yq) - 1);
end

function v = nearest_1d(x, C)
    [~, i] = min(abs(C - x));
    v = C(i);
end
