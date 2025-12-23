%% Discrete-time 2nd-order system: LQR + (Uniform vs K-means) quantization + cost
clear; clc; close all;

Ts = 0.1;

% Plant
A = [1.2  0.4;
    -0.3  0.9];
B = [1; 0.5];

% LQR weights
Q = diag([10, 2]);
R = 0.5;

% LQR gain and Riccati solution
[K,S,e] = dlqr(A,B,Q,R);
Acl = A - B*K;

disp('K_LQR ='); disp(K);
disp('Closed-loop eigenvalues (no quantization) ='); disp(eig(Acl));

%% Cost horizon
Ncost = 300;  % cost horizon (longer -> better approximation)
% Helper to compute truncated cost with terminal term:
cost_fun = @(x,u) sum(arrayfun(@(k) x(:,k).'*Q*x(:,k) + u(k)*R*u(k), 1:size(u,2))) ...
                  + x(:,end).'*S*x(:,end);

%% -------- Baseline initial condition to test (same as before) ----------
x0 = [1.0; -0.5];

% Analytic optimal infinite-horizon cost (non-quantized)
J_star = x0.'*S*x0;

%% ======================= 1) Train K-means quantizer =====================
N_bits_dim1 = 8;
N_bits_dim2 = 8;

% Monte-Carlo data generation (non-quantized closed-loop to get state cloud)
M      = 5000;    % number of random initial conditions (increase for better codebooks)
Ntrain = 200;     % steps per trajectory for training data
xmax   = 1.0;    % initial condition sampling range [-xmax, xmax]

Xdata = zeros(M*(Ntrain+1), 2);
idx = 1;

for m = 1:M
    if mod(m,1000) == 0
        disp(m)
    end
    x = zeros(2, Ntrain+1);
    x(:,1) = (2*rand(2,1)-1)*xmax;   % uniform random initial condition

    for k = 1:Ntrain
        u = -K*x(:,k);
        x(:,k+1) = A*x(:,k) + B*u;
    end

    Xdata(idx:idx+Ntrain, :) = x.';  % store as rows
    idx = idx + (Ntrain+1);
end

% K-means settings
Kq = 1024;  % number of clusters / codewords (increase -> finer quantization)
opts = statset('MaxIter', 300, 'Display', 'iter');

% Train K-means codebook (centroids)
% Using multiple replicates helps avoid bad local minima
[cluster_id, C] = kmeans(Xdata, Kq, ...
    'Replicates', 10, ...
    'Options', opts);

% C is Kq-by-2, each row is a centroid.

% Quantizer (vector quantizer): map x -> nearest centroid
kmeans_quantize = @(x) nearest_centroid(x, C);
% kmeans_quantize = @(x) nearest_centroid(x, C, S);

%% ======================= 2) Define uniform quantizer ====================
Delta1 = (5-(-5))/pow2(N_bits_dim1);
Delta2 = (5-(-5))/pow2(N_bits_dim2);
qfun_scalar = @(x,Delta) Delta .* round(x./Delta);
uniform_quantize = @(x) [qfun_scalar(x(1),Delta1); qfun_scalar(x(2),Delta2)];

%% ======================= 3) Simulators =================================
% 3a) Non-quantized closed-loop: u = -K x
[x_nq, u_nq] = simulate_cl(A,B,K,x0,Ncost, @(x) x);

% 3b) Uniform-quantized measurement: u = -K q(x)
[x_uq, u_uq, xhat_uq] = simulate_cl(A,B,K,x0,Ncost, uniform_quantize);

% 3c) K-means quantized measurement: u = -K q_kmeans(x)
[x_kq, u_kq, xhat_kq] = simulate_cl(A,B,K,x0,Ncost, kmeans_quantize);

%% ======================= 4) Costs ======================================
J_nq = cost_fun(x_nq, u_nq);
J_uq = cost_fun(x_uq, u_uq);
J_kq = cost_fun(x_kq, u_kq);

fprintf('\n==== Cost comparison ====\n');
fprintf('J_star (analytic optimum)        = %.10f\n', J_star);
fprintf('J_nq   (sim non-quantized)       = %.10f   |err|=%.3e\n', J_nq, abs(J_nq-J_star));
fprintf('J_uq   (uniform quantized)       = %.10f\n', J_uq);
fprintf('J_kq   (K-means quantized)       = %.10f\n', J_kq);

%% ======================= 5) Plots ======================================
t = (0:Ncost)*Ts;

% Time responses (true states)
figure('Name','State vs time (true state) - compare quantizers');
subplot(2,1,1);
plot(t, x_nq(1,:), 'LineWidth', 1.6); hold on;
plot(t, x_uq(1,:), 'LineWidth', 1.6);
plot(t, x_kq(1,:), 'LineWidth', 1.6);
grid on; ylabel('x_1');
legend('Non-quantized','Uniform','K-means','Location','best');
title(sprintf('True state response (Kq=%d, M=%d, Ntrain=%d)', Kq, M, Ntrain));

subplot(2,1,2);
plot(t, x_nq(2,:), 'LineWidth', 1.6); hold on;
plot(t, x_uq(2,:), 'LineWidth', 1.6);
plot(t, x_kq(2,:), 'LineWidth', 1.6);
grid on; xlabel('Time (s)'); ylabel('x_2');
legend('Non-quantized','Uniform','K-means','Location','best');

% Phase portraits (true state)
figure('Name','Phase portrait (true state) - compare quantizers');
plot(x_nq(1,:), x_nq(2,:), '-o', 'LineWidth', 1.2, 'MarkerSize', 3); hold on;
plot(x_uq(1,:), x_uq(2,:), '-s', 'LineWidth', 1.2, 'MarkerSize', 3);
plot(x_kq(1,:), x_kq(2,:), '-d', 'LineWidth', 1.2, 'MarkerSize', 3);
grid on; axis equal;
xlabel('x_1'); ylabel('x_2');
title('True-state phase portrait: Non-quantized vs Uniform vs K-means');
legend('Non-quantized','Uniform','K-means','Location','best');

% Show codebook (centroids) and training data (optional, can be heavy)
figure('Name','K-means codebook in state space');
plot(Xdata(:,1), Xdata(:,2), '.', 'MarkerSize', 4); hold on;
plot(C(:,1), C(:,2), 'kx', 'LineWidth', 2, 'MarkerSize', 10);
grid on; axis equal;
xlabel('x_1'); ylabel('x_2');
title('Training state cloud and K-means centroids');
legend('Training samples','Centroids','Location','best');

% Quantized-state trajectories (what controller "sees")
figure('Name','Quantized state seen by controller (Uniform vs K-means)');
plot(xhat_uq(1,:), xhat_uq(2,:), '-s', 'LineWidth', 1.2, 'MarkerSize', 3); hold on;
plot(xhat_kq(1,:), xhat_kq(2,:), '-d', 'LineWidth', 1.2, 'MarkerSize', 3);
grid on; axis equal;
xlabel('xhat_1'); ylabel('xhat_2');
title('Quantized trajectories (controller measurement)');
legend('Uniform xhat','K-means xhat','Location','best');

%% ============================================================
% Time-domain comparison: Non-quantized vs Uniform vs K-means
% ============================================================
t = (0:Ncost)*Ts;

figure('Name','Time simulations: Non-quantized vs Uniform vs K-means');

subplot(2,1,1);
plot(t, x_nq(1,:), 'LineWidth', 1.8); hold on;
plot(t, x_uq(1,:), 'LineWidth', 1.8);
plot(t, x_kq(1,:), 'LineWidth', 1.8);
grid on;
ylabel('x_1');
title('State x_1 time response');
legend('Non-quantized','Uniform quantization','K-means quantization', ...
       'Location','best');

subplot(2,1,2);
plot(t, x_nq(2,:), 'LineWidth', 1.8); hold on;
plot(t, x_uq(2,:), 'LineWidth', 1.8);
plot(t, x_kq(2,:), 'LineWidth', 1.8);
grid on;
xlabel('Time (s)');
ylabel('x_2');
title('State x_2 time response');
legend('Non-quantized','Uniform quantization','K-means quantization', ...
       'Location','best');

%% ======================= Local functions ===============================
function xhat = nearest_centroid(x, C)
    % x: 2x1, C: Kq x 2
    % Return centroid (2x1) closest in Euclidean distance
    dif = C - x.';                 % Kq x 2
    d2  = sum(dif.^2, 2);          % Kq x 1
    [~, i] = min(d2);
    xhat = C(i,:).';
end

% function xhat = nearest_centroid(x, C, S)
%     W = S;  % or (Q + K'*R*K)
% 
%     % In nearest centroid:
%     dif = C - x.';                 % K x 2
%     d2  = sum((dif*W).*dif, 2);     % quadratic form row-wise
%     [~, i] = min(d2);
%     xhat = C(i,:).';
% end

function [x, u, xhat_hist] = simulate_cl(A,B,K,x0,N,quantizer)
    % Closed-loop sim with quantized measurement:
    % x[k+1] = A x[k] + B u[k], u[k] = -K * xhat[k], xhat[k]=quantizer(x[k])
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
