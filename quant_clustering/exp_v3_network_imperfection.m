%% ============================================================
%  Canonical benchmark: NCS tracking with packet drops + hold-last
%     u = -K*xhat + F*r
%
%  Sensor-to-controller link: transmit a reconstructed state xhat.
%  Quantization is applied to deviation state dev = x - x*(r).
%
%  Network imperfection (single, canonical): i.i.d. packet drops with hold-last
%     if drop -> controller uses last received xhat (ZOH / hold)
%
%  Compared encoders / quantizers (same bit budget):
%     1) Ideal link (no quantization):      xhat = x
%     2) Uniform (dev, per-channel):        xhat = x* + qu(dev)
%     3) Log (mu-law dev, per-channel):     xhat = x* + qlog(dev)
%     4) K-means 1D shaped (dev,z):         z=L*dev, K-means each zi, devhat = L^{-1} zhat
%     5) K-means 1D raw (dev,x):            K-means each devi directly
%
%  Training is OFFLINE on closed-loop rollouts UNDER THE SAME DROP MODEL.
%  Testing is a paired Monte Carlo set (same drop sequence per scheme per trial).
%
%  Requires Statistics and Machine Learning Toolbox (kmeans).
%% ============================================================

clear; clc; close all;
rng(1);

Ts = 0.1;

%% -------------------- Network imperfection: packet drops --------------------
p_drop = 0.2;          % 0.05..0.3 (try 0.2 to make effect visible)
sigma_w = 0.005;       % small process noise
fprintf('Network: i.i.d. drops p_drop=%.2f, hold-last at controller\n', p_drop);

%% -------------------- Plant (A,B,C) --------------------
A = [0.97061827, 0.16090772;
     0.16090772, 0.85348702];

B = [0.2;
     1.0];

C = [1 0];   % track x1

n = size(A,1);
m = size(B,2);
p = size(C,1);

%% -------------------- LQR (K, S) --------------------
theta = deg2rad(35);               % rotation of "important direction"
Rrot = [cos(theta) -sin(theta);
        sin(theta)  cos(theta)];

Qeig = diag([10an, 1]);             % huge penalty in one direction
Q = Rrot * Qeig * Rrot.';          % rotated Q (off-diagonal)

R = 0.15;
[K,S,~] = dlqr(A,B,Q,R);
Acl = A - B*K;

disp('K ='); disp(K);
disp('eig(A-BK) ='); disp(eig(Acl).');

% Riccati shaping: S = L'*L
L = chol(S,'upper');
Linvt = inv(L);

%% -------------------- Prefilter F (zero steady-state error for constant r) --------------------
I = eye(n);
Gdc = C * ((I - Acl)\B);  % p x m
if rank(Gdc) < min(size(Gdc))
    warning('Prefilter may be rank-deficient; using pinv.');
    F = pinv(Gdc);
else
    F = inv(Gdc);
end
disp('F ='); disp(F);

%% -------------------- Step reference signal (piecewise constant) --------------------
r_levels = [-1.2,  1.2, -0.6,  0.6, -1.2,  0];
t_switch  = [0,    6,    12,   18,   24,   30];
Nsim = 360;                                     % 36s at Ts=0.1
t = (0:Nsim)*Ts;

r = zeros(p, Nsim+1);
for i = 1:numel(r_levels)
    k0 = round(t_switch(i)/Ts) + 1;
    if i < numel(r_levels)
        k1 = round(t_switch(i+1)/Ts);
        r(:,k0:k1) = r_levels(i);
    else
        r(:,k0:end) = r_levels(i);
    end
end

%% -------------------- Equilibrium map (x*(r), u*(r)) --------------------
eq_solve = @(rval) equilibrium_from_r(A,B,C,rval);

xstar_levels = zeros(n, numel(r_levels));
ustar_levels = zeros(m, numel(r_levels));
for i = 1:numel(r_levels)
    [xs, us] = eq_solve(r_levels(i));
    xstar_levels(:,i) = xs;
    ustar_levels(:,i) = us;
end

mode_of_r = @(rk) find(abs(r_levels - rk(1)) == min(abs(r_levels - rk(1))), 1, 'first');

xstar = zeros(n, Nsim+1);
ustar = zeros(m, Nsim+1);
for k = 1:Nsim+1
    idx = mode_of_r(r(:,k));
    xstar(:,k) = xstar_levels(:,idx);
    ustar(:,k) = ustar_levels(:,idx);
end

%% -------------------- Tracking cost in deviation coords --------------------
cost_fun_tracking = @(x,u,xs,us) ...
    sum(arrayfun(@(k) (x(:,k)-xs(:,k)).'*Q*(x(:,k)-xs(:,k)) + (u(:,k)-us(:,k)).'*R*(u(:,k)-us(:,k)), 1:size(u,2))) ...
    + (x(:,end)-xs(:,end)).'*S*(x(:,end)-xs(:,end));

%% ======================= BIT BUDGET (fair comparisons) =======================
bits_total = 8;         % IMPORTANT: tighten bits to expose differences (try 8 or 10)
b1 = floor(bits_total/2);
b2 = bits_total - b1;
fprintf('\nBit budget: bits_total=%d (b1=%d, b2=%d)\n', bits_total, b1, b2);

%% ======================= TRAINING DATA (offline, WITH DROPS) =======================
M_train = 2000;
Ntrain  = Nsim;

p_local = 0.85;
sigma_local = 0.35;
xmax_wide = 1.5;

Zdata = zeros(M_train*(Ntrain+1), n);   % shaped deviations z = L*(x-x*)
Xdev  = zeros(M_train*(Ntrain+1), n);   % raw deviations (optional baseline)
idx = 1;

disp('Generating training data under packet drops + hold-last...');
for tr = 1:M_train
    if mod(tr,500)==0, fprintf('  train traj %d / %d\n', tr, M_train); end

    x = zeros(n, Ntrain+1);
    u = zeros(m, Ntrain);

    xdev0 = sample_dev(p_local, sigma_local, xmax_wide, n);
    x(:,1) = xstar(:,1) + xdev0;

    % Controller holds deviation estimate (training uses held dev without quantization)
    devhat_prev = x(:,1) - xstar(:,1);

    for k = 1:Ntrain
        dev = x(:,k) - xstar(:,k);

        % i.i.d. drop: hold last
        if rand < p_drop
            devhat = devhat_prev;
        else
            devhat = dev;
        end
        devhat_prev = devhat;

        u(:,k) = -K*devhat + ustar(:,k);
        x(:,k+1) = A*x(:,k) + B*u(:,k) + sigma_w*randn(n,1);
    end

    dev_traj = (x - xstar(:,1:Ntrain+1));   % true deviations
    z_traj   = L * dev_traj;

    rows = idx:idx+Ntrain;
    Xdev(rows,:)  = dev_traj.';   % store as rows
    Zdata(rows,:) = z_traj.';
    idx = idx + (Ntrain+1);
end

%% ======================= Train 1D K-means codebooks =======================
opts = statset('MaxIter', 300, 'Display', 'final');

K1 = 2^b1;
K2 = 2^b2;

% --- shaped in z-space (recommended) ---
[~, C1z] = kmeans(Zdata(:,1), K1, 'Replicates', 6, 'Options', opts);
[~, C2z] = kmeans(Zdata(:,2), K2, 'Replicates', 6, 'Options', opts);
C1z = sort(C1z);
C2z = sort(C2z);
q_kmeans1D_shaped = @(xk, xs_k) quantize_dev_kmeans1D_z(xk, xs_k, L, Linvt, C1z, C2z);

% --- raw in x-space (baseline) ---
[~, C1x] = kmeans(Xdev(:,1), K1, 'Replicates', 6, 'Options', opts);
[~, C2x] = kmeans(Xdev(:,2), K2, 'Replicates', 6, 'Options', opts);
C1x = sort(C1x);
C2x = sort(C2x);
q_kmeans1D_raw = @(xk, xs_k) quantize_dev_kmeans1D_x(xk, xs_k, C1x, C2x);

%% ======================= Uniform baseline (deviation quantization) =======================
xmin = -5; xmax = 5;                 % fixed dev range
Delta1 = (xmax-xmin)/2^b1;
Delta2 = (xmax-xmin)/2^b2;
q_uniform = @(xk, xs_k) quantize_dev_uniform(xk, xs_k, xmin, xmax, Delta1, Delta2);

%% ======================= Logarithmic (mu-law) baseline =======================
mu1 = 255; mu2 = 255;
Xmax1 = xmax; Xmax2 = xmax;

DeltaY1 = 2 / 2^b1;      % companded y in [-1,1]
DeltaY2 = 2 / 2^b2;
q_log = @(xk, xs_k) quantize_dev_mulaw(xk, xs_k, mu1, mu2, Xmax1, Xmax2, DeltaY1, DeltaY2);

%% ======================= Single-episode demo (paired drops) =======================
x0 = [1.0; -0.5];
seed = 5;

rng(seed);
drop_seq = rand(1,Nsim) < p_drop;   % common dropout pattern for all schemes

[x_id, u_id] = simulate_tracking_drop(A,B,K,F, x0, r, @(xk,xs_k) xk,            xstar, ustar, sigma_w, drop_seq, seed+100);
[x_uq, u_uq] = simulate_tracking_drop(A,B,K,F, x0, r, q_uniform,                xstar, ustar, sigma_w, drop_seq, seed+100);
[x_lq, u_lq] = simulate_tracking_drop(A,B,K,F, x0, r, q_log,                    xstar, ustar, sigma_w, drop_seq, seed+100);
[x_kz, u_kz] = simulate_tracking_drop(A,B,K,F, x0, r, q_kmeans1D_shaped,         xstar, ustar, sigma_w, drop_seq, seed+100);
[x_kx, u_kx] = simulate_tracking_drop(A,B,K,F, x0, r, q_kmeans1D_raw,            xstar, ustar, sigma_w, drop_seq, seed+100);

J_id = cost_fun_tracking(x_id, u_id, xstar, ustar);
J_uq = cost_fun_tracking(x_uq, u_uq, xstar, ustar);
J_lq = cost_fun_tracking(x_lq, u_lq, xstar, ustar);
J_kz = cost_fun_tracking(x_kz, u_kz, xstar, ustar);
J_kx = cost_fun_tracking(x_kx, u_kx, xstar, ustar);

fprintf('\n==== Single-episode tracking cost (drops+hold, deviation LQR) ====\n');
fprintf('Ideal link (no quantization)     J = %.6f\n', J_id);
fprintf('Uniform (dev)                   J = %.6f\n', J_uq);
fprintf('Log (mu-law dev)                J = %.6f\n', J_lq);
fprintf('K-means 1D shaped (dev,z)        J = %.6f\n', J_kz);
fprintf('K-means 1D raw (dev,x)           J = %.6f\n', J_kx);

%% ======================= Monte Carlo TEST evaluation (paired) =======================
M_test = 500;

J_id_t = zeros(M_test,1);
J_uq_t = zeros(M_test,1);
J_lq_t = zeros(M_test,1);
J_kz_t = zeros(M_test,1);
J_kx_t = zeros(M_test,1);

disp('Running Monte Carlo test set (paired dropout sequences)...');
for tr = 1:M_test
    x0_dev = sample_dev(p_local, sigma_local, xmax_wide, n);
    x0_abs = xstar(:,1) + x0_dev;

    rng(10+tr);
    drop_seq = rand(1,Nsim) < p_drop;   % paired across schemes for this trial
    base_seed = 1000 + tr;

    [x1,u1] = simulate_tracking_drop(A,B,K,F, x0_abs, r, @(xk,xs_k) xk,        xstar, ustar, sigma_w, drop_seq, base_seed);
    [x2,u2] = simulate_tracking_drop(A,B,K,F, x0_abs, r, q_uniform,            xstar, ustar, sigma_w, drop_seq, base_seed);
    [x3,u3] = simulate_tracking_drop(A,B,K,F, x0_abs, r, q_log,                xstar, ustar, sigma_w, drop_seq, base_seed);
    [x4,u4] = simulate_tracking_drop(A,B,K,F, x0_abs, r, q_kmeans1D_shaped,     xstar, ustar, sigma_w, drop_seq, base_seed);
    [x5,u5] = simulate_tracking_drop(A,B,K,F, x0_abs, r, q_kmeans1D_raw,        xstar, ustar, sigma_w, drop_seq, base_seed);

    J_id_t(tr) = cost_fun_tracking(x1,u1,xstar,ustar);
    J_uq_t(tr) = cost_fun_tracking(x2,u2,xstar,ustar);
    J_lq_t(tr) = cost_fun_tracking(x3,u3,xstar,ustar);
    J_kz_t(tr) = cost_fun_tracking(x4,u4,xstar,ustar);
    J_kx_t(tr) = cost_fun_tracking(x5,u5,xstar,ustar);
end

dU  = J_uq_t - J_id_t;
dL  = J_lq_t - J_id_t;
dKz = J_kz_t - J_id_t;
dKx = J_kx_t - J_id_t;

fprintf('\n==== Test-set excess cost (relative to ideal link) ====\n');
fprintf('Uniform:   mean=%.4g, median=%.4g, 95%%=%.4g\n', mean(dU),  median(dU),  prctile(dU,95));
fprintf('Log(mu):   mean=%.4g, median=%.4g, 95%%=%.4g\n', mean(dL),  median(dL),  prctile(dL,95));
fprintf('Kmeans(z): mean=%.4g, median=%.4g, 95%%=%.4g\n', mean(dKz), median(dKz), prctile(dKz,95));
fprintf('Kmeans(x): mean=%.4g, median=%.4g, 95%%=%.4g\n', mean(dKx), median(dKx), prctile(dKx,95));

fprintf('Win rate (Log < Uniform):      %.2f %%\n', 100*mean(J_lq_t < J_uq_t));
fprintf('Win rate (Kmeans(z) < Uniform):%.2f %%\n', 100*mean(J_kz_t < J_uq_t));
fprintf('Win rate (Kmeans(z) < Log):    %.2f %%\n', 100*mean(J_kz_t < J_lq_t));

% Simple paired sign test vs mu-law (two-sided)
dKzL = J_kz_t - J_lq_t;
wins = sum(dKzL < 0);
pval = 2*binocdf(min(wins, M_test-wins), M_test, 0.5);
fprintf('Paired sign-test vs mu-law (two-sided) p=%.3g (wins=%d/%d)\n', pval, wins, M_test);

%% ======================= Plots (single episode) =======================
y_id = C*x_id;  y_uq = C*x_uq;  y_lq = C*x_lq;  y_kz = C*x_kz;  y_kx = C*x_kx;

figure('Name','Reference tracking output (drops+hold)');
plot(t, r, 'k--', 'LineWidth', 1.6); hold on;
plot(t, y_id, 'LineWidth', 1.6);
plot(t, y_uq, 'LineWidth', 1.6);
plot(t, y_lq, 'LineWidth', 1.6);
plot(t, y_kz, 'LineWidth', 1.6);
plot(t, y_kx, 'LineWidth', 1.6);
grid on;
xlabel('Time (s)'); ylabel('y');
title(sprintf('Tracking with drops (p=%.2f): y vs r', p_drop));
legend('r','Ideal','Uniform','Log(\mu-law)','K-means shaped','K-means raw','Location','best');

figure('Name','State trajectories (single episode)');
subplot(2,1,1);
plot(t, x_id(1,:), 'LineWidth', 1.4); hold on;
plot(t, x_uq(1,:), 'LineWidth', 1.4);
plot(t, x_lq(1,:), 'LineWidth', 1.4);
plot(t, x_kz(1,:), 'LineWidth', 1.4);
plot(t, x_kx(1,:), 'LineWidth', 1.4);
grid on; ylabel('x_1');
legend('Ideal','Uniform','Log','Kmeans(z)','Kmeans(x)','Location','best');
title('State x_1');

subplot(2,1,2);
plot(t, x_id(2,:), 'LineWidth', 1.4); hold on;
plot(t, x_uq(2,:), 'LineWidth', 1.4);
plot(t, x_lq(2,:), 'LineWidth', 1.4);
plot(t, x_kz(2,:), 'LineWidth', 1.4);
plot(t, x_kx(2,:), 'LineWidth', 1.4);
grid on; ylabel('x_2'); xlabel('Time (s)');
title('State x_2');

figure('Name','Control input (single episode)');
stairs(t(1:end-1), u_id, 'LineWidth', 1.4); hold on;
stairs(t(1:end-1), u_uq, 'LineWidth', 1.4);
stairs(t(1:end-1), u_lq, 'LineWidth', 1.4);
stairs(t(1:end-1), u_kz, 'LineWidth', 1.4);
stairs(t(1:end-1), u_kx, 'LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('u');
title('Control inputs');
legend('Ideal','Uniform','Log','Kmeans(z)','Kmeans(x)','Location','best');

figure('Name','Excess cost histogram (test set)');
histogram(dU,  'Normalization','pdf'); hold on;
histogram(dL,  'Normalization','pdf');
histogram(dKz, 'Normalization','pdf');
histogram(dKx, 'Normalization','pdf');
grid on;
xlabel('J - J_{ideal}');
ylabel('PDF');
title(sprintf('Excess cost over test set (p_{drop}=%.2f, bits=%d)', p_drop, bits_total));
legend('Uniform','Log(\mu-law)','K-means shaped','K-means raw','Location','best');

%% ============================================================
%  Local functions
%% ============================================================

function [xstar, ustar] = equilibrium_from_r(A,B,C,rval)
    n = size(A,1);
    m = size(B,2);
    I = eye(n);
    M = [I - A, -B;
         C,     zeros(size(C,1), m)];
    b = [zeros(n,1); rval];
    sol = M\b;
    xstar = sol(1:n);
    ustar = sol(n+1:end);
end

function dev = sample_dev(p_local, sigma_local, xmax_wide, n)
    if rand < p_local
        dev = sigma_local * randn(n,1);
    else
        dev = (2*rand(n,1)-1) * xmax_wide;
    end
end

function xhat = quantize_dev_uniform(xk, xstar_k, xmin, xmax, D1, D2)
    dev = xk - xstar_k;
    q1 = D1 * round( sat(dev(1), xmin, xmax) / D1 );
    q2 = D2 * round( sat(dev(2), xmin, xmax) / D2 );
    xhat = xstar_k + [q1; q2];
end

function xhat = quantize_dev_mulaw(xk, xstar_k, mu1, mu2, Xmax1, Xmax2, Dy1, Dy2)
    dev = xk - xstar_k;

    y1  = mulaw_compress(dev(1), mu1, Xmax1);
    y1q = Dy1 * round( sat(y1, -1, 1) / Dy1 );
    d1h = mulaw_expand(y1q, mu1, Xmax1);

    y2  = mulaw_compress(dev(2), mu2, Xmax2);
    y2q = Dy2 * round( sat(y2, -1, 1) / Dy2 );
    d2h = mulaw_expand(y2q, mu2, Xmax2);

    xhat = xstar_k + [d1h; d2h];
end

function y = mulaw_compress(x, mu, Xmax)
    x = sat(x, -Xmax, Xmax);
    y = sign(x) .* log(1 + mu*abs(x)/Xmax) ./ log(1 + mu);
end

function x = mulaw_expand(y, mu, Xmax)
    y = sat(y, -1, 1);
    x = sign(y) .* (Xmax/mu) .* ((1 + mu).^abs(y) - 1);
end

function xhat = quantize_dev_kmeans1D_z(xk, xstar_k, L, Linvt, C1z, C2z)
    dev = xk - xstar_k;
    z = L*dev;
    zhat = [nearest_centroid_1d(z(1), C1z);
            nearest_centroid_1d(z(2), C2z)];
    xhat = xstar_k + Linvt*zhat;
end

function xhat = quantize_dev_kmeans1D_x(xk, xstar_k, C1x, C2x)
    dev = xk - xstar_k;
    devhat = [nearest_centroid_1d(dev(1), C1x);
              nearest_centroid_1d(dev(2), C2x)];
    xhat = xstar_k + devhat;
end

function zhat = nearest_centroid_1d(z, C)
    [~, i] = min(abs(C - z));
    zhat = C(i);
end

function y = sat(x, xmin, xmax)
    y = min(max(x, xmin), xmax);
end

function [x, u] = simulate_tracking_drop(A,B,K,F, x0, r, quantizer, xstar, ustar, sigma_w, drop_seq, seed_noise)
    % Controller uses hold-last xhat when drop_seq(k)=true.
    n = size(A,1);
    N = size(r,2) - 1;

    x = zeros(n, N+1);
    u = zeros(1, N);

    x(:,1) = x0;

    % initialize held estimate (assume first packet is received)
    xhat_prev = quantizer(x(:,1), xstar(:,1));

    rng(seed_noise); % noise seed
    for k = 1:N
        if drop_seq(k)
            xhat = xhat_prev; % hold last received
        else
            xhat = quantizer(x(:,k), xstar(:,k));
        end
        xhat_prev = xhat;

        u(k) = -K*xhat + F*r(:,k);
        x(:,k+1) = A*x(:,k) + B*u(k) + sigma_w*randn(n,1);
    end
end
