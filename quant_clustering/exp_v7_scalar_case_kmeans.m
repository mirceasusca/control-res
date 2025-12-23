%% ============================================================
%  Canonical benchmark: encoder DOES NOT know reference r_k
%  ------------------------------------------------------------
%  Plant:      x_{k+1} = a x_k + u_k + w_k   (scalar LTI)
%  Controller: u_k = -K * xhat_k + F * r_k
%
%  Network imperfection: the SENSOR/ENCODER does NOT know r_k,
%  so it cannot form dev = x - x*(r). It must quantize ABSOLUTE x.
%
%  Compare (same B bits/sample):
%    1) Ideal link:    xhat = x
%    2) Uniform:       xhat = qu(x)
%    3) mu-law:        xhat = qmulaw(x)
%    4) K-means (1D):  xhat = qkmeans(x) trained on closed-loop x samples
%
%  This benchmark creates a bimodal x distribution (two equilibria ±d).
%  mu-law is single-center (dense near 0), so it can be dominated by K-means
%  as d grows (for fixed bits).
%
%  Requires Statistics and Machine Learning Toolbox (kmeans).
%% ============================================================

clear; clc; close all;
rng(1);

%% -------------------- System + controller design --------------------
Ts = 0.1; %#ok<NASGU>

% a = 0.90;      % stable open-loop to keep things simple/clean
a = 1.10;      % stable open-loop to keep things simple/clean
b = 1.00;      % scalar input gain
c = 1;
d = 0;

% LQR (scalar) for u = -K x  (design still "clean")
Q = 1;
R = 1;
[K,S,~] = dlqr(a,b,Q,R); %#ok<ASGLU>
Acl = a - b*K;

fprintf('K = %.6f\n', K);
fprintf('Acl = %.6f\n\n', Acl);

% Tracking prefilter choice so that x* = r for constant r:
% steady-state: x = a x + (-K x + F r)  => (1-a+K)x = F r
% choose F = (1-a+K) => x* = r
F = (1 - a + K);
fprintf('F = %.6f (so that x* = r)\n\n', F);

syscl = ss(a-b*K,b*F,c,d,Ts);

%% -------------------- Reference pattern: two equilibria (bimodal) --------------------
d = 4.0;                   % separation (increase to make mu-law fail harder)
dwell = 120;               % steps per mode (long dwell => strong clustering)
nSwitch = 6;               % total number of blocks
Nsim = nSwitch * dwell;    % total simulation length
t = (0:Nsim) * Ts;

r = zeros(1, Nsim+1);
levels = d * ((-1).^(0:nSwitch-1));  % +d, -d, +d, -d, ...
levels(3) = 0;

for j = 1:nSwitch
    k0 = (j-1)*dwell + 1;
    k1 = j*dwell + 1;
    r(k0:k1) = levels(j);
end

%% -------------------- Bit budget and quantizer ranges --------------------
Bbits = 6;              % bits per sample
Kc = 2^Bbits;           % number of codewords for K-means
fprintf('Bits/sample B = %d (K = %d)\n\n', Bbits, Kc);

% Choose full-scale for uniform/mu-law (avoid saturation but keep comparable)
sigma_w = 0.002;         % process noise (also makes "within-cluster" width)
Xmax = d + 0.5;         % full-scale; adjust if you want occasional sat

% Uniform (mid-tread via round)
Delta = 2*Xmax / (2^Bbits);

q_uniform = @(x) Delta * round( sat(x, -Xmax, Xmax) / Delta );

% mu-law: common "telephony-like" choice is mu = 2^B - 1
mu = 2^Bbits - 1;

% quantize in y-domain [-1,1] with step Dy (mid-tread via round)
Dy = 2 / (2^Bbits);

q_mulaw = @(x) mulaw_expand( ...
    Dy * round( sat(mulaw_compress(x, mu, Xmax), -1, 1) / Dy ), ...
    mu, Xmax);

%% ======================= TRAIN K-MEANS (offline) =======================
M_train = 200;          % training rollouts (keep moderate)
Ntrain  = Nsim;

Xtrain = zeros(M_train*(Ntrain+1), 1);
ptr = 1;

disp('Generating training data for K-means (encoder sees only x)...');
for tr = 1:M_train
    % random initial condition around 0 (unimodal init, bimodal due to r)
    x = zeros(1, Ntrain+1);
    x(1) = 0.5*randn;

    for k = 1:Ntrain
        % encoder is irrelevant here (we simulate ideal plant for training)
        u = -K * x(k) + F * r(k);
        x(k+1) = a*x(k) + b*u + sigma_w*randn;
    end

    Xtrain(ptr:ptr+Ntrain) = x(:);
    ptr = ptr + (Ntrain+1);
end

opts = statset('MaxIter', 1200, 'Display', 'off');
% Use kmeans++ init ("plus") and a few replicates to reduce bad local minima
[~, Ck] = kmeans(Xtrain, Kc, 'Replicates', 4, 'Start', 'plus', 'Options', opts);
Ck = sort(Ck);

q_kmeans = @(x) nearest_centroid_1d(x, Ck);

%% ======================= Single-episode demo =======================
x0 = 0.2;         % initial state
seed_demo = 7;    % fix noise for fair visual comparison

[x_id,  u_id]  = sim_closedloop(a,b,K,F,r,x0,sigma_w,seed_demo, @(x) x);
[x_u,   u_u]   = sim_closedloop(a,b,K,F,r,x0,sigma_w,seed_demo, q_uniform);
[x_mu,  u_mu]  = sim_closedloop(a,b,K,F,r,x0,sigma_w,seed_demo, q_mulaw);
[x_km,  u_km]  = sim_closedloop(a,b,K,F,r,x0,sigma_w,seed_demo, q_kmeans);

J = @(x,u) sum( (x(1:end-1)-r(1:end-1)).^2 * Q + (u.^2) * R );

J_id  = J(x_id, u_id);
J_u   = J(x_u,  u_u);
J_mu  = J(x_mu, u_mu);
J_km  = J(x_km, u_km);

fprintf('\n==== Single-episode cost (track r, encoder lacks r) ====\n');
fprintf('Ideal      J = %.6f\n', J_id);
fprintf('Uniform    J = %.6f\n', J_u);
fprintf('mu-law     J = %.6f\n', J_mu);
fprintf('K-means    J = %.6f\n', J_km);

%% ======================= Monte Carlo TEST =======================
M_test = 400;
J_id_t = zeros(M_test,1);
J_u_t  = zeros(M_test,1);
J_mu_t = zeros(M_test,1);
J_km_t = zeros(M_test,1);

disp('Running Monte Carlo test...');
for tr = 1:M_test
    x0 = 0.5*randn;
    seed = 1000 + tr;  % different noise each run

    [x1,u1] = sim_closedloop(a,b,K,F,r,x0,sigma_w,seed, @(x) x);
    [x2,u2] = sim_closedloop(a,b,K,F,r,x0,sigma_w,seed, q_uniform);
    [x3,u3] = sim_closedloop(a,b,K,F,r,x0,sigma_w,seed, q_mulaw);
    [x4,u4] = sim_closedloop(a,b,K,F,r,x0,sigma_w,seed, q_kmeans);

    J_id_t(tr) = J(x1,u1);
    J_u_t(tr)  = J(x2,u2);
    J_mu_t(tr) = J(x3,u3);
    J_km_t(tr) = J(x4,u4);
end

dU  = J_u_t  - J_id_t;
dMU = J_mu_t - J_id_t;
dKM = J_km_t - J_id_t;

fprintf('\n==== Excess cost over ideal (test set) ====\n');
fprintf('Uniform: mean=%.4g, median=%.4g, 95%%=%.4g\n', mean(dU),  median(dU),  prctile(dU,95));
fprintf('mu-law : mean=%.4g, median=%.4g, 95%%=%.4g\n', mean(dMU), median(dMU), prctile(dMU,95));
fprintf('Kmeans : mean=%.4g, median=%.4g, 95%%=%.4g\n', mean(dKM), median(dKM), prctile(dKM,95));
fprintf('Win rate (Kmeans < mu-law):  %.2f %%\n', 100*mean(J_km_t < J_mu_t));
fprintf('Win rate (Kmeans < Uniform): %.2f %%\n', 100*mean(J_km_t < J_u_t));

%% ======================= Plots =======================
figure('Name','Tracking (encoder lacks reference)');
plot(t, r, 'k--', 'LineWidth', 1.6); hold on;
plot(t, x_id, 'LineWidth', 1.3);
plot(t, x_u,  'LineWidth', 1.3);
plot(t, x_mu, 'LineWidth', 1.3);
plot(t, x_km, 'LineWidth', 1.3);
grid on;
xlabel('Time (s)'); ylabel('x');
title(sprintf('State tracking with B=%d bits, d=%.2f (encoder does NOT know r)', Bbits, d));
legend('r','Ideal','Uniform','\mu-law','K-means','Location','best');

figure('Name','State distribution (bimodal) + reconstruction levels');
histogram(Xtrain, 120, 'Normalization','pdf'); hold on; grid on;
xlabel('x'); ylabel('PDF');
title('Closed-loop state distribution used for K-means training');

% overlay reconstruction levels (subsample to avoid a black wall if B is big)
stepShow = max(1, round(numel(Ck)/60));
stem(Ck(1:stepShow:end), 0.02*ones(size(Ck(1:stepShow:end))), '.', 'LineWidth', 1.0);
legend('train pdf','K-means levels (subsampled)');

figure('Name','Excess-cost histograms');
histogram(dU,  40, 'Normalization','pdf'); hold on;
histogram(dMU, 40, 'Normalization','pdf');
histogram(dKM, 40, 'Normalization','pdf');
grid on;
xlabel('J - J_{ideal}'); ylabel('PDF');
title('Excess cost distribution (test set)');
legend('Uniform','\mu-law','K-means','Location','best');

%% ======================= Local functions =======================
function [x,u] = sim_closedloop(a,b,K,F,r,x0,sigma_w,seed,quantizer)
    rng(seed);
    N = numel(r)-1;
    x = zeros(1,N+1);
    u = zeros(1,N);
    x(1) = x0;
    for k = 1:N
        xhat = quantizer(x(k));          % encoder only sees x (no r)
        u(k) = -K*xhat + F*r(k);
        x(k+1) = a*x(k) + b*u(k) + sigma_w*randn;
    end
end

function y = sat(x, xmin, xmax)
    y = min(max(x, xmin), xmax);
end

function y = mulaw_compress(x, mu, Xmax)
    x = sat(x, -Xmax, Xmax);
    y = sign(x).*log(1 + mu*abs(x)/Xmax)./log(1+mu);
end

function x = mulaw_expand(y, mu, Xmax)
    y = sat(y, -1, 1);
    x = sign(y).*(Xmax/mu).*((1+mu).^abs(y) - 1);
end

function xq = nearest_centroid_1d(x, C)
    [~,i] = min(abs(C - x));
    xq = C(i);
end
