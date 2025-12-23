%% ============================================================
%  Lyapunov-shaped quantization for networked LQR (proof-of-concept)
%
%  Plant:      x_{k+1} = A x_k + B u_k
%  Controller: u_k = -K * xhat_k
%  Network:    controller receives quantized state xhat_k = Q(x_k)
%
%  Narrative:
%   - LQR yields Riccati S and Lyapunov/value geometry V(x)=x' S x.
%   - Cholesky factorization S = L' L gives shaped coords z = L x,
%     where V(x)=||z||^2.
%   - A quantizer that is "uniform in z" is aligned with Lyapunov geometry.
%
%  Compare (same bit budget, scalar/product quantizers):
%   1) Ideal:              xhat = x
%   2) Uniform in x:       quantize each xi on a fixed grid
%   3) Uniform in z=Lx:    quantize each zi on a fixed grid, reconstruct xhat=L^{-1} zhat
%   4) K-means in z=Lx:    1D k-means per zi (product quantizer), reconstruct xhat=L^{-1} zhat
%
%  Outputs:
%   - Trajectories, control, V(x) decay
%   - Monte Carlo excess cost + stability proxy
%   - Quantization geometry plots (x-grid vs shaped grid vs k-means bins)
%
%  Requires: Control System Toolbox (dlqr), Stats Toolbox (kmeans) for k-means case.
%% ============================================================

clear; clc; close all;
rng(1);

%% -------------------- System (choose anistropy-friendly benchmark) --------------------
Ts = 0.1;

% Mildly unstable/slow plant that makes "tilted" Lyapunov ellipses visible
A = [1.02  0.15;
     0.05  0.98];
B = [0.2;
     1.0];
n = size(A,1); m = size(B,2);

%% -------------------- LQR design (tilted Q) --------------------
theta = deg2rad(35);
Rrot = [cos(theta) -sin(theta);
        sin(theta)  cos(theta)];
Qeig = diag([20, 1]);         % penalize one direction heavily
Q = Rrot * Qeig * Rrot.';
R = 0.15;

[K,S,~] = dlqr(A,B,Q,R);
Acl = A - B*K;

fprintf('eig(A)      = [%g %g]\n', eig(A));
fprintf('eig(A-BK)   = [%g %g]\n\n', eig(Acl));

% Lyapunov shaping
L = chol(S,'upper');
Linvt = inv(L);

%% -------------------- Simulation settings --------------------
N   = 300;                   % horizon steps
t   = (0:N)*Ts;

bits_total = 16;              % total bits per sample (2D state)
b1 = floor(bits_total/2);    % per-channel bits
b2 = bits_total - b1;
K1 = 2^b1; K2 = 2^b2;

% disturbance (small) to avoid trivial collapse and to stress quantizer
sigma_w = 0.01;

% Uniform quantizer ranges (in x and z)
xmax = 2.5;                  % symmetric ranges; tune if needed
zmax = 5.0;

% Uniform steps
Dx1 = 2*xmax / K1;
Dx2 = 2*xmax / K2;
Dz1 = 2*zmax / K1;
Dz2 = 2*zmax / K2;

%% -------------------- Training data for k-means in z (offline) --------------------
% Roll out ideal closed-loop from randomized initial conditions, collect z = Lx.
% Keep it small & fast for "letter-style" reproducibility.
M_train = 2000;               % training trajectories
Ztrain  = zeros(M_train*(N+1), n);
idx = 1;

for tr = 1:M_train
    x = sample_x0();         % in x-space
    for k = 1:N
        u = -K*x;
        x = A*x + B*u + sigma_w*randn(n,1);
        Ztrain(idx,:) = (L*x).';
        idx = idx + 1;
    end
    Ztrain(idx,:) = (L*x).';
    idx = idx + 1;
end

% 1D k-means per coordinate (product codebook)
opts = statset('MaxIter', 300, 'Display', 'final');
[~, C1z] = kmeans(Ztrain(:,1), K1, 'Replicates', 7, 'Options', opts);
[~, C2z] = kmeans(Ztrain(:,2), K2, 'Replicates', 7, 'Options', opts);
C1z = sort(C1z); C2z = sort(C2z);

%% -------------------- Quantizers (function handles) --------------------
q_id   = @(x) x;

q_ux   = @(x) quantize_uniform_x(x, xmax, Dx1, Dx2);

q_uz   = @(x) Linvt * quantize_uniform_z(L*x, zmax, Dz1, Dz2);

% z_aux = L*x;
% q_kz   = @(x) Linvt * [nearest_1d(z_aux(1), C1z);   %#ok<NBRAK> (ok in MATLAB R2023+)
%                        nearest_1d(z_aux(2), C2z)];

% If your MATLAB version dislikes (L*x)(1) syntax, replace q_kz with:
q_kz = @(x) Linvt * kmeans1d_on_z(L*x, C1z, C2z);

%% -------------------- Single-run demo --------------------
x0 = [2.2; -1.8];     % choose a nontrivial initial condition (shows tilt)

[x_id,u_id,V_id] = simulate(A,B,K,S, x0, N, sigma_w, q_id);
[x_ux,u_ux,V_ux] = simulate(A,B,K,S, x0, N, sigma_w, q_ux);
[x_uz,u_uz,V_uz] = simulate(A,B,K,S, x0, N, sigma_w, q_uz);
[x_kz,u_kz,V_kz] = simulate(A,B,K,S, x0, N, sigma_w, q_kz);

J_id = cost_reg(x_id,u_id,Q,R);
J_ux = cost_reg(x_ux,u_ux,Q,R);
J_uz = cost_reg(x_uz,u_uz,Q,R);
J_kz = cost_reg(x_kz,u_kz,Q,R);

fprintf('==== Single-run cost (sum x''Qx + u''Ru) ====\n');
fprintf('Ideal          J = %.4f\n', J_id);
fprintf('Uniform in x   J = %.4f\n', J_ux);
fprintf('Uniform in z   J = %.4f\n', J_uz);
fprintf('K-means in z   J = %.4f\n\n', J_kz);

%% -------------------- Monte Carlo evaluation --------------------
M_test = 300;
J = zeros(M_test,4);
stableProxy = zeros(M_test,4); % fraction of steps with V_{k+1} <= V_k + c

for tr = 1:M_test
    x0 = sample_x0();
    [x1,u1,V1] = simulate(A,B,K,S, x0, N, sigma_w, q_id);
    [x2,u2,V2] = simulate(A,B,K,S, x0, N, sigma_w, q_ux);
    [x3,u3,V3] = simulate(A,B,K,S, x0, N, sigma_w, q_uz);
    [x4,u4,V4] = simulate(A,B,K,S, x0, N, sigma_w, q_kz);

    J(tr,1) = cost_reg(x1,u1,Q,R);
    J(tr,2) = cost_reg(x2,u2,Q,R);
    J(tr,3) = cost_reg(x3,u3,Q,R);
    J(tr,4) = cost_reg(x4,u4,Q,R);

    stableProxy(tr,1) = frac_decrease(V1);
    stableProxy(tr,2) = frac_decrease(V2);
    stableProxy(tr,3) = frac_decrease(V3);
    stableProxy(tr,4) = frac_decrease(V4);
end

dUx = J(:,2) - J(:,1);
dUz = J(:,3) - J(:,1);
dKz = J(:,4) - J(:,1);

fprintf('==== Test-set excess cost vs Ideal ====\n');
fprintf('Uniform x : mean=%.3g, median=%.3g, 95%%=%.3g\n', mean(dUx), median(dUx), prctile(dUx,95));
fprintf('Uniform z : mean=%.3g, median=%.3g, 95%%=%.3g\n', mean(dUz), median(dUz), prctile(dUz,95));
fprintf('Kmeans z  : mean=%.3g, median=%.3g, 95%%=%.3g\n', mean(dKz), median(dKz), prctile(dKz,95));
fprintf('\nWin-rate (Uniform z < Uniform x): %.1f %%\n', 100*mean(J(:,3) < J(:,2)));
fprintf('Win-rate (Kmeans z  < Uniform x): %.1f %%\n', 100*mean(J(:,4) < J(:,2)));
fprintf('Win-rate (Kmeans z  < Uniform z): %.1f %%\n\n', 100*mean(J(:,4) < J(:,3)));

fprintf('==== Lyapunov decrease proxy (higher is better) ====\n');
fprintf('Ideal     : %.3f\n', mean(stableProxy(:,1)));
fprintf('Uniform x : %.3f\n', mean(stableProxy(:,2)));
fprintf('Uniform z : %.3f\n', mean(stableProxy(:,3)));
fprintf('Kmeans z  : %.3f\n\n', mean(stableProxy(:,4)));

%% -------------------- Figures --------------------

% 1) Geometry: Lyapunov ellipses + grids
figure('Name','Quantization geometry in x-space (Lyapunov ellipses + grids)');
hold on; grid on; axis equal;
xlim([-2.8 2.8]); ylim([-2.8 2.8]);
xlabel('x_1'); ylabel('x_2');
title('Lyapunov geometry and quantization grids (x-space)');

% Lyapunov level sets: x' S x = alpha
plot_ellipses(S, [0.5 1 2 4 7]);

% Uniform-x grid lines
plot_uniform_grid([-xmax xmax], [-xmax xmax], Dx1, Dx2, ':');

% Shaped-uniform grid lines mapped back to x: z1=const, z2=const -> x = L^{-1} z
plot_shaped_grid_in_x(Linvt, [-zmax zmax], [-zmax zmax], Dz1, Dz2, '--');

legend('Lyap contours','Uniform grid in x','Uniform grid in z (mapped)','Location','best');

% 2) Trajectories
figure('Name','State trajectories');
subplot(2,1,1);
plot(t, x_id(1,:), 'LineWidth', 1.4); hold on;
plot(t, x_ux(1,:), 'LineWidth', 1.4);
plot(t, x_uz(1,:), 'LineWidth', 1.4);
plot(t, x_kz(1,:), 'LineWidth', 1.4);
grid on; ylabel('x_1'); title('x_1');
legend('Ideal','Uniform x','Uniform z','Kmeans z','Location','best');

subplot(2,1,2);
plot(t, x_id(2,:), 'LineWidth', 1.4); hold on;
plot(t, x_ux(2,:), 'LineWidth', 1.4);
plot(t, x_uz(2,:), 'LineWidth', 1.4);
plot(t, x_kz(2,:), 'LineWidth', 1.4);
grid on; ylabel('x_2'); xlabel('Time (s)'); title('x_2');

% 3) V(x) decay
figure('Name','Lyapunov function V(x)=x''Sx');
plot(t, V_id, 'LineWidth', 1.4); hold on;
plot(t, V_ux, 'LineWidth', 1.4);
plot(t, V_uz, 'LineWidth', 1.4);
plot(t, V_kz, 'LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('V(x)');
title('Lyapunov decay under quantized state feedback');
legend('Ideal','Uniform x','Uniform z','Kmeans z','Location','best');

% 4) Excess cost histograms
figure('Name','Excess cost histograms (test set)');
histogram(dUx, 'Normalization','pdf'); hold on;
histogram(dUz, 'Normalization','pdf');
histogram(dKz, 'Normalization','pdf');
grid on;
xlabel('J - J_{ideal}'); ylabel('PDF');
title(sprintf('Excess cost over %d Monte Carlo runs (bits=%d)', M_test, bits_total));
legend('Uniform x','Uniform z','Kmeans z','Location','best');

% 5) Optional: show learned 1D bins in z-space
figure('Name','Learned 1D k-means bins in z-space');
subplot(1,2,1);
stem(C1z, ones(size(C1z)), '.'); grid on;
xlabel('z_1 centroid'); title('Codebook for z_1');
subplot(1,2,2);
stem(C2z, ones(size(C2z)), '.'); grid on;
xlabel('z_2 centroid'); title('Codebook for z_2');

%% ======================= Local functions =======================

function x0 = sample_x0()
    % Mixture: mostly near origin, sometimes wide (shows quantizer range issues)
    if rand < 0.8
        x0 = 0.8*randn(2,1);
    else
        x0 = (2*rand(2,1)-1) * 2.5;
    end
end

function [x,u,V] = simulate(A,B,K,S, x0, N, sigma_w, Qfun)
    n = size(A,1);
    x = zeros(n, N+1);
    u = zeros(1, N);
    V = zeros(1, N+1);

    x(:,1) = x0;
    V(1) = x0.'*S*x0;

    for k = 1:N
        xhat = Qfun(x(:,k));
        u(k) = -K*xhat;
        x(:,k+1) = A*x(:,k) + B*u(k) + sigma_w*randn(n,1);
        V(k+1) = x(:,k+1).'*S*x(:,k+1);
    end
end

function J = cost_reg(x,u,Q,R)
    % finite-horizon regulation cost (no terminal term needed for this POC)
    N = size(u,2);
    J = 0;
    for k = 1:N
        J = J + x(:,k).'*Q*x(:,k) + u(:,k).'*R*u(:,k);
    end
end

function frac = frac_decrease(V)
    % simple proxy: how often V decreases (or nearly decreases) step-to-step
    % (with noise, require only mild increase tolerance)
    tol = 1e-3;
    frac = mean(V(2:end) <= V(1:end-1) + tol);
end

function xq = quantize_uniform_x(x, xmax, Dx1, Dx2)
    x1 = sat(x(1), -xmax, xmax);
    x2 = sat(x(2), -xmax, xmax);
    xq = [Dx1*round(x1/Dx1); Dx2*round(x2/Dx2)];
end

function zq = quantize_uniform_z(z, zmax, Dz1, Dz2)
    z1 = sat(z(1), -zmax, zmax);
    z2 = sat(z(2), -zmax, zmax);
    zq = [Dz1*round(z1/Dz1); Dz2*round(z2/Dz2)];
end

function y = sat(x, lo, hi)
    y = min(max(x, lo), hi);
end

function c = nearest_1d(val, C)
    [~,i] = min(abs(C - val));
    c = C(i);
end

function zhat = kmeans1d_on_z(z, C1z, C2z)
    zhat = [nearest_1d(z(1), C1z); nearest_1d(z(2), C2z)];
end

function plot_ellipses(S, alphas)
    th = linspace(0,2*pi,300);
    [V,D] = eig(S);
    % x' S x = alpha => x = sqrt(alpha)* V*D^{-1/2}*[cos;sin]
    T = V * sqrt(inv(D));
    for a = alphas
        e = sqrt(a) * T * [cos(th); sin(th)];
        plot(e(1,:), e(2,:), 'k-', 'LineWidth', 1.0);
    end
end

function plot_uniform_grid(xlimv, ylimv, Dx1, Dx2, ls)
    x1_edges = xlimv(1):Dx1:xlimv(2);
    x2_edges = ylimv(1):Dx2:ylimv(2);
    for x1 = x1_edges
        plot([x1 x1], ylimv, ['k' ls], 'LineWidth', 0.8);
    end
    for x2 = x2_edges
        plot(xlimv, [x2 x2], ['k' ls], 'LineWidth', 0.8);
    end
end

function plot_shaped_grid_in_x(Linv, z1lim, z2lim, Dz1, Dz2, ls)
    % Draw lines of constant z1 and constant z2 in x-space, using x = Linv*z.
    z1_edges = z1lim(1):Dz1:z1lim(2);
    z2_edges = z2lim(1):Dz2:z2lim(2);

    % For z1 = const: z = [c; s], sweep s
    s = linspace(z2lim(1), z2lim(2), 200);
    for c = z1_edges
        Z = [c*ones(1,numel(s)); s];
        X = Linv * Z;
        plot(X(1,:), X(2,:), ['b' ls], 'LineWidth', 0.9);
    end

    % For z2 = const: z = [s; c], sweep s
    s = linspace(z1lim(1), z1lim(2), 200);
    for c = z2_edges
        Z = [s; c*ones(1,numel(s))];
        X = Linv * Z;
        plot(X(1,:), X(2,:), ['b' ls], 'LineWidth', 0.9);
    end
end
