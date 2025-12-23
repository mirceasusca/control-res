%% ============================================================
%  Lyapunov-shaped quantization for networked LQR (minimal POC)
%
%  Plant:      x_{k+1} = A x_k + B u_k
%  Controller: u_k = -K * xhat_k
%  Network:    controller receives quantized state xhat_k = Q(x_k)
%
%  Compare (same bit budget, scalar/product quantizers):
%   1) Ideal:              xhat = x
%   2) Uniform in x:       quantize each xi on a fixed grid
%   3) Log (mu-law) in x:  per-channel compand+uniform in [-1,1]
%   4) Uniform in z=Lx:    quantize each zi, reconstruct xhat=L^{-1} zhat
%   5) K-means in z=Lx:    1D k-means per zi, reconstruct xhat=L^{-1} zhat
%
%  Outputs:
%   - Trajectories, control, V(x)=x'Sx decay
%   - Monte Carlo excess cost + Lyapunov decrease proxy
%   - Quantization geometry: x-grid, shaped grid, mu-law grid (mapped edges)
%
%  Requires:
%   - Control System Toolbox (dlqr)
%   - Statistics Toolbox (kmeans) for k-means case
%% ============================================================

clear; clc; close all;
rng(1);

%% -------------------- System (anisotropy-friendly benchmark) --------------------
Ts = 0.1;

A = [1.02  0.15;
     0.05  0.98];
B = [0.2;
     1.0];
n = size(A,1);

%% -------------------- LQR design (tilted Q) --------------------
theta = deg2rad(35);
Rrot = [cos(theta) -sin(theta);
        sin(theta)  cos(theta)];
Qeig = diag([40, 1]);
Q = Rrot * Qeig * Rrot.';
R = 0.15;

[K,S,~] = dlqr(A,B,Q,R);
Acl = A - B*K;

fprintf('eig(A)      = [%g %g]\n', eig(A));
fprintf('eig(A-BK)   = [%g %g]\n\n', eig(Acl));

% Lyapunov shaping
L = chol(S,'upper');
Linvt = inv(L);

%% ---- Show "tilt" induced by rotated Q and by S (Riccati) ----
% assumes you already computed Q,R,K,S,L as in your snippet.

% grid for contour plots
x1 = linspace(-2,2,301);
x2 = linspace(-2,2,301);
[X1,X2] = meshgrid(x1,x2);

% quadratic forms
VQ = Q(1,1)*X1.^2 + 2*Q(1,2)*X1.*X2 + Q(2,2)*X2.^2;
VS = S(1,1)*X1.^2 + 2*S(1,2)*X1.*X2 + S(2,2)*X2.^2;

levels = [0.2 0.5 1 2 4 8];

figure('Name','Tilt / anisotropy: Q vs S vs shaping','Color','w');

subplot(2,2,1);
contour(X1,X2,VQ,levels,'LineWidth',1.2);
axis equal; grid on;
xlabel('x_1'); ylabel('x_2');
title('Level sets of x^T Q x (tilted ellipses)');

subplot(2,2,2);
contour(X1,X2,VS,levels,'LineWidth',1.2);
axis equal; grid on;
xlabel('x_1'); ylabel('x_2');
title('Level sets of x^T S x (value / Lyapunov geometry)');

% % show that z = L x makes VS become circles: VS = ||z||^2
% % pick some points on circles in z-space and map back to x-space
% subplot(2,2,3); hold on;
% radii = sqrt(levels);                 % since ||z||^2 = level
% phi = linspace(0,2*pi,400);
% for rr = radii
%     Z = rr*[cos(phi); sin(phi)];      % circle in z
%     X = (L \ Z);                      % x = L^{-1} z
%     plot(X(1,:),X(2,:),'LineWidth',1.2);
% end
% axis equal; grid on;
% xlabel('x_1'); ylabel('x_2');
% title('Same S-level sets via Cholesky: x = L^{-1} z');
subplot(2,2,3); hold on;
theta = linspace(0,2*pi,400);
radii = sqrt(levels);   % since V = ||z||^2

for r = radii
    plot(r*cos(theta), r*sin(theta), 'LineWidth', 1.4);
end

axis equal; grid on;
xlabel('z_1'); ylabel('z_2');
title('Lyapunov level sets in z-space: ||z||^2 = const');

% overlay closed-loop trajectory cloud to "see" the tilt in data
subplot(2,2,4); hold on; grid on; axis equal;
xlabel('x_1'); ylabel('x_2');
title('Closed-loop trajectories + S-level sets');

% S-level sets for reference
contour(X1,X2,VS,[1 2 4],'LineWidth',1.0);

% simulate many random ICs under u=-Kx (no noise)
Nsim = 120;
M = 60;                      % number of trajectories
for i = 1:M
    x = zeros(2,Nsim);
    x(:,1) = 1.5*(2*rand(2,1)-1);     % random IC in box
    for k = 1:Nsim-1
        u = -K*x(:,k);
        x(:,k+1) = A*x(:,k) + B*u;
    end
    plot(x(1,:),x(2,:),'-');
end
legend('S-level sets','trajectories','Location','best');


%% -------------------- Simulation settings --------------------
N   = 300;
t   = (0:N)*Ts;

bits_total = 12;
b1 = floor(bits_total/2);
b2 = bits_total - b1;
K1 = 2^b1; K2 = 2^b2;

sigma_w = 0.01;

% Uniform ranges
xmax = 2.5;
zmax = 5.0;

Dx1 = 2*xmax / K1;
Dx2 = 2*xmax / K2;
Dz1 = 2*zmax / K1;
Dz2 = 2*zmax / K2;

% mu-law parameters (per-channel)
mu1 = 255;  mu2 = 255;
Xmax1 = xmax; Xmax2 = xmax;          % same full-scale as uniform-x
Dy1 = 2 / K1;                         % y in [-1,1] uniformly quantized
Dy2 = 2 / K2;

%% -------------------- Training data for k-means in z (offline) --------------------
M_train = 2000;
Ztrain  = zeros(M_train*(N+1), n);
idx = 1;

for tr = 1:M_train
    x = sample_x0();
    for k = 1:N
        u = -K*x;
        x = A*x + B*u + sigma_w*randn(n,1);
        Ztrain(idx,:) = (L*x).';
        idx = idx + 1;
    end
    Ztrain(idx,:) = (L*x).';
    idx = idx + 1;
end

opts = statset('MaxIter', 250, 'Display', 'final');
[~, C1z] = kmeans(Ztrain(:,1), K1, 'Replicates', 4, 'Start','plus', 'Options', opts);
[~, C2z] = kmeans(Ztrain(:,2), K2, 'Replicates', 4, 'Start','plus', 'Options', opts);
C1z = sort(C1z); C2z = sort(C2z);

%% -------------------- Quantizers --------------------
q_id = @(x) x;

q_ux = @(x) quantize_uniform_x(x, xmax, Dx1, Dx2);

q_logx = @(x) quantize_mulaw_x(x, mu1, mu2, Xmax1, Xmax2, Dy1, Dy2);

q_uz = @(x) Linvt * quantize_uniform_z(L*x, zmax, Dz1, Dz2);

q_kz = @(x) Linvt * kmeans1d_on_z(L*x, C1z, C2z);

%% -------------------- Single-run demo --------------------
x0 = [2.2; -1.8];

[x_id,u_id,V_id]     = simulate(A,B,K,S, x0, N, sigma_w, q_id);
[x_ux,u_ux,V_ux]     = simulate(A,B,K,S, x0, N, sigma_w, q_ux);
[x_log,u_log,V_log]  = simulate(A,B,K,S, x0, N, sigma_w, q_logx);
[x_uz,u_uz,V_uz]     = simulate(A,B,K,S, x0, N, sigma_w, q_uz);
[x_kz,u_kz,V_kz]     = simulate(A,B,K,S, x0, N, sigma_w, q_kz);

J_id  = cost_reg(x_id,u_id,Q,R);
J_ux  = cost_reg(x_ux,u_ux,Q,R);
J_log = cost_reg(x_log,u_log,Q,R);
J_uz  = cost_reg(x_uz,u_uz,Q,R);
J_kz  = cost_reg(x_kz,u_kz,Q,R);

fprintf('==== Single-run cost ====\n');
fprintf('Ideal          J = %.4f\n', J_id);
fprintf('Uniform in x   J = %.4f\n', J_ux);
fprintf('Log (mu-law)x  J = %.4f\n', J_log);
fprintf('Uniform in z   J = %.4f\n', J_uz);
fprintf('K-means in z   J = %.4f\n\n', J_kz);

%% -------------------- Monte Carlo evaluation --------------------
M_test = 300;
J = zeros(M_test,5);
proxy = zeros(M_test,5);

for tr = 1:M_test
    x0 = sample_x0();

    [x1,u1,V1] = simulate(A,B,K,S, x0, N, sigma_w, q_id);
    [x2,u2,V2] = simulate(A,B,K,S, x0, N, sigma_w, q_ux);
    [x3,u3,V3] = simulate(A,B,K,S, x0, N, sigma_w, q_logx);
    [x4,u4,V4] = simulate(A,B,K,S, x0, N, sigma_w, q_uz);
    [x5,u5,V5] = simulate(A,B,K,S, x0, N, sigma_w, q_kz);

    J(tr,1) = cost_reg(x1,u1,Q,R);
    J(tr,2) = cost_reg(x2,u2,Q,R);
    J(tr,3) = cost_reg(x3,u3,Q,R);
    J(tr,4) = cost_reg(x4,u4,Q,R);
    J(tr,5) = cost_reg(x5,u5,Q,R);

    proxy(tr,1) = frac_decrease(V1);
    proxy(tr,2) = frac_decrease(V2);
    proxy(tr,3) = frac_decrease(V3);
    proxy(tr,4) = frac_decrease(V4);
    proxy(tr,5) = frac_decrease(V5);
end

dUx  = J(:,2) - J(:,1);
dLog = J(:,3) - J(:,1);
dUz  = J(:,4) - J(:,1);
dKz  = J(:,5) - J(:,1);

fprintf('==== Test-set excess cost vs Ideal ====\n');
fprintf('Uniform x : mean=%.3g, median=%.3g, 95%%=%.3g\n', mean(dUx),  median(dUx),  prctile(dUx,95));
fprintf('Log x     : mean=%.3g, median=%.3g, 95%%=%.3g\n', mean(dLog), median(dLog), prctile(dLog,95));
fprintf('Uniform z : mean=%.3g, median=%.3g, 95%%=%.3g\n', mean(dUz),  median(dUz),  prctile(dUz,95));
fprintf('Kmeans z  : mean=%.3g, median=%.3g, 95%%=%.3g\n\n', mean(dKz), median(dKz), prctile(dKz,95));

fprintf('Win-rate (Log x   < Uniform x): %.1f %%\n', 100*mean(J(:,3) < J(:,2)));
fprintf('Win-rate (Uniform z < Uniform x): %.1f %%\n', 100*mean(J(:,4) < J(:,2)));
fprintf('Win-rate (Kmeans z  < Uniform x): %.1f %%\n', 100*mean(J(:,5) < J(:,2)));
fprintf('Win-rate (Kmeans z  < Log x):     %.1f %%\n', 100*mean(J(:,5) < J(:,3)));
fprintf('Win-rate (Kmeans z  < Uniform z): %.1f %%\n\n', 100*mean(J(:,5) < J(:,4)));

fprintf('==== Lyapunov decrease proxy (higher is better) ====\n');
fprintf('Ideal     : %.3f\n', mean(proxy(:,1)));
fprintf('Uniform x : %.3f\n', mean(proxy(:,2)));
fprintf('Log x     : %.3f\n', mean(proxy(:,3)));
fprintf('Uniform z : %.3f\n', mean(proxy(:,4)));
fprintf('Kmeans z  : %.3f\n\n', mean(proxy(:,5)));

%% -------------------- Figures --------------------

% 1) Geometry: Lyapunov ellipses + grids
figure('Name','Quantization geometry in x-space');
hold on; grid on; axis equal;
xlim([-2.8 2.8]); ylim([-2.8 2.8]);
xlabel('x_1'); ylabel('x_2');
title('Lyapunov geometry and quantization grids (x-space)');

plot_ellipses(S, [0.5 1 2 4 7]);                          % black
plot_uniform_grid([-xmax xmax], [-xmax xmax], Dx1, Dx2, ':'); % black dotted
plot_shaped_grid_in_x(Linvt, [-zmax zmax], [-zmax zmax], Dz1, Dz2, '--'); % blue dashed
plot_mulaw_grid([-xmax xmax], [-xmax xmax], mu1, mu2, Xmax1, Xmax2, Dy1, Dy2, ':'); % red-ish style via marker

legend('Lyap contours','Uniform-x grid','Uniform-z grid mapped','Mu-law-x grid','Location','best');

% 2) Trajectories
figure('Name','State trajectories');
subplot(2,1,1);
plot(t, x_id(1,:), 'LineWidth', 1.3); hold on;
plot(t, x_ux(1,:), 'LineWidth', 1.3);
plot(t, x_log(1,:), 'LineWidth', 1.3);
plot(t, x_uz(1,:), 'LineWidth', 1.3);
plot(t, x_kz(1,:), 'LineWidth', 1.3);
grid on; ylabel('x_1'); title('x_1');
legend('Ideal','Uniform x','Log x','Uniform z','Kmeans z','Location','best');

subplot(2,1,2);
plot(t, x_id(2,:), 'LineWidth', 1.3); hold on;
plot(t, x_ux(2,:), 'LineWidth', 1.3);
plot(t, x_log(2,:), 'LineWidth', 1.3);
plot(t, x_uz(2,:), 'LineWidth', 1.3);
plot(t, x_kz(2,:), 'LineWidth', 1.3);
grid on; ylabel('x_2'); xlabel('Time (s)'); title('x_2');

% 3) V(x) decay
figure('Name','Lyapunov function V(x)=x''Sx');
plot(t, V_id, 'LineWidth', 1.3); hold on;
plot(t, V_ux, 'LineWidth', 1.3);
plot(t, V_log,'LineWidth', 1.3);
plot(t, V_uz, 'LineWidth', 1.3);
plot(t, V_kz, 'LineWidth', 1.3);
grid on; xlabel('Time (s)'); ylabel('V(x)');
title('Lyapunov decay under quantized state feedback');
legend('Ideal','Uniform x','Log x','Uniform z','Kmeans z','Location','best');

% 4) Excess cost histograms
figure('Name','Excess cost histograms (test set)');
histogram(dUx,  'Normalization','pdf'); hold on;
histogram(dLog, 'Normalization','pdf');
histogram(dUz,  'Normalization','pdf');
histogram(dKz,  'Normalization','pdf');
grid on;
xlabel('J - J_{ideal}'); ylabel('PDF');
title(sprintf('Excess cost over %d Monte Carlo runs (bits=%d)', M_test, bits_total));
legend('Uniform x','Log x','Uniform z','Kmeans z','Location','best');

% 5) Learned bins in z
figure('Name','Learned 1D k-means bins in z-space');
subplot(1,2,1); stem(C1z, ones(size(C1z)), '.'); grid on; xlabel('z_1 centroid'); title('Codebook z_1');
subplot(1,2,2); stem(C2z, ones(size(C2z)), '.'); grid on; xlabel('z_2 centroid'); title('Codebook z_2');

%% ======================= Local functions =======================

function x0 = sample_x0()
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
    N = size(u,2);
    J = 0;
    for k = 1:N
        J = J + x(:,k).'*Q*x(:,k) + u(:,k).'*R*u(:,k);
    end
end

function frac = frac_decrease(V)
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

function xq = quantize_mulaw_x(x, mu1, mu2, Xmax1, Xmax2, Dy1, Dy2)
    % per-channel mu-law companding then uniform quantization in y in [-1,1]
    d1 = sat(x(1), -Xmax1, Xmax1);
    d2 = sat(x(2), -Xmax2, Xmax2);

    y1  = mulaw_compress(d1, mu1, Xmax1);
    y1q = Dy1 * round( sat(y1, -1, 1) / Dy1 );
    x1q = mulaw_expand(y1q, mu1, Xmax1);

    y2  = mulaw_compress(d2, mu2, Xmax2);
    y2q = Dy2 * round( sat(y2, -1, 1) / Dy2 );
    x2q = mulaw_expand(y2q, mu2, Xmax2);

    xq = [x1q; x2q];
end

function y = mulaw_compress(x, mu, Xmax)
    x = sat(x, -Xmax, Xmax);
    y = sign(x) .* log(1 + mu*abs(x)/Xmax) ./ log(1 + mu);
end

function x = mulaw_expand(y, mu, Xmax)
    y = sat(y, -1, 1);
    x = sign(y) .* (Xmax/mu) .* ((1 + mu).^abs(y) - 1);
end

function zhat = kmeans1d_on_z(z, C1z, C2z)
    zhat = [nearest_1d(z(1), C1z); nearest_1d(z(2), C2z)];
end

function c = nearest_1d(val, C)
    [~,i] = min(abs(C - val));
    c = C(i);
end

function y = sat(x, lo, hi)
    y = min(max(x, lo), hi);
end

function plot_ellipses(S, alphas)
    th = linspace(0,2*pi,300);
    [V,D] = eig(S);
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
    z1_edges = z1lim(1):Dz1:z1lim(2);
    z2_edges = z2lim(1):Dz2:z2lim(2);

    s = linspace(z2lim(1), z2lim(2), 200);
    for c = z1_edges
        Z = [c*ones(1,numel(s)); s];
        X = Linv * Z;
        plot(X(1,:), X(2,:), ['b' ls], 'LineWidth', 0.9);
    end

    s = linspace(z1lim(1), z1lim(2), 200);
    for c = z2_edges
        Z = [s; c*ones(1,numel(s))];
        X = Linv * Z;
        plot(X(1,:), X(2,:), ['b' ls], 'LineWidth', 0.9);
    end
end

function plot_mulaw_grid(xlimv, ylimv, mu1, mu2, Xmax1, Xmax2, Dy1, Dy2, ls)
    % Draw mu-law bin boundaries by mapping uniform y-edges back to x-edges.
    % This is the "Voronoi-equivalent" grid for mu-law (still axis-aligned, but nonuniform spacing).
    y1_edges = -1:Dy1:1;
    y2_edges = -1:Dy2:1;

    x1_edges = arrayfun(@(y) mulaw_expand(y, mu1, Xmax1), y1_edges);
    x2_edges = arrayfun(@(y) mulaw_expand(y, mu2, Xmax2), y2_edges);

    for x1 = x1_edges
        plot([x1 x1], ylimv, ['r' ls], 'LineWidth', 0.7);
    end
    for x2 = x2_edges
        plot(xlimv, [x2 x2], ['r' ls], 'LineWidth', 0.7);
    end
end
