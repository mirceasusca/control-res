%% ============================================================
%  Minimal PAA-style "inverse quantizer" prototype (scalar LTI)
%
%  Plant:    x_{k+1} = a x_k + b u_k
%  Control:  u_k = -k * xhat_k       (remote state feedback)
%  ADC:      xq_k = Q(x_k)           (uniform quantizer, known)
%
%  Goal: learn a *dynamic inverse* of Q using only (xq,u) by enforcing
%  model-consistency: xhat_{k+1} ≈ a xhat_k + b u_k.
%
%  Decoder model (parametric):
%     xhat_k = xq_k + sat( phi_k' * theta,  [-Δ/2, +Δ/2] )
%     phi_k  = [1; xq_k; u_{k-1}]
%
%  Adaptation (simple PAA / LMS on one-step prediction error):
%     e_k = xhat_{k+1} - (a xhat_k + b u_k)
%     theta <- theta - gamma * e_k * phi_{k+1}
%
%  Notes:
%   - This is intentionally minimal + heuristic (proof of concept).
%   - Because sat() is non-smooth, this is not a strict gradient method.
%   - Still, it often reduces inter-sample quantization error by using dynamics.
%% ============================================================

clear; clc; close all;
rng(1);

%% -------------------- System + controller --------------------
a = 1.05;          % open-loop unstable (scalar)
b = 0.6;

% choose stabilizing state feedback (scalar pole placement)
p_cl = 0.6;        % desired closed-loop pole
k = (a - p_cl)/b;  % so that a - b*k = p_cl

fprintf('a=%.3f, b=%.3f, k=%.3f, closed-loop pole a-bk=%.3f\n', a,b,k,a-b*k);

%% -------------------- Quantizer (uniform, known) --------------------
Delta = 0.20;      % step size (bad ADC)
Q = @(x) Delta * round(x/Delta);

%% -------------------- Simulation setup --------------------
N  = 350;          % steps
x0 = 1.7;          % initial state
sigma_w = 0.00;    % set >0 if you want process noise

% adaptation
theta0 = [0; 0; 0];      % initial decoder params
gamma  = 0.03;           % step size (try 0.005..0.08)
theta_max = 5;           % safety bound on theta entries

%% -------------------- Run baseline (midpoint decode) --------------------
[x_mid, xq_mid, xhat_mid, u_mid] = run_closed_loop( ...
    a,b,k,Q,Delta,N,x0,sigma_w, ...
    false, theta0, gamma, theta_max);

%% -------------------- Run adaptive decoder (PAA) --------------------
[x_paa, xq_paa, xhat_paa, u_paa, theta_hist] = run_closed_loop( ...
    a,b,k,Q,Delta,N,x0,sigma_w, ...
    true, theta0, gamma, theta_max);

%% -------------------- Metrics --------------------
rmse_mid = rms(x_mid - xhat_mid);
rmse_paa = rms(x_paa - xhat_paa);

J_mid = sum(x_mid(1:end-1).^2 + 0.05*u_mid.^2);
J_paa = sum(x_paa(1:end-1).^2 + 0.05*u_paa.^2);

fprintf('\nRMSE state estimation (x - xhat): midpoint=%.4g, PAA=%.4g\n', rmse_mid, rmse_paa);
fprintf('Simple quadratic cost J = Σ(x^2 + 0.05 u^2): midpoint=%.4g, PAA=%.4g\n', J_mid, J_paa);

%% -------------------- Plots --------------------
t = 0:N;

figure('Name','State + reconstructions');
plot(t, x_mid, 'LineWidth', 1.4); hold on;
plot(t, xhat_mid, '--', 'LineWidth', 1.4);
plot(t, x_paa, 'LineWidth', 1.4);
plot(t, xhat_paa, '--', 'LineWidth', 1.4);
grid on; xlabel('k'); ylabel('x');
title('True state vs reconstructed state');
legend('x (midpoint)','xhat (midpoint)','x (PAA)','xhat (PAA)','Location','best');

figure('Name','Quantized measurement (ADC output)');
stairs(t, xq_mid, 'LineWidth', 1.2); hold on;
stairs(t, xq_paa, '--', 'LineWidth', 1.2);
grid on; xlabel('k'); ylabel('x_q');
title('ADC output x_q = Q(x)');
legend('midpoint run','PAA run','Location','best');

figure('Name','Control input');
stairs(0:N-1, u_mid, 'LineWidth', 1.2); hold on;
stairs(0:N-1, u_paa, '--', 'LineWidth', 1.2);
grid on; xlabel('k'); ylabel('u');
title('Control inputs');
legend('midpoint','PAA','Location','best');

figure('Name','Decoder parameters (theta)');
plot(0:N, theta_hist.', 'LineWidth', 1.2);
grid on; xlabel('k'); ylabel('\theta');
title('Adaptive inverse-quantizer parameters');
legend('\theta_1','\theta_2','\theta_3','Location','best');

%% ===================== Local functions =====================

function [x, xq, xhat, u, theta_hist] = run_closed_loop( ...
    a,b,k,Q,Delta,N,x0,sigma_w, do_adapt, theta0, gamma, theta_max)

    % storage
    x    = zeros(1, N+1);
    xq   = zeros(1, N+1);
    xhat = zeros(1, N+1);
    u    = zeros(1, N);
    theta_hist = zeros(numel(theta0), N+1);

    % init
    x(1)  = x0;
    xq(1) = Q(x(1));

    theta = theta0(:);
    theta_hist(:,1) = theta;

    % previous input for phi_k
    u_prev = 0;

    % decode at k=0
    xhat(1) = decode_with_theta(xq(1), u_prev, theta, Delta);

    for kstep = 1:N
        % control (uses reconstructed state)
        u(kstep) = -k * xhat(kstep);

        % plant update
        x(kstep+1) = a*x(kstep) + b*u(kstep) + sigma_w*randn();

        % ADC
        xq(kstep+1) = Q(x(kstep+1));

        % decode next state (needed for prediction error)
        xhat_next = decode_with_theta(xq(kstep+1), u(kstep), theta, Delta);

        if do_adapt
            % model-consistency error
            e = xhat_next - (a*xhat(kstep) + b*u(kstep));

            % regressor uses *available* signals at time k+1
            phi_next = [1; xq(kstep+1); u(kstep)];

            % PAA / LMS update (heuristic)
            theta = theta - gamma * e * phi_next;

            % safety bound to avoid blow-up
            theta = min(max(theta, -theta_max), theta_max);

            % recompute xhat_next with updated theta (optional, stabilizes)
            xhat_next = decode_with_theta(xq(kstep+1), u(kstep), theta, Delta);
        end

        xhat(kstep+1) = xhat_next;

        theta_hist(:,kstep+1) = theta;

        % shift
        u_prev = u(kstep);
    end
end

function xhat = decode_with_theta(xq, u_prev, theta, Delta)
    % parametric "inverse quantizer" correction within the cell
    phi = [1; xq; u_prev];
    delta_hat = phi.' * theta;

    % projection to the quantization cell (prevents nonsense)
    delta_hat = sat(delta_hat, -Delta/2, +Delta/2);

    xhat = xq + delta_hat;
end

function y = sat(x, xmin, xmax)
    y = min(max(x, xmin), xmax);
end
