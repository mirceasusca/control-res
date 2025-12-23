%% ============================================================
%  Minimal demo: output-only networked control with TWO bottlenecks
%
%  Plant: x_{k+1} = A x_k + B u_k
%         y_k     = C x_k
%
%  Controller uses an observer (Luenberger/Kalman-style).
%  Two bottlenecks:
%    (1) ADC + sensor->controller link: y is quantized -> yq
%    (2) controller->actuator link + DAC: u is quantized AND may drop
%
%  Compare two controllers:
%   A) Naive observer: assumes applied input = u_cmd (ignores bottleneck #2)
%   B) "Aware" observer: predicts applied input = u_applied_model
%      (uses the same quantizer + dropout model as the actuator)
%
%  You should see B) track/regulate better when command-link distortion exists.
%
%  Requires: Control System Toolbox (dlqr). No Statistics Toolbox.
%% ============================================================

clear; clc; close all;
rng(1);

%% -------------------- Discrete-time plant (stable-ish) --------------------
Ts = 0.1;

A = [1.08  0.12;
     0.00  0.93];
B = [0.10;
     0.50];
C = [1 0];     % output-only measurement

n = size(A,1); m = size(B,2);

%% -------------------- State feedback (LQR) --------------------
Q = diag([20, 1]);
R = 0.3;
K = dlqr(A,B,Q,R);             % u = -K xhat
Acl = A - B*K;
fprintf('eig(A-BK) = [%s]\n', num2str(eig(Acl).', '%.3f '));

%% -------------------- Observer gain (simple Luenberger) --------------------
% Choose L by placing observer poles faster than controller poles.
p_obs = [0.35, 0.40];
L = place(A', C', p_obs).';    % xhat_{k+1} = A xhat + B u + L(y - C xhat)

%% -------------------- Network imperfections --------------------
% (1) Measurement quantizer (ADC)
Dy = 0.02;          % y quantization step
ymin = -2; ymax = 2;

% (2) Command quantizer + packet drops (DAC + network)
Du = 0.03;          % u quantization step
umin = -2; umax = 2;
p_drop = 0.15;      % drop probability; when dropped, actuator holds last u

%% -------------------- Simulation --------------------
N = 220;
t = (0:N)*Ts;

x0 = [1.2; -0.7];   % initial condition
sigma_w = 0.01;     % small process noise (optional, helps show robustness)

% Run both controllers on the SAME plant noise
seed_noise = 7;

res_naive = simulate_two_link(A,B,C,K,L, ...
    x0,N, sigma_w, seed_noise, ...
    Dy,ymin,ymax, Du,umin,umax, p_drop, ...
    "naive");

res_aware = simulate_two_link(A,B,C,K,L, ...
    x0,N, sigma_w, seed_noise, ...
    Dy,ymin,ymax, Du,umin,umax, p_drop, ...
    "aware");

%% -------------------- Costs (simple) --------------------
J_naive = sum(vecnorm(res_naive.x,2,1).^2) + 0.2*sum(res_naive.u.^2);
J_aware = sum(vecnorm(res_aware.x,2,1).^2) + 0.2*sum(res_aware.u.^2);
fprintf('\nCost proxy (smaller is better):\n');
fprintf('  naive = %.3f\n', J_naive);
fprintf('  aware = %.3f\n', J_aware);

%% -------------------- Plots --------------------
figure('Name','Output y and measurement');
plot(t, C*res_naive.x, 'LineWidth', 1.4); hold on;
plot(t, C*res_aware.x, 'LineWidth', 1.4);
stairs(t(1:end-1), res_naive.yq, 'k:', 'LineWidth', 1.0);
grid on; xlabel('Time (s)'); ylabel('y');
legend('y (naive)','y (aware)','y_q (sent)','Location','best');
title('Output regulation with quantized measurements');

figure('Name','States');
subplot(2,1,1);
plot(t, res_naive.x(1,:), 'LineWidth', 1.4); hold on;
plot(t, res_aware.x(1,:), 'LineWidth', 1.4);
grid on; ylabel('x_1'); legend('naive','aware','Location','best');

subplot(2,1,2);
plot(t, res_naive.x(2,:), 'LineWidth', 1.4); hold on;
plot(t, res_aware.x(2,:), 'LineWidth', 1.4);
grid on; ylabel('x_2'); xlabel('Time (s)');

figure('Name','Commands: commanded vs applied');
subplot(2,1,1);
stairs(t(1:end-1), res_naive.u_cmd, 'LineWidth', 1.1); hold on;
stairs(t(1:end-1), res_naive.u,     'LineWidth', 1.4);
grid on; ylabel('u'); title('Naive: command vs applied');
legend('u_{cmd}','u_{applied}','Location','best');

subplot(2,1,2);
stairs(t(1:end-1), res_aware.u_cmd, 'LineWidth', 1.1); hold on;
stairs(t(1:end-1), res_aware.u,     'LineWidth', 1.4);
grid on; ylabel('u'); xlabel('Time (s)'); title('Aware: command vs applied');
legend('u_{cmd}','u_{applied}','Location','best');

figure('Name','Estimation error norm');
en = vecnorm(res_naive.x - res_naive.xhat,2,1);
ea = vecnorm(res_aware.x - res_aware.xhat,2,1);
plot(t, en, 'LineWidth', 1.4); hold on;
plot(t, ea, 'LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('||x - xhat||_2');
legend('naive','aware','Location','best');
title('Observer mismatch reduced by modeling the command link');

%% ======================= Local functions =======================

function out = simulate_two_link(A,B,C,K,L, x0,N, sigma_w, seed_noise, ...
    Dy,ymin,ymax, Du,umin,umax, p_drop, mode)

n = size(A,1);

x    = zeros(n, N+1);
xhat = zeros(n, N+1);
u    = zeros(1, N);      % applied at actuator
u_cmd= zeros(1, N);      % computed at controller (pre-quant)
yq   = zeros(1, N);      % transmitted measurement

% Initial conditions
x(:,1) = x0;
xhat(:,1) = [0;0];       % intentionally wrong

u_hold = 0;              % actuator holds last applied command

rng(seed_noise);
w = sigma_w * randn(n, N);   % same noise sequence for fair compare

for k = 1:N
    % ---- Sensor measures y and quantizes it (bottleneck #1) ----
    y  = C*x(:,k);
    yq(k) = quant_uniform(y, Dy, ymin, ymax);

    % ---- Controller computes command from estimate ----
    u_cmd(k) = -K*xhat(:,k);      % "ideal" command in software

    % ---- Command link + DAC (bottleneck #2) happens physically at actuator ----
    % Quantize then drop with hold:
    uq = quant_uniform(u_cmd(k), Du, umin, umax);
    if rand < p_drop
        u(k) = u_hold;            % drop -> hold previous
    else
        u(k) = uq;
        u_hold = u(k);
    end

    % ---- Plant updates with applied input ----
    x(:,k+1) = A*x(:,k) + B*u(k) + w(:,k);

    % ---- Observer update at controller ----
    % The difference between modes is what input the controller uses
    % inside the predictor:
    switch lower(mode)
        case "naive"
            u_for_observer = u_cmd(k);  % WRONG: ignores quant/drop
        case "aware"
            % Controller predicts the same "applied input" using the
            % SAME quantizer+drop model (it can do this because it knows
            % the quantizer and the packet-loss realization via an ACK bit,
            % OR because the channel is deterministic/known).
            %
            % Minimal demo assumption: controller knows whether drop occurred.
            % (If not, replace this with an expectation model.)
            did_drop = (u(k) ~= uq); %#ok<NASGU>
            u_for_observer = u(k);      % BEST: use actual applied input
        otherwise
            error('Unknown mode.');
    end

    % One-step Luenberger observer:
    xpred = A*xhat(:,k) + B*u_for_observer;
    xhat(:,k+1) = xpred + L*(yq(k) - C*xpred);
end

out.x = x;
out.xhat = xhat;
out.u = u;
out.u_cmd = u_cmd;
out.yq = yq;

end

function q = quant_uniform(x, Delta, xmin, xmax)
x = min(max(x, xmin), xmax);
q = Delta * round(x/Delta);
end
