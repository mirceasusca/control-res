%% Discrete-time 2nd-order system: LQR + uniform state quantization + cost J
clear; clc; close all;

Ts = 0.1;

% A, B (controllable)
A = [1.2  0.4;
    -0.3  0.9];
B = [1; 0.5];

% LQR design (discrete-time)
Q = diag([10, 2]);
R = 0.5;

% Discrete-time LQR gain
[K, S, e] = dlqr(A, B, Q, R);
Acl = A - B*K;

disp('K ='); disp(K);
disp('Closed-loop eigenvalues (no quantization) ='); disp(eig(Acl));

%% Quantization settings: separate steps for x1 and x2
Delta1 = 0.05;
Delta2 = 0.10;

% Uniform mid-tread quantizer: q = Delta * round(x/Delta)
qfun = @(x,Delta) Delta .* round(x./Delta);

%% Simulation
N  = 80;
x0 = [1.0; -0.5];

% Analytic infinite-horizon optimal cost (non-quantized)
J_star = x0.' * S * x0;

%% -------------------- Non-quantized case: u = -K x --------------------
x_nq = zeros(2, N+1);
u_nq = zeros(1, N);
x_nq(:,1) = x0;

for k = 1:N
    u_nq(k)   = -K * x_nq(:,k);
    x_nq(:,k+1) = A*x_nq(:,k) + B*u_nq(k);
end

% Truncated approximation of infinite-horizon cost (add terminal term x_N' S x_N)
J_nq = 0;
for k = 1:N
    J_nq = J_nq + x_nq(:,k).'*Q*x_nq(:,k) + u_nq(k)*R*u_nq(k);
end
J_nq = J_nq + x_nq(:,N+1).'*S*x_nq(:,N+1);

%% -------------------- Quantized case: u = -K x_q --------------------
x  = zeros(2, N+1);   % true state
xq = zeros(2, N+1);   % quantized state (measurement)
u  = zeros(1, N);     % control from quantized measurement

x(:,1)  = x0;
xq(:,1) = [qfun(x(1,1),Delta1); qfun(x(2,1),Delta2)];

for k = 1:N
    % Quantize measurement
    xq(:,k) = [qfun(x(1,k),Delta1); qfun(x(2,k),Delta2)];

    % Control uses quantized state
    u(k) = -K * xq(:,k);

    % True plant update
    x(:,k+1) = A*x(:,k) + B*u(k);
end
xq(:,N+1) = [qfun(x(1,N+1),Delta1); qfun(x(2,N+1),Delta2)];

% Truncated cost computed on TRUE state x and applied input u (add terminal term)
J_q = 0;
for k = 1:N
    J_q = J_q + x(:,k).'*Q*x(:,k) + u(k)*R*u(k);
end
J_q = J_q + x(:,N+1).'*S*x(:,N+1);

%% Print cost comparison
fprintf('\n==== Cost comparison (same Q,R) ====\n');
fprintf('J_star (analytic optimum, infinite horizon)   = %.10f\n', J_star);
fprintf('J_nq   (simulated non-quantized, + terminal)  = %.10f\n', J_nq);
fprintf('|J_nq - J_star|                               = %.3e\n', abs(J_nq - J_star));
fprintf('J_q    (quantized measurement, + terminal)    = %.10f\n', J_q);

if J_q >= J_star
    fprintf('As expected: J_q >= J_star (quantization cannot beat the true optimum).\n');
else
    fprintf(['Unexpected: J_q < J_star.\n' ...
             'Check you used TRUE x in the cost, and increase N if needed.\n']);
end

%% Time axis
t = (0:N)*Ts;

%% Plot time responses: quantized case (true vs quantized)
figure('Name','State vs time (true vs quantized) - quantized controller');
subplot(2,1,1);
plot(t, x(1,:), 'LineWidth', 1.8); hold on;
stairs(t, xq(1,:), 'LineWidth', 1.4);
grid on; ylabel('x_1');
legend('true x_1','quantized x_1','Location','best');
title(sprintf('Quantized feedback: \\Delta_1=%.3g, \\Delta_2=%.3g',Delta1,Delta2));

subplot(2,1,2);
plot(t, x(2,:), 'LineWidth', 1.8); hold on;
stairs(t, xq(2,:), 'LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('x_2');
legend('true x_2','quantized x_2','Location','best');

%% Control input (quantized case)
figure('Name','Control input (quantized case)');
stairs((0:N-1)*Ts, u, 'LineWidth', 1.6);
grid on;
xlabel('Time (s)');
ylabel('u[k]');
title('Control input computed from quantized state');

%% Phase portrait (x1-x2): quantized case (true and quantized trajectories)
figure('Name','Phase portrait (true vs quantized) - quantized controller');
plot(x(1,:),  x(2,:),  '-o', 'LineWidth', 1.6, 'MarkerSize', 3); hold on;
plot(xq(1,:), xq(2,:), '-s', 'LineWidth', 1.2, 'MarkerSize', 3);
plot(x(1,1), x(2,1), 'ks', 'MarkerFaceColor','k', 'MarkerSize', 8);
plot(x(1,end), x(2,end), 'kp', 'MarkerFaceColor','y', 'MarkerSize', 10);
grid on; axis equal;
xlabel('x_1'); ylabel('x_2');
title('Phase portrait with uniform quantization (separate steps)');
legend('True trajectory','Quantized trajectory','Start','End','Location','best');

%% (Optional) Phase portrait: non-quantized trajectory for comparison
figure('Name','Phase portrait - non-quantized vs quantized (true state)');
plot(x_nq(1,:), x_nq(2,:), '-o', 'LineWidth', 1.6, 'MarkerSize', 3); hold on;
plot(x(1,:),    x(2,:),    '-s', 'LineWidth', 1.2, 'MarkerSize', 3);
grid on; axis equal;
xlabel('x_1'); ylabel('x_2');
title('True-state phase portrait: non-quantized LQR vs quantized measurement');
legend('Non-quantized (u=-Kx)','Quantized (u=-Kx_q)','Location','best');
