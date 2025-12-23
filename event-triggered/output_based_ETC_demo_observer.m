clear;
close all;
clc

%% System Definition
A = [2 3; 1 0];
B = [0; 1];
C = [1 0];

x0 = [2; 0];
Tsim = 3;
dt = 1e-3;
t = 0:dt:Tsim;

ref = 3 * ones(1, length(t));

%% Controller Design
K = place(A, B, [-5 -6]);
fprintf('Controller gain K: [%.4f, %.4f]\n', K(1), K(2));

%% Observer Design
% Discretize system
sys_c = ss(A, B, C, 0);
sys_d = c2d(sys_c, dt, 'zoh');
A_d = sys_d.A;
B_d = sys_d.B;

% Check observability
if rank(obsv(A_d, C)) < size(A_d, 1)
    error('System is not observable!');
end

% Design observer poles
% Controller poles: [-5, -6]
obs_poles_c = [-12, -15];
obs_poles_d = exp(obs_poles_c * dt);  % Convert to discrete-time??

L = place(A_d', C', obs_poles_d)';
% L = place(A', C', obs_poles_c)';
fprintf('Observer gain L: [%.4f; %.4f]\n', L(1), L(2));

% Verify observer stability
fprintf('Observer poles: '); disp(eig(A_d - L*C)');

%% Prefilter Design

fprintf('\n=== Prefilter Design ===\n');

% Acl_discrete = A_d - B_d*K;
% Nr_discrete = -1 / (C * inv(Acl_discrete) * B_d);
% 
% fprintf('Discrete-time prefilter:\n');
% fprintf('  Nr = %.6f\n', Nr_discrete);

% Formula (for continuous-time)
% Nr = -1 / (C * inv(A - B*K) * B)

Acl_continuous = A - B*K;
Nr_continuous = -1 / (C * inv(Acl_continuous) * B);

fprintf('Continuous-time prefilter:\n');
fprintf('  Nr = %.6f\n', Nr_continuous);

Nr = Nr_continuous;

% % Verify: Check steady-state gain
% dc_gain = C * inv(-Acl_discrete) * B_d * Nr;
% fprintf('\nVerification:\n');
% fprintf('  DC gain (should be 1.0): %.6f\n', dc_gain);
% fprintf('  This means: y_ss = %.6f × r\n\n', dc_gain);

% Verify: Check steady-state gain
dc_gain = C * inv(-Acl_continuous) * B * Nr;
fprintf('\nVerification:\n');
fprintf('  DC gain (should be 1.0): %.6f\n', dc_gain);
fprintf('  This means: y_ss = %.6f × r\n\n', dc_gain);


%% Event-Triggering Parameter
sigma_s = 0.2;  % Relative threshold
fprintf('Event threshold sigma_s: %.2f\n', sigma_s);

%% Lyapunov Matrix
Acl = A - B*K;
Q_lyap = eye(2);
P = lyap(Acl', Q_lyap);

%%  SIMULATION 1: PERIODIC OUTPUT-FEEDBACK

fprintf('\n=== Simulating Periodic Output Feedback ===\n');

Te_per = 0.05;

x_true_per = x0;
x_s_per = x0;
x_c_per = x0;
u_per = -K * x_c_per;
next_transmission = 0;

x_true_per_traj = zeros(2, length(t));
u_per_traj = zeros(1, length(t));
event_times_per = [];
event_indices_per = [];

for k = 1:length(t)
    tk = t(k);
    
    y = C * x_true_per;
    
    % Observer update
    x_s_per_next = A_d*x_s_per + B_d*u_per + L*(y - C*x_s_per);
    
    % Predictor update
    x_c_per_next = A_d*x_c_per + B_d*u_per;
    
    % PERIODIC TRIGGERING
    if tk >= next_transmission - dt/2
        x_c_per_next = x_s_per_next;
        next_transmission = next_transmission + Te_per;
        event_times_per(end+1) = tk;
        event_indices_per(end+1) = k;
    end
    
    u_per_next = -K * x_c_per_next;
    
    % Plant evolution
    x_true_per_dot = A*x_true_per + B*u_per;
    x_true_per_next = x_true_per + dt * x_true_per_dot;
    
    x_true_per_traj(:, k) = x_true_per;
    u_per_traj(k) = u_per;
    
    x_true_per = x_true_per_next;
    x_s_per = x_s_per_next;
    x_c_per = x_c_per_next;
    u_per = u_per_next;
end

num_events_per = length(event_times_per);
comm_rate_per = num_events_per / 61 * 100;
fprintf('Number of events: %d / %d (%.2f%%)\n', ...
        num_events_per, length(t), comm_rate_per);

%%  SIMULATION 2: OUTPUT-BASED EVENT-TRIGGERED CONTROL
fprintf('\n=== Simulating Output-Based ETC ===\n');

% Initialize states
x_true = x0;           % True plant state (unknown to controller)
x_s = x0;              % Observer estimate (sensor side)
x_c = x0;              % Predictor estimate (controller side)
u_etc = -K * x_c + Nr * ref(1);      % Initial control
% u_etc = -K * x_c;      % Initial control

% Storage
x_true_etc = zeros(2, length(t));
x_s_traj = zeros(2, length(t));
x_c_traj = zeros(2, length(t));
u_etc_traj = zeros(1, length(t));
y_traj = zeros(1, length(t));
event_times_etc = [];
event_indices_etc = [];

for k = 1:length(t)
    % Measurement
    y = C * x_true;
    
    % SENSOR SIDE: Observer update (uses measurement)
    x_s_next = A_d*x_s + B_d*u_etc + L*(y - C*x_s);
    
    % BOTH SIDES: Predictor update (no measurement)
    x_c_next = A_d*x_c + B_d*u_etc;
    
    % EVENT-TRIGGERING CONDITION
    error_norm = norm(x_s_next - x_c_next);
    threshold = sigma_s * norm(x_s_next);
    
    if error_norm > threshold
        % EVENT TRIGGERED!
        x_c_next = x_s_next;  % Reset predictor to observer
        event_times_etc(end+1) = t(k);
        event_indices_etc(end+1) = k;
    end
    
    % CONTROLLER: Compute control using predictor estimate
    u_etc_next = -K * x_c_next + Nr * ref(1);
    % u_etc_next = -K * x_c_next;
    
    % PLANT: True state evolution (continuous-time)
    x_true_dot = A*x_true + B*u_etc;
    x_true_next = x_true + dt * x_true_dot;
    
    % Store trajectories
    x_true_etc(:, k) = x_true;
    x_s_traj(:, k) = x_s;
    x_c_traj(:, k) = x_c;
    u_etc_traj(k) = u_etc;
    y_traj(k) = y;
    
    % Update for next iteration
    x_true = x_true_next;
    x_s = x_s_next;
    x_c = x_c_next;
    u_etc = u_etc_next;
end

num_events_etc = length(event_times_etc);
comm_rate_etc = num_events_etc / length(event_times_per) * 100;
fprintf('Number of events: %d / %d (%.2f%%)\n', ...
        num_events_etc, num_events_per, comm_rate_etc);

%%  SIMULATION 3: STATE-FEEDBACK ETC (Ce aveam pana acum)

fprintf('\n=== Simulating State-Feedback ETC (Ideal) ===\n');

sigma = 0.9;
tau_min = 1e-3;
tol = 1e-12;

BKterm = P * B * (-K);
Psi = [(sigma-1)*Q_lyap, BKterm; BKterm', zeros(2)];

x = x0;
x_last = x;
u_sf = -K * x_last;
last_trigger_time = -Inf;

x_sf_traj = zeros(2, length(t));
u_sf_traj = zeros(1, length(t));
event_times_sf = [];
event_indices_sf = [];

for k = 1:length(t)
    tk = t(k);
    
    xdot = A*x + B*u_sf;
    x = x + dt * xdot;
    
    e = x_last - x;
    z = [x; e];
    val = z' * Psi * z;
    
    if (val > tol) && (tk - last_trigger_time > tau_min)
        x_last = x;
        u_sf = -K * x_last;
        last_trigger_time = tk;
        event_times_sf(end+1) = tk;
        event_indices_sf(end+1) = k;
    end
    
    x_sf_traj(:, k) = x;
    u_sf_traj(k) = u_sf;
end

num_events_sf = length(event_times_sf);
comm_rate_sf = num_events_sf / length(event_times_per) * 100;
fprintf('Number of events: %d / %d (%.2f%%)\n', ...
        num_events_sf, num_events_per, comm_rate_sf);


%  PLOTTING RESULTS
%% Figure 1: State Trajectories
figure('Position', [100, 100, 1400, 900]);

% x1 comparison
subplot(3,3,1)
plot(t, x_true_etc(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_s_traj(1,:), 'r--', 'LineWidth', 1.2);
plot(t, x_c_traj(1,:), 'g:', 'LineWidth', 1.2);
if ~isempty(event_indices_etc)
    plot(t(event_indices_etc), x_true_etc(1, event_indices_etc), ...
         'mo', 'MarkerFaceColor', 'm', 'MarkerSize', 4);
end
ylabel('x_1');
legend('True x_1', 'Observer x̂_s', 'Predictor x̂_c', 'Events', ...
       'Location', 'best', 'FontSize', 8);
title('Output-Based ETC');
grid on;

subplot(3,3,4)
plot(t, x_true_etc(2,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_s_traj(2,:), 'r--', 'LineWidth', 1.2);
plot(t, x_c_traj(2,:), 'g:', 'LineWidth', 1.2);
ylabel('x_2');
legend('True x_2', 'Observer x̂_s', 'Predictor x̂_c', ...
       'Location', 'best', 'FontSize', 8);
grid on;

subplot(3,3,7)
stairs(t, u_etc_traj, 'b-', 'LineWidth', 1.3); hold on;
if ~isempty(event_indices_etc)
    plot(t(event_indices_etc), u_etc_traj(event_indices_etc), ...
         'mo', 'MarkerFaceColor', 'm', 'MarkerSize', 4);
end
ylabel('u');
xlabel('Time [s]');
legend('Control', 'Events', 'Location', 'best', 'FontSize', 8);
grid on;

% State-feedback ETC
subplot(3,3,2)
plot(t, x_sf_traj(1,:), 'b-', 'LineWidth', 1.5); hold on;
if ~isempty(event_indices_sf)
    plot(t(event_indices_sf), x_sf_traj(1, event_indices_sf), ...
         'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
end
ylabel('x_1');
legend('True x_1', 'Events', 'Location', 'best', 'FontSize', 8);
title('State-Feedback ETC');
grid on;

subplot(3,3,5)
plot(t, x_sf_traj(2,:), 'b-', 'LineWidth', 1.5); hold on;
ylabel('x_2');
legend('True x_2', 'Location', 'best', 'FontSize', 8);
grid on;

subplot(3,3,8)
stairs(t, u_sf_traj, 'b-', 'LineWidth', 1.3); hold on;
if ~isempty(event_indices_sf)
    plot(t(event_indices_sf), u_sf_traj(event_indices_sf), ...
         'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
end
ylabel('u');
xlabel('Time [s]');
legend('Control', 'Events', 'Location', 'best', 'FontSize', 8);
grid on;

% Periodic
subplot(3,3,3)
plot(t, x_true_per_traj(1,:), 'b-', 'LineWidth', 1.5); hold on;
if ~isempty(event_indices_per)
    plot(t(event_indices_per), x_true_per_traj(1, event_indices_per), ...
         'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 4);
end
ylabel('x_1');
legend('True x_1', 'Events', 'Location', 'best', 'FontSize', 8);
title('Periodic Output Feedback');
grid on;

subplot(3,3,6)
plot(t, x_true_per_traj(2,:), 'b-', 'LineWidth', 1.5);
ylabel('x_2');
legend('True x_2', 'Location', 'best', 'FontSize', 8);
grid on;

subplot(3,3,9)
stairs(t, u_per_traj, 'b-', 'LineWidth', 1.3); hold on;
if ~isempty(event_indices_per)
    plot(t(event_indices_per), u_per_traj(event_indices_per), ...
         'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 4);
end
ylabel('u');
xlabel('Time [s]');
legend('Control', 'Events', 'Location', 'best', 'FontSize', 8);
grid on;

%% Figure 2: Estimation Errors (Output-Based)
figure('Position', [150, 150, 1200, 600]);

% Observer error: ‖x - x̂_s‖
subplot(2,2,1)
obs_error = vecnorm(x_true_etc - x_s_traj);
plot(t, obs_error, 'r-', 'LineWidth', 1.5);
ylabel('‖x - x̂_s‖');
xlabel('Time [s]');
title('Observer Estimation Error');
grid on;

% Predictor error: ‖x̂_s - x̂_c‖
subplot(2,2,2)
pred_error = vecnorm(x_s_traj - x_c_traj);
threshold_line = sigma_s * vecnorm(x_s_traj);
plot(t, pred_error, 'g-', 'LineWidth', 1.5); hold on;
plot(t, threshold_line, 'k--', 'LineWidth', 1.2);
if ~isempty(event_indices_etc)
    plot(t(event_indices_etc), pred_error(event_indices_etc), ...
         'mo', 'MarkerFaceColor', 'm', 'MarkerSize', 6);
end
ylabel('‖x̂_s - x̂_c‖');
xlabel('Time [s]');
legend('Predictor Error', 'Threshold σ_s‖x̂_s‖', 'Events', ...
       'Location', 'best');
title('Event-Triggering Condition Monitor');
grid on;

% State estimates vs true state (x1)
subplot(2,2,3)
plot(t, x_true_etc(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_s_traj(1,:), 'r--', 'LineWidth', 1.2);
plot(t, x_c_traj(1,:), 'g:', 'LineWidth', 1.2);
ylabel('x_1');
xlabel('Time [s]');
legend('True', 'Observer x̂_s', 'Predictor x̂_c', 'Location', 'best');
title('State x_1: True vs Estimates');
grid on;

% State estimates vs true state (x2)
subplot(2,2,4)
plot(t, x_true_etc(2,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_s_traj(2,:), 'r--', 'LineWidth', 1.2);
plot(t, x_c_traj(2,:), 'g:', 'LineWidth', 1.2);
ylabel('x_2');
xlabel('Time [s]');
legend('True', 'Observer x̂_s', 'Predictor x̂_c', 'Location', 'best');
title('State x_2: True vs Estimates (Unmeasured)');
grid on;

%% Figure 3: Three-Way Comparison
figure('Position', [200, 200, 1400, 500]);

subplot(1,3,1)
plot(t, x_true_etc(1,:), 'b-', 'LineWidth', 1.8); hold on;
plot(t, x_sf_traj(1,:), 'r--', 'LineWidth', 1.5);
plot(t, x_true_per_traj(1,:), 'g:', 'LineWidth', 1.5);
ylabel('x_1');
xlabel('Time [s]');
legend('Output-based ETC', 'State-feedback ETC', 'Periodic', ...
       'Location', 'best');
title('State x_1 Comparison');
grid on;

subplot(1,3,2)
plot(t, x_true_etc(2,:), 'b-', 'LineWidth', 1.8); hold on;
plot(t, x_sf_traj(2,:), 'r--', 'LineWidth', 1.5);
plot(t, x_true_per_traj(2,:), 'g:', 'LineWidth', 1.5);
ylabel('x_2');
xlabel('Time [s]');
legend('Output-based ETC', 'State-feedback ETC', 'Periodic', ...
       'Location', 'best');
title('State x_2 Comparison');
grid on;

subplot(1,3,3)
stairs(t, u_etc_traj, 'b-', 'LineWidth', 1.8); hold on;
stairs(t, u_sf_traj, 'r--', 'LineWidth', 1.5);
stairs(t, u_per_traj, 'g:', 'LineWidth', 1.5);
ylabel('Control u');
xlabel('Time [s]');
legend('Output-based ETC', 'State-feedback ETC', 'Periodic', ...
       'Location', 'best');
title('Control Input Comparison');
grid on;

%% Figure 4: Communication Statistics
figure('Position', [250, 250, 1000, 400]);

subplot(1,2,1)
bar([num_events_etc, num_events_sf, num_events_per]);
set(gca, 'XTickLabel', {'Output-based', 'State-feedback', 'Periodic'});
ylabel('Number of Transmissions');
title('Communication Cost');
grid on;
text(1, num_events_etc+50, sprintf('%d (%.1f%%)', num_events_etc, comm_rate_etc), ...
     'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(2, num_events_sf+50, sprintf('%d (%.1f%%)', num_events_sf, comm_rate_sf), ...
     'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(3, num_events_per+50, sprintf('%d (%.1f%%)', num_events_per, comm_rate_per), ...
     'HorizontalAlignment', 'center', 'FontWeight', 'bold');

subplot(1,2,2)
if ~isempty(event_times_etc)
    iet_etc = diff([0, event_times_etc]);
    stairs(1:length(iet_etc), iet_etc, 'b-', 'LineWidth', 1.5); hold on;
end
if ~isempty(event_times_sf)
    iet_sf = diff([0, event_times_sf]);
    stairs(1:length(iet_sf), iet_sf, 'r--', 'LineWidth', 1.5);
end
if ~isempty(event_times_per)
    iet_per = diff([0, event_times_per]);
    stairs(1:length(iet_per), iet_per, 'g:', 'LineWidth', 1.5);
end
ylabel('Inter-Event Time [s]');
xlabel('Event Number');
legend('Output-based', 'State-feedback', 'Periodic', 'Location', 'best');
title('Inter-Event Times');
grid on;

%% Figure 5: Lyapunov Functions
figure('Position', [300, 300, 1000, 400]);

V_etc = zeros(1, length(t));
V_sf = zeros(1, length(t));
V_per = zeros(1, length(t));

for k = 1:length(t)
    V_etc(k) = x_true_etc(:,k)' * P * x_true_etc(:,k);
    V_sf(k) = x_sf_traj(:,k)' * P * x_sf_traj(:,k);
    V_per(k) = x_true_per_traj(:,k)' * P * x_true_per_traj(:,k);
end

subplot(1,2,1)
semilogy(t, V_etc, 'b-', 'LineWidth', 1.8); hold on;
semilogy(t, V_sf, 'r--', 'LineWidth', 1.5);
semilogy(t, V_per, 'g:', 'LineWidth', 1.5);
if ~isempty(event_indices_etc)
    semilogy(t(event_indices_etc), V_etc(event_indices_etc), ...
             'mo', 'MarkerFaceColor', 'm', 'MarkerSize', 4);
end
ylabel('V(x) = x^T P x');
xlabel('Time [s]');
legend('Output-based ETC', 'State-feedback ETC', 'Periodic', 'Events', ...
       'Location', 'best');
title('Lyapunov Function Evolution');
grid on;

subplot(1,2,2)
control_effort_etc = cumsum(u_etc_traj.^2) * dt;
control_effort_sf = cumsum(u_sf_traj.^2) * dt;
control_effort_per = cumsum(u_per_traj.^2) * dt;

plot(t, control_effort_etc, 'b-', 'LineWidth', 1.8); hold on;
plot(t, control_effort_sf, 'r--', 'LineWidth', 1.5);
plot(t, control_effort_per, 'g:', 'LineWidth', 1.5);
ylabel('Cumulative ∫u² dt');
xlabel('Time [s]');
legend('Output-based ETC', 'State-feedback ETC', 'Periodic', ...
       'Location', 'best');
title('Control Effort');
grid on;

%% Performance Summary
fprintf('\n========================================\n');
fprintf('PERFORMANCE SUMMARY\n');
fprintf('========================================\n');
fprintf('Communication Events:\n');
fprintf('  Output-based ETC:     %d (%.2f%%)\n', num_events_etc, comm_rate_etc);
fprintf('  State-feedback ETC:   %d (%.2f%%)\n', num_events_sf, comm_rate_sf);
fprintf('  Periodic:             %d (%.2f%%)\n', num_events_per, comm_rate_per);
% fprintf('\nCommunication Savings:\n');
% fprintf('  Output vs State-fb:   %.1f%% penalty\n', ...
%         (num_events_etc/num_events_sf - 1)*100);
% fprintf('  Output vs Periodic:   %.1f%% savings\n', ...
%         (1 - num_events_etc/num_events_per)*100);
fprintf('\nFinal Lyapunov Values:\n');
fprintf('  Output-based ETC:     %.6f\n', V_etc(end));
fprintf('  State-feedback ETC:   %.6f\n', V_sf(end));
fprintf('  Periodic:             %.6f\n', V_per(end));
fprintf('\nControl Effort:\n');
fprintf('  Output-based ETC:     %.4f\n', control_effort_etc(end));
fprintf('  State-feedback ETC:   %.4f\n', control_effort_sf(end));
fprintf('  Periodic:             %.4f\n', control_effort_per(end));
fprintf('========================================\n');

%% Tuning Analysis: Effect of sigma_s

sigma_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5];
events_vs_sigma = zeros(1, length(sigma_values));
performance_vs_sigma = zeros(1, length(sigma_values));

for idx = 1:length(sigma_values)
    sigma_test = sigma_values(idx);

    x_test = x0;
    x_s_test = x0;
    x_c_test = x0;
    u_test = -K * x_c_test;
    event_count = 0;
    V_sum = 0;

    for k = 1:length(t)
        y = C * x_test;
        x_s_next = A_d*x_s_test + B_d*u_test + L*(y - C*x_s_test);
        x_c_next = A_d*x_c_test + B_d*u_test;

        if norm(x_s_next - x_c_next) > sigma_test * norm(x_s_next)
            x_c_next= x_s_next;
            event_count = event_count + 1;
        end
        u_test = -K * x_c_next;
        x_test_dot = A*x_test + B*u_test;
        x_test = x_test + dt * x_test_dot;
    
        V_sum = V_sum + x_test' * P * x_test;
    
        x_s_test = x_s_next;
        x_c_test = x_c_next;
    end

events_vs_sigma(idx) = event_count;
performance_vs_sigma(idx) = V_sum / length(t);
end
figure('Position', [350, 350, 1000, 400]);
subplot(1,2,1)
plot(sigma_values, events_vs_sigma, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('σ_s');
ylabel('Number of Events');
title('Communication vs Threshold');
grid on;
subplot(1,2,2)
plot(sigma_values, performance_vs_sigma, 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('σ_s');
ylabel('Average V(x)');
title('Performance vs Threshold');
grid on;