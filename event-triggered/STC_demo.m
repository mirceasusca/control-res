clear
clc
close all

% Sistemul
A = [2 3;1 0];
B = [0; 1];
x0 = [1; 0];
Tsim = 8;

% Regulator
Q = eye(2); R = 1;
% K = lqr(A,B,Q,R);

K = place(A,B,[-2 -3]);

Acl = A - B*K;
P = lyap(Acl', Q);
lambda = 0.5;   % rata dorita de amortizare
V = @(x) x'*P*x;

% Simulare STC
x = x0;
t = 0;
dt = 1e-3;
x_traj = x;
t_traj = t;
u_traj = [];
event_times = [];

while t < Tsim
    xk = x;
    u = -K*xk;
    tau = Gamma_d(xk, A, B, K, P, lambda);

    % % Integrare pe [t, t+tau] cu u constant
    % Nt = ceil(tau/dt);
    % for i = 1:Nt
    %     xdot = A*x + B*u;
    %     x = x + dt * xdot;
    %     t = t + dt;
    %     x_traj(:,end+1) = x;
    %     t_traj(end+1) = t;
    %     u_traj(end+1) = u;
    % end
    % event_times(end+1) = t;
    xdot = @(t,x)(A*x+B*u);
    [t_ode, x_ode] = ode23(xdot, [t, t+tau], x);
    x = x_ode(end,:)';
    t_traj = [t_traj t_ode'];
    x_traj = [x_traj x_ode'];
    u_traj = [u_traj repmat(u,1,length(t_ode))];  % u constant
    t = t + tau;

    event_times(end+1) = t;
end

% Plotare

figure
subplot(311)
plot(t_traj, x_traj(1,:), 'b', 'LineWidth', 1.3);
ylabel('x_1'); grid on;

subplot(312)
plot(t_traj, x_traj(2,:), 'b', 'LineWidth', 1.3);
ylabel('x_2'); grid on;

subplot(313)
plot(t_traj(1:end-1), u_traj, 'b', 'LineWidth', 1.3);
ylabel('u'); grid on;

%% Functie Lyapunov

V_st = sum((P * x_traj).*x_traj, 1);
V0 = x0' * P * x0;
V_desired = V0 * exp(-lambda * t_traj);


figure;
plot(t_traj, V_st, 'b', 'LineWidth', 1.6); hold on;
plot(t_traj, V_desired, 'r--', 'LineWidth', 1.5);
xlabel('Timp [s]');
ylabel('V(x)');
legend('V(x) real','Decadere dorita e^{-\lambda t}','Location','best');
title('Evolutia functiei Lyapunov in Self-Triggered Control');
grid on;

% if ~isempty(event_times)
%     plot(event_times, interp1(t_traj, V_st, event_times), ...
%          'ko', 'MarkerFaceColor','k', 'MarkerSize',5);
%     legend('V(x) real','Decadere dorita','Evenimente','Location','best');
% end




























