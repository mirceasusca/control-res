clear;
close all;
clc

% Definesc un sistem de ordinul II
% A = [0 1;0 0];
A = [2 3;1 0];
% A = [-2 3;-1 0];
B = [0;1];

x0 = [1;0]; % stare initiala
Tsim = 3; % timp de simulare
dt = 1e-3;
t = 0:dt:Tsim;

% Design regulator stare (LQR => u = -K x)
% Q_lqr = eye(2); R_lqr = 1;
% K = lqr(A,B,Q_lqr,R_lqr);    % K este 1x2 (u = -K*x)

% K = place(A,B, [-5 -6]);
K = place(A,B, [-10 -10.1]);

Acl = A - B*K; % bucla inchisa

Q_lyap = eye(2);
P = lyap(Acl', Q_lyap);

% Parametrii regulatorului ET
sigma = 0.9; % cat de departe ma abat de ideal
tau_min = 1e-3; % evit Zeno
tol = 1e-12; % toleranta numerica

% Definesc matricea Psi din triggering condition
BKterm = P * B * (-K);
Psi = [(sigma-1)*Q_lyap, BKterm; ...
       BKterm', zeros(2)];

% Simulare ETC
x = x0;
x_last = x; % starea la ultimul trigger
u_ev = -K * x_last; % comanda initiala
last_trigger_time = -Inf;

x_ev_traj = zeros(2, numel(t));
u_ev_traj = zeros(1, numel(t));
event_times = [];

for k = 1:numel(t)
    tk = t(k);

    xdot = A*x + B*u_ev;
    x = x + dt * xdot; % integrare Euler. Poate reusesc sa schimb cu ode

    e = x_last - x; % eroarea
    z = [x;e];
    val = z' * Psi * z; % conditia de trigger

    % Declansez cand se trece de prag
    if (val > tol) && (tk - last_trigger_time > tau_min)
        x_last = x;
        u_ev = -K * x_last;
        last_trigger_time = tk;
        event_times(end+1) = tk;
    end
    
    x_ev_traj(:,k) = x;
    u_ev_traj(k) = u_ev;
end


% Implementarea regulatorului normal pentru comparatie
Te = 0.05;
x = x0;
next_sample = 0;
u_per = -K * x;
x_per_traj = zeros(2, numel(t));
u_per_traj = zeros(1, numel(t));
sample_count = 0;

for k = 1:numel(t)
    tk = t(k);
    if tk + dt/2 >= next_sample
        u_per = -K * x;
        next_sample = next_sample + Te;
        sample_count = sample_count + 1;
    end
    xdot = A*x + B*u_per;
    x = x + dt * xdot; % integrarea basic

    x_per_traj(:,k) = x;
    u_per_traj(k) = u_per;
end


figure

subplot(311)
plot(t, x_ev_traj(1,:), 'b-', 'LineWidth', 1.3); hold on;
plot(t, x_per_traj(1,:), 'r--', 'LineWidth', 1.0);
ylabel('x_1');
legend('Event-triggered','Periodic','Location','best'); grid on;
grid on

subplot(312)
plot(t, x_ev_traj(2,:), 'b-', 'LineWidth', 1.3); hold on;
plot(t, x_per_traj(2,:), 'r--', 'LineWidth', 1.0);
ylabel('x_2');
grid on

subplot(313)
plot(t, u_ev_traj(1,:), 'b-', 'LineWidth', 1.3); hold on;
plot(t, u_per_traj, 'r--', 'LineWidth', 1.0);
if ~isempty(event_times)
    index_event = round(event_times / dt + 1);
    plot(t(index_event), u_ev_traj(index_event), 'ko', 'MarkerFaceColor','k', 'MarkerSize',4);
end
ylabel('u (comanda)');
grid on

%% Plotare evolutie functie Lyapunov ETC

V_ev = zeros(1, length(t));
for k = 1:length(t)
    V_ev(k) = x_ev_traj(:,k)' * P * x_ev_traj(:,k);
end

figure;
plot(t, V_ev, 'b', 'LineWidth', 1.5);
xlabel('Timp [s]');
ylabel('V(x) = x^T P x');
title('Evolutia functiei Lyapunov in timp');
grid on;

if ~isempty(event_times)
    hold on;
    plot(event_times, interp1(t, V_ev, event_times), 'ko', 'MarkerFaceColor', 'k');
end


%% Plotare evoultie functie Lyapunov Periodic

V_per = zeros(1, length(t));
for k = 1:length(t)
    V_per(k) = x_per_traj(:,k)' * P * x_per_traj(:,k);
end

figure
plot(t, V_per, 'r', 'LineWidth', 1.5);
legend('V(x) ETC','Evenimente de declansare', "V(x)", 'Location','northeast');

%% Plotare derivata functiei Lyapunov

Vdot_ev = zeros(1, length(t));
for k = 1:length(t)
    xk = x_ev_traj(:,k);
    uk = u_ev_traj(k);
    xdot = A*xk + B*uk;
    Vdot_ev(k) = 2 * xk' * P * xdot;
end

figure;
plot(t, Vdot_ev, 'b', 'LineWidth',1.5);
xlabel('Timp [s]');
ylabel('V(x) derivat');
title('Derivata functiei Lyapunov in timp (ETC)');
grid on;

% Vdot_ref = -lambda * V_ev;
for k = 1:length(t)
    xk = x_per_traj(:,k);
    Vdot_ref(k) = xk' * Q_lyap * xk;
end
hold on;
plot(t, Vdot_ref, 'r--', 'LineWidth',1.3);
legend('V(x) derivat','Referinta','Location','best');

%% INTREBARI

% Tot nu imi dau seama de unde apare coltul la x_2? Sa fie oare de la
% metoda de integrare? Cu toate ca ma cam indoiesc.









