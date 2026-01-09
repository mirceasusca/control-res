%% ============================================================
%  Scalar NCS demo: "bad ADC" uniform quantization on state only
%  Plant: x_{k+1} = a x_k + b u_k + w_k
%  Remote control: u_k = -K * xhat_k
%
%  Compare reconstruction from quantized ADC x_q,k = Q(x_k):
%    (1) Midpoint (baseline):        xhat_k = x_q,k
%    (2) Interval observer (set):    x_k in [x_q-Δ/2, x_q+Δ/2], propagate+intersect
%    (3) PAA ext-RLS (trained):      xhat_k = x_q,k + (Δ/2)*tanh(phi_k'*theta)
%         phi_k = [xq_k..xq_{k-nx}, u_{k-1}..u_{k-nu}]'
%
%  Upgrade: Train PAA over multiple experiments (episodes), then freeze theta.
%  Requires NO toolboxes.
%% ============================================================
clear; clc; close all;
rng(1);

%% -------------------- Plant / Controller --------------------
a = 1.05;     % open-loop unstable
b = 1.00;

p_des = 0.95;                 % desired closed-loop pole
K = (a - p_des)/b;            % scalar pole placement, no toolbox
fprintf('a=%.3f, b=%.3f, K=%.3f, closed-loop (a-bK)=%.3f\n', a, b, K, a-b*K);

%% -------------------- ADC uniform quantizer --------------------
Xmax  = 6;                     % saturation range
bits  = 6;                     % ADC bits
Delta = 2*Xmax / 2^bits;
fprintf('ADC: bits=%d, range=[-%.1f,%.1f], Delta=%.4f\n', bits, Xmax, Xmax, Delta);

%% -------------------- Disturbance model --------------------
wmax_train = 0.02;             % known bound (interval observer)
wmax_test  = 0.02;

%% -------------------- Cost (for comparison) --------------------
qx = 1.0;   ru = 0.05;

%% -------------------- PAA / ext-RLS hyperparams --------------------
nx = 5;                % xq lags (xq_k..xq_{k-nx})
nu = 5;                % u  lags (u_{k-1}..u_{k-nu})
d  = (nx+1) + nu;        % feature dimension

lambda = 0.995;         % forgetting factor
P0     = 1000;           % initial covariance scale

% "Guaranteed constraint" (practical): keep theta bounded (projection)
theta_max = 50;         % ||theta||_2 <= theta_max

% Gate updates to avoid dead-zones / saturation / near-zero derivative
gate_enable = true;
gate_min_abs_xq = 0.5*Delta;    % need some excitation
gate_no_saturation = true;      % skip updates if ADC saturates

%% ============================================================
%  TRAINING: multiple experiments for PAA (ext-RLS)
%% ============================================================
M_train = 2000;          % number of episodes
N_train = 400;          % length each episode
x0_span = 5.0;          % random initial conditions in [-x0_span, x0_span]

theta = zeros(d,1);
P = P0*eye(d);

J_train = zeros(M_train,1);
theta_norm = zeros(M_train,1);

fprintf('\n--- TRAIN PAA over %d episodes ---\n', M_train);
for ep = 1:M_train
    % random episode conditions
    x0 = (2*rand-1)*x0_span;
    w  = wmax_train*(2*rand(1,N_train)-1);

    % train using the PAA case (online updates enabled)
    cfg = struct();
    cfg.do_update = true;
    cfg.lambda = lambda;
    cfg.theta_max = theta_max;
    cfg.gate_enable = gate_enable;
    cfg.gate_min_abs_xq = gate_min_abs_xq;
    cfg.gate_no_saturation = gate_no_saturation;

    out = sim_case_extRLS(a,b,K,x0,w,Delta,Xmax,qx,ru,nx,nu,theta,P,cfg);

    theta = out.theta;
    P     = out.P;

    J_train(ep) = out.J;
    theta_norm(ep) = norm(theta);

    if mod(ep,25)==0
        fprintf('  ep %4d/%4d: J=%.3f, ||theta||=%.3f, updates=%d\n',...
            ep, M_train, J_train(ep), theta_norm(ep), out.n_updates);
    end
end

theta_frozen = theta;   % freeze after training

%% ============================================================
%  TEST: single long episode, compare midpoint / interval / PAA(frozen)
%% ============================================================
N_test = 1200;
x0_test = 3.5;
w_test  = wmax_test*(2*rand(1,N_test)-1);

out_mid = sim_case_midpoint(a,b,K,x0_test,w_test,Delta,Xmax,qx,ru);
out_int = sim_case_interval(a,b,K,x0_test,w_test,Delta,Xmax,wmax_test,qx,ru);

% PAA frozen: no parameter updates, just apply learned theta_frozen
cfgT = struct();
cfgT.do_update = false;
cfgT.lambda = lambda;
cfgT.theta_max = theta_max;
cfgT.gate_enable = false;
cfgT.gate_min_abs_xq = 0;
cfgT.gate_no_saturation = false;

out_paa = sim_case_extRLS(a,b,K,x0_test,w_test,Delta,Xmax,qx,ru,nx,nu,theta_frozen,P0*eye(d),cfgT);

fprintf('\n--- TEST costs J = sum(qx*x^2 + ru*u^2) ---\n');
fprintf('  (1) Midpoint baseline:   J = %.4f\n', out_mid.J);
fprintf('  (2) Interval observer:   J = %.4f\n', out_int.J);
fprintf('  (3) PAA (frozen theta):  J = %.4f\n', out_paa.J);

%% ============================================================
%  Plots: training convergence
%% ============================================================
figure('Name','TRAIN: episode cost and parameter norm');
subplot(2,1,1);
plot(1:M_train, J_train, 'LineWidth', 1.2); grid on;
xlabel('training episode'); ylabel('J (episode)');
title('TRAIN: PAA episode cost');

subplot(2,1,2);
plot(1:M_train, theta_norm, 'LineWidth', 1.2); grid on;
xlabel('training episode'); ylabel('||\theta||_2');
title('TRAIN: parameter norm');

%% ============================================================
%  Plots: test signals
%% ============================================================
t = 0:N_test;

figure('Name','TEST: reconstruction from bad ADC');
plot(t, out_mid.x, 'LineWidth', 1.2); hold on;
stairs(t, out_mid.xq, '--', 'LineWidth', 1.0);
plot(t, out_mid.xhat, 'LineWidth', 1.2);
plot(t, out_int.xhat, 'LineWidth', 1.2);
plot(t, out_paa.xhat, 'LineWidth', 1.2);
grid on; xlabel('k'); ylabel('x');
title('TEST: reconstruction from bad ADC');
legend('x true','x_q','xhat midpoint','xhat interval','xhat PAA(frozen)','Location','best');

figure('Name','TEST: control u_k=-K xhat_k');
stairs(0:N_test-1, out_mid.u, 'LineWidth', 1.1); hold on;
stairs(0:N_test-1, out_int.u, 'LineWidth', 1.1);
stairs(0:N_test-1, out_paa.u, 'LineWidth', 1.1);
grid on; xlabel('k'); ylabel('u');
title('TEST: control  u_k=-K xhat_k');
legend('midpoint','interval','PAA(frozen)','Location','best');

figure('Name','TEST: reconstruction error');
plot(t, abs(out_mid.x - out_mid.xhat), 'LineWidth', 1.1); hold on;
plot(t, abs(out_int.x - out_int.xhat), 'LineWidth', 1.1);
plot(t, abs(out_paa.x - out_paa.xhat), 'LineWidth', 1.1);
grid on; xlabel('k'); ylabel('|x-xhat|');
title('TEST: reconstruction error');
legend('midpoint','interval','PAA(frozen)','Location','best');

figure('Name','TEST: interval observer bounds (sanity)');
plot(t, out_int.x, 'LineWidth', 1.1); hold on;
plot(t, out_int.xL, '--', 'LineWidth', 1.0);
plot(t, out_int.xU, '--', 'LineWidth', 1.0);
grid on; xlabel('k'); ylabel('x / interval');
title('TEST: interval observer feasible set');
legend('x true','lower','upper','Location','best');

%% ============================================================
%  Local functions
%% ============================================================

function out = sim_case_midpoint(a,b,K,x0,w,Delta,Xmax,qx,ru)
    N = numel(w);
    x    = zeros(1,N+1); x(1)=x0;
    xq   = zeros(1,N+1);
    xhat = zeros(1,N+1);
    u    = zeros(1,N);
    J = 0;

    xq(1)   = q_uniform(x(1), Delta, Xmax);
    xhat(1) = xq(1);

    for k=1:N
        u(k) = -K*xhat(k);
        x(k+1)  = a*x(k) + b*u(k) + w(k);

        xq(k+1)   = q_uniform(x(k+1), Delta, Xmax);
        xhat(k+1) = xq(k+1);

        J = J + qx*x(k)^2 + ru*u(k)^2;
    end

    out.x=x; out.xq=xq; out.xhat=xhat; out.u=u; out.J=J;
end

function out = sim_case_interval(a,b,K,x0,w,Delta,Xmax,wmax,qx,ru)
    N = numel(w);
    x  = zeros(1,N+1); x(1)=x0;
    xq = zeros(1,N+1);
    u  = zeros(1,N);
    xL = zeros(1,N+1);
    xU = zeros(1,N+1);
    xhat = zeros(1,N+1);
    J = 0;

    xq(1) = q_uniform(x(1), Delta, Xmax);
    [iL,iU] = cell_from_xq(xq(1), Delta, Xmax);
    xL(1)=iL; xU(1)=iU; xhat(1)=(xL(1)+xU(1))/2;

    for k=1:N
        u(k) = -K*xhat(k);

        x(k+1) = a*x(k) + b*u(k) + w(k);
        xq(k+1)= q_uniform(x(k+1), Delta, Xmax);

        [mL,mU] = cell_from_xq(xq(k+1), Delta, Xmax);
        [pL,pU] = propagate_interval(a,b,xL(k),xU(k),u(k),wmax);
        [xL(k+1), xU(k+1)] = intersect_intervals(pL,pU,mL,mU);

        if xL(k+1) > xU(k+1)   % fallback
            xL(k+1)=mL; xU(k+1)=mU;
        end

        xhat(k+1) = 0.5*(xL(k+1)+xU(k+1));
        J = J + qx*x(k)^2 + ru*u(k)^2;
    end

    out.x=x; out.xq=xq; out.xhat=xhat; out.u=u; out.J=J;
    out.xL=xL; out.xU=xU;
end

function out = sim_case_extRLS(a,b,K,x0,w,Delta,Xmax,qx,ru,nx,nu,theta,P,cfg)
    % xhat_k = xq_k + (Δ/2)*tanh(phi_k'*theta)
    % "Soft" residual:
    %   e_{k+1} = xq_{k+1} - (a*xhat_k + b*u_k)
    %
    % NOTE: If you instead want "quantization matching residual"
    %   e_{k+1} = xq_{k+1} - Q(a*xhat_k + b*u_k)
    % then the derivative is (almost everywhere) zero; you would need a smooth
    % approximation of Q(·) for a gradient-type update.

    N = numel(w);
    d = (nx+1)+nu;

    x    = zeros(1,N+1); x(1)=x0;
    xq   = zeros(1,N+1);
    xhat = zeros(1,N+1);
    u    = zeros(1,N);
    J = 0;

    xq_hist = zeros(nx+1,1);
    u_hist  = zeros(nu,1);

    n_updates = 0;

    % init
    xq(1) = q_uniform(x(1), Delta, Xmax);
    xq_hist = shift_in(xq_hist, xq(1));
    phi = [xq_hist; u_hist];
    [xhat(1), ~, fp] = recon_nonlinear(xq(1), phi, theta, Delta);

    for k=1:N
        u(k) = -K*xhat(k);

        x(k+1)  = a*x(k) + b*u(k) + w(k);
        xq(k+1) = q_uniform(x(k+1), Delta, Xmax);

        % residual (soft)
        e = xq(k+1) - (a*xhat(k) + b*u(k));

        % update gate
        do_update = cfg.do_update;
        if cfg.gate_enable
            if abs(xq(k+1)) < cfg.gate_min_abs_xq
                do_update = false;
            end
            if cfg.gate_no_saturation
                if abs(xq(k+1)) >= Xmax-1e-12
                    do_update = false;
                end
            end
        end

        if do_update
            psi = fp * phi;   % d(xhat_k)/d(theta) = fp * phi

            denom = cfg.lambda + (psi.'*P*psi);
            Kgain = (P*psi)/denom;
            theta = theta + Kgain*e;
            P = (P - Kgain*(psi.'*P))/cfg.lambda;

            % projection (practical "guaranteed constraint"): ||theta|| <= theta_max
            nt = norm(theta);
            if nt > cfg.theta_max
                theta = (cfg.theta_max/nt)*theta;
            end
            n_updates = n_updates + 1;
        end

        % update histories for next reconstruction
        xq_hist = shift_in(xq_hist, xq(k+1));
        u_hist  = shift_in(u_hist, u(k));
        phi = [xq_hist; u_hist];

        [xhat(k+1), ~, fp] = recon_nonlinear(xq(k+1), phi, theta, Delta);

        J = J + qx*x(k)^2 + ru*u(k)^2;
    end

    out.x=x; out.xq=xq; out.xhat=xhat; out.u=u; out.J=J;
    out.theta=theta; out.P=P; out.n_updates=n_updates;
end

function xq = q_uniform(x, Delta, Xmax)
    x = min(max(x, -Xmax), Xmax);
    xq = Delta * round(x/Delta);
    xq = min(max(xq, -Xmax), Xmax);
end

function [cL,cU] = cell_from_xq(xq, Delta, Xmax)
    cL = max(xq - 0.5*Delta, -Xmax);
    cU = min(xq + 0.5*Delta,  Xmax);
end

function [pL,pU] = propagate_interval(a,b,xL,xU,u,wmax)
    if a >= 0
        pL = a*xL + b*u - wmax;
        pU = a*xU + b*u + wmax;
    else
        pL = a*xU + b*u - wmax;
        pU = a*xL + b*u + wmax;
    end
end

function [iL,iU] = intersect_intervals(aL,aU,bL,bU)
    iL = max(aL,bL);
    iU = min(aU,bU);
end

function v = shift_in(v, newval)
    v = [newval; v(1:end-1)];
end

function [xhat, s, fp] = recon_nonlinear(xq, phi, theta, Delta)
    s = phi.'*theta;
    t = tanh(s);
    xhat = xq + 0.5*Delta*t;
    fp = 0.5*Delta*(1 - t^2);   % d/ds of (Δ/2)*tanh(s)
end
