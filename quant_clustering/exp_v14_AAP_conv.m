%% ============================================================
%  Scalar NCS demo: BAD ADC only (uniform quantization on state)
%  Plant: x_{k+1} = a x_k + b u_k + w_k
%  Remote control: u_k = -K * xhat_k
%  ADC reports: x_q,k = Q(x_k)  (uniform, saturated)
%
%  Compare:
%    (1) Midpoint:       xhat_k = x_q,k
%    (2) Interval set:   x in [xq-Δ/2,xq+Δ/2], propagate + intersect
%    (3) PAA ext-RLS:    xhat_k = xq_k + sat_{[-Δ/2,Δ/2]}( (Δ/2)*tanh(phi'*theta) )
%
%  UPGRADES INCLUDED:
%   (A) Training uses control probing (dither) to excite the ADC cells.
%   (B) Quantization-matching residual:
%       e_{k+1} = xq_{k+1} - Q( a*xhat_k + b*u_k )
%
%  No toolboxes required.
%% ============================================================

clear; clc; close all;
rng(1);

%% -------------------- Plant / Controller --------------------
a = 1.05;     % open-loop unstable
b = 1.00;
K = acker(a,b,0.95);
fprintf('a = %.3f, b = %.3f, K = %.3f, (a-bK)=%.3f\n', a, b, K, a-b*K);

%% -------------------- ADC uniform quantizer --------------------
Xmax  = 6;
bits  = 6;
Delta = 2*Xmax / 2^bits;
fprintf('ADC: bits=%d, range=[-%.1f,%.1f], Delta=%.4f\n', bits, Xmax, Xmax, Delta);

%% -------------------- Costs --------------------
qx = 1.0;   ru = 0.05;

%% -------------------- Disturbance bounds --------------------
wmax  = 0.02;

%% ============================================================
%  PAA (ext-RLS) settings
%% ============================================================
nx = 8;          % xq lags: xq_k,...,xq_{k-nx}
nu = 8;          % u  lags: u_{k-1},...,u_{k-nu}
d  = (nx+1) + nu;

lambda = 0.997;      % slightly closer to 1 for stability
P0     = 80;         % smaller than before to avoid aggressive jumps
theta0 = zeros(d,1);

% gating threshold on residual magnitude (skip microscopic updates)
gate_e = 1e-5;

% probing (training only)
dither_max = 0.08;    % try 0.02..0.15; keep small enough for stability

%% ============================================================
%  TRAINING: multiple episodes (theta,P carried across episodes)
%% ============================================================
NtrainEpisodes = 50;
NtrainSteps    = 400;
x0_range       = 5.0;
rng_base       = 100;

theta = theta0;
P     = P0*eye(d);

J_mid_train  = zeros(NtrainEpisodes,1);
J_paa_train  = zeros(NtrainEpisodes,1);
theta_norm   = zeros(NtrainEpisodes,1);
emed_abs     = zeros(NtrainEpisodes,1);

fprintf('\n--- TRAINING PAA over %d episodes ---\n', NtrainEpisodes);

for ep = 1:NtrainEpisodes
    rng(rng_base + ep);

    x0 = (2*rand-1)*x0_range;
    w  = wmax*(2*rand(1,NtrainSteps)-1);

    out_mid = sim_case_midpoint(a,b,K,x0,w,Delta,Xmax,qx,ru);

    [out_paa, theta, P, stats] = sim_case_extRLS_episode_Qmatch( ...
        a,b,K,x0,w,Delta,Xmax,qx,ru, ...
        theta,P,lambda,nx,nu,gate_e, ...
        dither_max);

    J_mid_train(ep) = out_mid.J;
    J_paa_train(ep) = out_paa.J;
    theta_norm(ep)  = norm(theta);
    emed_abs(ep)    = stats.median_abs_e;

    if mod(ep,10)==0 || ep==1
        fprintf('ep=%02d: Jmid=%.3f, Jpaa=%.3f, ||theta||=%.3f, med|e|=%.3g\n', ...
            ep, J_mid_train(ep), J_paa_train(ep), theta_norm(ep), emed_abs(ep));
    end
end

%% ============================================================
%  TEST: fresh episode (theta frozen) + compare all methods
%% ============================================================
fprintf('\n--- TEST (fresh episode, theta frozen, NO dither) ---\n');
rng(999);
Ntest = 1200;
x0    = 3.5;
w     = wmax*(2*rand(1,Ntest)-1);

out1 = sim_case_midpoint(a,b,K,x0,w,Delta,Xmax,qx,ru);
out2 = sim_case_interval(a,b,K,x0,w,Delta,Xmax,wmax,qx,ru);
out3 = sim_case_extRLS_frozen(a,b,K,x0,w,Delta,Xmax,qx,ru,theta,nx,nu);

fprintf('\nCosts J = sum(qx*x^2 + ru*u^2)\n');
fprintf('  (1) Midpoint:      J = %.4f\n', out1.J);
fprintf('  (2) Interval:      J = %.4f\n', out2.J);
fprintf('  (3) PAA (frozen):  J = %.4f\n', out3.J);

%% ============================================================
%  Plots: learning curves
%% ============================================================
figure('Name','Training curves (episode cost)');
plot(1:NtrainEpisodes, J_mid_train, 'LineWidth', 1.2); hold on;
plot(1:NtrainEpisodes, J_paa_train, 'LineWidth', 1.2);
grid on; xlabel('episode'); ylabel('J');
title('Training: episode cost');
legend('midpoint','PAA (learning)','Location','best');

figure('Name','Training: parameter norm');
plot(1:NtrainEpisodes, theta_norm, 'LineWidth', 1.2);
grid on; xlabel('episode'); ylabel('||\theta||_2');
title('Training: parameter norm');

figure('Name','Training: median residual magnitude');
plot(1:NtrainEpisodes, emed_abs, 'LineWidth', 1.2);
grid on; xlabel('episode'); ylabel('median |e_k|');
title('Training: quantization-matching residual (median)');

%% ============================================================
%  Plots: test trajectories
%% ============================================================
t = 0:Ntest;

figure('Name','TEST: state and reconstructions');
plot(t, out1.x, 'LineWidth', 1.3); hold on;
stairs(t, out1.xq, '--', 'LineWidth', 1.1);
plot(t, out1.xhat, 'LineWidth', 1.2);
plot(t, out2.xhat, 'LineWidth', 1.2);
plot(t, out3.xhat, 'LineWidth', 1.2);
grid on; xlabel('k'); ylabel('x');
title('TEST: reconstruction from bad ADC');
legend('x true','x_q','xhat midpoint','xhat interval','xhat PAA(frozen)','Location','best');

figure('Name','TEST: absolute reconstruction error');
plot(t, abs(out1.x-out1.xhat), 'LineWidth', 1.2); hold on;
plot(t, abs(out2.x-out2.xhat), 'LineWidth', 1.2);
plot(t, abs(out3.x-out3.xhat), 'LineWidth', 1.2);
grid on; xlabel('k'); ylabel('|x-xhat|');
title('TEST: reconstruction error');
legend('midpoint','interval','PAA(frozen)','Location','best');

figure('Name','TEST: control input');
stairs(0:Ntest-1, out1.u, 'LineWidth', 1.1); hold on;
stairs(0:Ntest-1, out2.u, 'LineWidth', 1.1);
stairs(0:Ntest-1, out3.u, 'LineWidth', 1.1);
grid on; xlabel('k'); ylabel('u');
title('TEST: control u_k=-K xhat_k');
legend('midpoint','interval','PAA(frozen)','Location','best');

figure('Name','TEST: interval bounds sanity');
plot(t, out2.x, 'LineWidth', 1.2); hold on;
plot(t, out2.xL, '--', 'LineWidth', 1.0);
plot(t, out2.xU, '--', 'LineWidth', 1.0);
grid on; xlabel('k'); ylabel('x / bounds');
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
    xL(1)=iL; xU(1)=iU; xhat(1)=0.5*(xL(1)+xU(1));

    for k=1:N
        u(k) = -K*xhat(k);

        x(k+1) = a*x(k) + b*u(k) + w(k);
        xq(k+1)= q_uniform(x(k+1), Delta, Xmax);

        [mL,mU] = cell_from_xq(xq(k+1), Delta, Xmax);
        [pL,pU] = propagate_interval(a,b,xL(k),xU(k),u(k),wmax);
        [xL(k+1), xU(k+1)] = intersect_intervals(pL,pU,mL,mU);

        if xL(k+1) > xU(k+1)
            xL(k+1)=mL; xU(k+1)=mU;
        end

        xhat(k+1) = 0.5*(xL(k+1)+xU(k+1));
        J = J + qx*x(k)^2 + ru*u(k)^2;
    end

    out.x=x; out.xq=xq; out.xhat=xhat; out.u=u; out.J=J;
    out.xL=xL; out.xU=xU;
end

function [out, theta, P, stats] = sim_case_extRLS_episode_Qmatch( ...
        a,b,K,x0,w,Delta,Xmax,qx,ru,theta,P,lambda,nx,nu,gate_e,dither_max)

    % Episode that UPDATES theta,P using quantization-matching residual:
    %   e_{k+1} = xq_{k+1} - Q( a*xhat_k + b*u_k )

    N = numel(w);
    x    = zeros(1,N+1); x(1)=x0;
    xq   = zeros(1,N+1);
    xhat = zeros(1,N+1);
    u    = zeros(1,N);
    J = 0;

    xq_hist = zeros(nx+1,1);
    u_hist  = zeros(nu,1);

    xq(1) = q_uniform(x(1), Delta, Xmax);
    xq_hist = shift_in(xq_hist, xq(1));
    phi = [xq_hist; u_hist];

    [xhat(1), ~, fp] = recon_nonlinear(xq(1), phi, theta, Delta);

    e_store = zeros(1,N);

    for k=1:N
        % training-only probing dither
        dither = dither_max * (2*rand-1);

        u(k) = -K*xhat(k) + dither;

        % true plant
        x(k+1) = a*x(k) + b*u(k) + w(k);
        xq(k+1)= q_uniform(x(k+1), Delta, Xmax);

        % predicted next (continuous) state
        xpred_cont = a*xhat(k) + b*u(k);

        % quantization-matching residual
        xpred_q = q_uniform(xpred_cont, Delta, Xmax);
        e = xq(k+1) - xpred_q;
        e_store(k) = e;

        % local regressor for GN-RLS: psi = d(xhat_k)/d(theta) = fp * phi
        psi = fp * phi;

        if abs(e) > gate_e
            denom = lambda + (psi.'*P*psi);
            Kgain = (P*psi)/denom;
            theta = theta + Kgain*e;
            P = (P - Kgain*(psi.'*P))/lambda;
        else
            P = P/lambda;
        end

        % update histories
        xq_hist = shift_in(xq_hist, xq(k+1));
        u_hist  = shift_in(u_hist,  u(k));
        phi = [xq_hist; u_hist];

        % reconstruct with guaranteed constraint: delta in [-Δ/2, Δ/2]
        [xhat_raw, ~, fp] = recon_nonlinear(xq(k+1), phi, theta, Delta);
        delta_hat = xhat_raw - xq(k+1);
        delta_hat = max(min(delta_hat, 0.5*Delta), -0.5*Delta);
        xhat(k+1) = xq(k+1) + delta_hat;

        J = J + qx*x(k)^2 + ru*u(k)^2;
    end

    out.x=x; out.xq=xq; out.xhat=xhat; out.u=u; out.J=J;
    stats.median_abs_e = median(abs(e_store));
end

function out = sim_case_extRLS_frozen(a,b,K,x0,w,Delta,Xmax,qx,ru,theta,nx,nu)
    % Inference only: theta fixed, NO updates, NO dither
    N = numel(w);
    x    = zeros(1,N+1); x(1)=x0;
    xq   = zeros(1,N+1);
    xhat = zeros(1,N+1);
    u    = zeros(1,N);
    J = 0;

    xq_hist = zeros(nx+1,1);
    u_hist  = zeros(nu,1);

    xq(1) = q_uniform(x(1), Delta, Xmax);
    xq_hist = shift_in(xq_hist, xq(1));
    phi = [xq_hist; u_hist];

    [xhat(1), ~, ~] = recon_nonlinear(xq(1), phi, theta, Delta);

    for k=1:N
        u(k) = -K*xhat(k);

        x(k+1) = a*x(k) + b*u(k) + w(k);
        xq(k+1)= q_uniform(x(k+1), Delta, Xmax);

        xq_hist = shift_in(xq_hist, xq(k+1));
        u_hist  = shift_in(u_hist,  u(k));
        phi = [xq_hist; u_hist];

        [xhat_raw, ~, ~] = recon_nonlinear(xq(k+1), phi, theta, Delta);

        delta_hat = xhat_raw - xq(k+1);
        delta_hat = max(min(delta_hat, 0.5*Delta), -0.5*Delta);
        xhat(k+1) = xq(k+1) + delta_hat;

        J = J + qx*x(k)^2 + ru*u(k)^2;
    end

    out.x=x; out.xq=xq; out.xhat=xhat; out.u=u; out.J=J;
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
    fp = 0.5*Delta*(1 - t^2);
end
