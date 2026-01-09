%% ============================================================
%  Scalar NCS demo: "bad ADC" uniform quantization on state only
%  Plant: x_{k+1} = a x_k + b u_k + w_k
%  Remote control: u_k = -K * xhat_k
%
%  Compare 3 reconstruction strategies from quantized ADC x_q,k = Q(x_k):
%    (1) Midpoint (naive):           xhat_k = x_q,k
%    (2) Interval observer (set):    x_k in [x_q-Δ/2, x_q+Δ/2], propagate+intersect
%    (3) Nonlinear ext-RLS (PAA):    xhat_k = x_q,k + (Δ/2)*tanh(phi_k'*theta_k)
%         phi_k = [xq_k, xq_{k-1}, ..., xq_{k-nx}, u_{k-1}, ..., u_{k-nu+1}]'
%
%  Single compact file. No reference tracking. Just regulation from x0.
%  Requires NO toolboxes.
%% ============================================================

clear; clc; close all;
rng(1);

%% -------------------- Plant / Controller --------------------
a = 1.05;     % open-loop unstable
b = 1.00;

% K = 1.20;     % stabilizing (a-bK = -0.15)
K = acker(a,b,0.95);
fprintf('a = %.3f, b = %.3f, K = %.3f, (a-bK)=%.3f\n', a, b, K, a-b*K);

%% -------------------- ADC uniform quantizer --------------------
Xmax  = 6;        % saturation range
bits  = 6;        % bits for ADC
Delta = 2*Xmax / 2^bits;
fprintf('ADC: bits=%d, range=[-%.1f,%.1f], Delta=%.4f\n', bits, Xmax, Xmax, Delta);

%% -------------------- Disturbance / simulation horizon --------------------
N     = 1000;
x0    = 3.5;
wmax  = 0.02;                      % bounded disturbance for interval observer
w     = wmax*(2*rand(1,N)-1);      % same noise for all cases

%% -------------------- Cost (for comparison) --------------------
qx = 1.0;   ru = 0.05;

%% ============================================================
%  Run all 3 cases with the same disturbance sequence
%% ============================================================
out1 = sim_case_midpoint(a,b,K,x0,w,Delta,Xmax,qx,ru);
out2 = sim_case_interval(a,b,K,x0,w,Delta,Xmax,wmax,qx,ru);
out3 = sim_case_extRLS(a,b,K,x0,w,Delta,Xmax,qx,ru);

fprintf('\nCosts J = sum(qx*x^2 + ru*u^2)\n');
fprintf('  (1) Midpoint:      J = %.4f\n', out1.J);
fprintf('  (2) Interval:      J = %.4f\n', out2.J);
fprintf('  (3) ext-RLS (PAA): J = %.4f\n', out3.J);

%% ============================================================
%  Plots
%% ============================================================
t = 0:N;

figure('Name','State: true, quantized, reconstructed');
plot(t, out1.x, 'LineWidth', 1.3); hold on;
stairs(t, out1.xq, '--', 'LineWidth', 1.1); % quantized measurement
plot(t, out1.xhat, 'LineWidth', 1.3);
plot(t, out2.xhat, 'LineWidth', 1.3);
plot(t, out3.xhat, 'LineWidth', 1.3);
grid on; xlabel('k'); ylabel('x');
title('State reconstruction from a bad ADC (uniform quantization)');
legend('x (true)','x_q = Q(x)','xhat midpoint','xhat interval','xhat ext-RLS','Location','best');

figure('Name','Control input');
stairs(0:N-1, out1.u, 'LineWidth', 1.2); hold on;
stairs(0:N-1, out2.u, 'LineWidth', 1.2);
stairs(0:N-1, out3.u, 'LineWidth', 1.2);
grid on; xlabel('k'); ylabel('u');
title('Control input u_k = -K xhat_k');
legend('midpoint','interval','ext-RLS','Location','best');

figure('Name','Absolute reconstruction error');
plot(t, abs(out1.x - out1.xhat), 'LineWidth', 1.2); hold on;
plot(t, abs(out2.x - out2.xhat), 'LineWidth', 1.2);
plot(t, abs(out3.x - out3.xhat), 'LineWidth', 1.2);
grid on; xlabel('k'); ylabel('|x - xhat|');
title('Reconstruction error');
legend('midpoint','interval','ext-RLS','Location','best');

figure('Name','Interval observer set (sanity)');
% show the interval bounds vs true x for the interval case
plot(t, out2.x, 'LineWidth', 1.2); hold on;
plot(t, out2.xL, '--', 'LineWidth', 1.0);
plot(t, out2.xU, '--', 'LineWidth', 1.0);
grid on; xlabel('k'); ylabel('x / interval');
title('Interval observer: maintained feasible set');
legend('x (true)','lower bound','upper bound','Location','best');

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
    % Maintains an interval [xL_k, xU_k] consistent with:
    %   measurement cell Ik = [xq-Δ/2, xq+Δ/2] (clipped),
    %   bounded disturbance |w_k|<=wmax,
    %   known input u_{k-1}.
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

        % true evolution (uses the same w sequence as other methods)
        x(k+1) = a*x(k) + b*u(k) + w(k);
        xq(k+1)= q_uniform(x(k+1), Delta, Xmax);

        % measurement cell at k+1
        [mL,mU] = cell_from_xq(xq(k+1), Delta, Xmax);

        % predict interval using model + bounded w
        [pL,pU] = propagate_interval(a,b,xL(k),xU(k),u(k),wmax);

        % intersect predicted with measurement-consistent set
        [xL(k+1), xU(k+1)] = intersect_intervals(pL,pU,mL,mU);

        % fallback if empty (can happen with saturation or too-small wmax)
        if xL(k+1) > xU(k+1)
            xL(k+1)=mL; xU(k+1)=mU;
        end

        xhat(k+1) = 0.5*(xL(k+1)+xU(k+1));
        J = J + qx*x(k)^2 + ru*u(k)^2;
    end

    out.x=x; out.xq=xq; out.xhat=xhat; out.u=u; out.J=J;
    out.xL=xL; out.xU=xU;
end

function out = sim_case_extRLS(a,b,K,x0,w,Delta,Xmax,qx,ru)
    % Nonlinear extended-RLS (Gauss-Newton RLS) on:
    %   xhat_k = xq_k + (Δ/2)*tanh(phi_k'*theta)
    % Uses "soft" one-step prediction error:
    %   e_{k+1} = xq_{k+1} - (a*xhat_k + b*u_k)
    N = numel(w);
    x    = zeros(1,N+1); x(1)=x0;
    xq   = zeros(1,N+1);
    xhat = zeros(1,N+1);
    u    = zeros(1,N);
    J = 0;

    % feature sizes
    nx = 10;     % number of xq lags (xq_k, xq_{k-1}, xq_{k-2})
    nu = 10;     % number of u lags  (u_{k-1}, u_{k-2})
    d  = (nx+1) + nu;   % total features
    theta = zeros(d,1);
    P = 100*eye(d);     % big initial uncertainty
    lambda = 0.995;     % forgetting

    % histories (initialize with zeros)
    xq_hist = zeros(nx+1,1);
    u_hist  = zeros(nu,1);

    % k=0 init
    xq(1) = q_uniform(x(1), Delta, Xmax);
    xq_hist = shift_in(xq_hist, xq(1));
    phi = [xq_hist; u_hist];
    [xhat(1), s, fp] = recon_nonlinear(xq(1), phi, theta, Delta);

    for k=1:N
        u(k) = -K*xhat(k);

        % true plant
        x(k+1) = a*x(k) + b*u(k) + w(k);
        xq(k+1)= q_uniform(x(k+1), Delta, Xmax);

        % --- RLS update uses e_{k+1} built from current xhat_k and new xq_{k+1}
        e = xq(k+1) - (a*xhat(k) + b*u(k));  % "soft" residual

        % local regressor psi = d(xhat_k)/d(theta) = fp * phi
        psi = fp * phi;

        % extended RLS update
        denom = lambda + (psi.'*P*psi);
        Kgain = (P*psi)/denom;
        theta = theta + Kgain*e;
        P = (P - Kgain*(psi.'*P))/lambda;

        % update histories for next reconstruction
        xq_hist = shift_in(xq_hist, xq(k+1));
        u_hist  = shift_in(u_hist, u(k));

        phi = [xq_hist; u_hist];
        [xhat(k+1), s, fp] = recon_nonlinear(xq(k+1), phi, theta, Delta);

        J = J + qx*x(k)^2 + ru*u(k)^2;
    end

    out.x=x; out.xq=xq; out.xhat=xhat; out.u=u; out.J=J;
    out.theta=theta;
end

function xq = q_uniform(x, Delta, Xmax)
    % mid-tread uniform with saturation
    x = min(max(x, -Xmax), Xmax);
    xq = Delta * round(x/Delta);
    xq = min(max(xq, -Xmax), Xmax);
end

function [cL,cU] = cell_from_xq(xq, Delta, Xmax)
    % measurement-consistent cell around reconstruction level xq
    cL = xq - 0.5*Delta;
    cU = xq + 0.5*Delta;
    % clip to ADC range
    cL = max(cL, -Xmax);
    cU = min(cU,  Xmax);
end

function [pL,pU] = propagate_interval(a,b,xL,xU,u,wmax)
    % propagate interval through x+ = a x + b u + w, |w|<=wmax
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
    % shifts down and inserts newval at the top
    v = [newval; v(1:end-1)];
end

function [xhat, s, fp] = recon_nonlinear(xq, phi, theta, Delta)
    % xhat = xq + (Δ/2)*tanh(phi'*theta)
    s = phi.'*theta;
    t = tanh(s);
    xhat = xq + 0.5*Delta*t;

    % derivative d/ds of (Δ/2)*tanh(s)
    fp = 0.5*Delta*(1 - t^2);
end
