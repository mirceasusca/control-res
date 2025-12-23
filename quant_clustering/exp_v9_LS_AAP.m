%% ============================================================
%  Scalar NCS demo: quantized state measurement + three decoders
%
%  Plant:   x_{k+1} = a x_k + b u_k
%  Control: u_k = -k * xhat_k   (DAC ideal)
%
%  ADC: uniform quantizer Q(x) with step Delta, saturation at +/-Xmax.
%  Decoder cases (sensor/controller side):
%   1) Midpoint decode:              xhat = xq
%   2) Interval observer (set):      propagate interval + intersect next bin
%   3) Learned bin-bias (bounded):   xhat = xq + sat(theta(bin), +/-Delta/2)
%                                   theta updated online via 1-step residual
%
%  Notes:
%   - Case 2 uses the *next* quantized sample to refine xhat(k). This is a
%     fair "observer" baseline (1-step smoothing). If you want causal-only,
%     comment the smoothing line marked below.
%   - Case 3 is strictly causal (uses only current bin + past).
%
%  Requires: no toolboxes.
%% ============================================================

clear; clc; close all; rng(1);

%% -------------------- System + controller --------------------
a = 1.08;            % open-loop unstable
b = 1.0;

% simple stabilizing feedback
k = 0.55;            % choose so |a-bk|<1
acl = a - b*k;
fprintf('a=%g, b=%g, k=%g, closed-loop (ideal) a-bk=%g\n', a,b,k,acl);

%% -------------------- Quantizer --------------------
Delta = 0.25;        % ADC step
Xmax  = 4.0;         % saturation range

quantize_uniform = @(x) sat(Delta*round(sat(x,-Xmax,Xmax)/Delta), -Xmax, Xmax);
bin_index = @(xq) round(xq/Delta);  % integer index (within saturation)

%% -------------------- Simulation setup --------------------
N = 200;
t = 0:N;

% add small process noise to make it interesting (optional)
sigma_w = 0.02;

x0 = 2.7;

%% ============================================================
%  Ground-truth closed-loop evolution (uses quantized measurement in control)
%  We'll simulate three closed-loops, one per decoder.
%% ============================================================

% --- Case 1: midpoint ---
x1 = zeros(1,N+1); u1 = zeros(1,N); xhat1 = zeros(1,N);
x1(1) = x0;

% --- Case 2: interval observer (set-membership) ---
x2 = zeros(1,N+1); u2 = zeros(1,N); xhat2 = zeros(1,N);
x2(1) = x0;
I2_lo = zeros(1,N+1); I2_hi = zeros(1,N+1);  % maintained posterior interval
I2_lo(1) = -inf; I2_hi(1) = inf;             % will be initialized from first bin

% --- Case 3: learned bin-bias ---
x3 = zeros(1,N+1); u3 = zeros(1,N); xhat3 = zeros(1,N);
x3(1) = x0;

% parameters for bin-bias learning
idx_max = round(Xmax/Delta);
theta = zeros(2*idx_max+1,1);  % theta( idx_max+1 + bin ) corresponds to bin index
eta   = 0.08;                  % learning rate (try 0.02..0.2)
theta_clip = 0.49*(Delta/2);   % keep strictly inside cell (optional margin)

% helper for theta access
theta_at = @(ibin) theta(idx_max+1 + ibin);
set_theta = @(ibin,val) set_theta_impl(ibin,val); %#ok<NASGU>

% store quantized measurements for plotting
xq1 = zeros(1,N); xq2 = zeros(1,N); xq3 = zeros(1,N);

%% -------------------- Run simulations --------------------
for kstep = 1:N
    %% ===== Case 1: midpoint decode =====
    xq = quantize_uniform(x1(kstep));
    xq1(kstep) = xq;
    xhat = xq;
    xhat1(kstep) = xhat;
    u1(kstep) = -k * xhat;
    x1(kstep+1) = a*x1(kstep) + b*u1(kstep) + sigma_w*randn();

    %% ===== Case 2: interval observer (set membership) =====
    % current measurement bin interval
    xq = quantize_uniform(x2(kstep));
    xq2(kstep) = xq;
    I_meas = [xq-Delta/2, xq+Delta/2];

    if kstep == 1
        % initialize posterior interval from first measurement
        I_post = I_meas;
    else
        % propagate previous posterior interval through dynamics:
        % x_{k} in [lo,hi] => x_{k+1} in [a*lo + b*u, a*hi + b*u] (if a>=0)
        lo = I2_lo(kstep); hi = I2_hi(kstep);
        % use last applied control u2(k-1)
        ukm1 = u2(kstep-1);
        [J_lo, J_hi] = affine_interval(a, b*ukm1, lo, hi);

        % intersect with measurement interval
        I_post = intersect_interval([J_lo,J_hi], I_meas);

        % if intersection is empty (noise/model mismatch), inflate a bit
        if isempty(I_post)
            inflate = 0.25*Delta;
            I_post = [I_meas(1)-inflate, I_meas(2)+inflate];
        end
    end

    I2_lo(kstep) = I_post(1);
    I2_hi(kstep) = I_post(2);

    % choose estimate as midpoint of posterior
    xhat = 0.5*(I_post(1)+I_post(2));

    % (optional) 1-step smoothing: enforce xhat to lie in current measurement bin
    % xhat = sat(xhat, I_meas(1), I_meas(2));  % <-- keep if you want stricter set consistency

    xhat2(kstep) = xhat;
    u2(kstep) = -k * xhat;
    x2(kstep+1) = a*x2(kstep) + b*u2(kstep) + sigma_w*randn();

    %% ===== Case 3: learned bin-bias (bounded) =====
    xq = quantize_uniform(x3(kstep));
    xq3(kstep) = xq;
    ib = bin_index(xq);
    ib = sat_int(ib, -idx_max, idx_max);

    bias = sat(theta_at(ib), -Delta/2, Delta/2);
    xhat = xq + bias;                 % decoder output
    xhat3(kstep) = xhat;

    u3(kstep) = -k * xhat;
    x3(kstep+1) = a*x3(kstep) + b*u3(kstep) + sigma_w*randn();

    % Online update using next-step residual r_{k+1} once we have xq_{k+1}
    % We use a one-step prediction error in the *quantized* domain:
    %   xq_{k+1} + bias(bin_{k+1})  ≈  a*(xq_k + bias(bin_k)) + b*u_k
    xq_next = quantize_uniform(x3(kstep+1));
    ib_next = bin_index(xq_next);
    ib_next = sat_int(ib_next, -idx_max, idx_max);

    bias_next = sat(theta_at(ib_next), -Delta/2, Delta/2);

    r = (xq_next + bias_next) - (a*(xq + bias) + b*u3(kstep));

    % gradient-like updates on the two involved bins (very simple):
    % r increases with bias_next (+1) and decreases with bias (-a)
    theta(idx_max+1 + ib_next) = theta(idx_max+1 + ib_next) - eta*(+1)*r;
    theta(idx_max+1 + ib)      = theta(idx_max+1 + ib)      - eta*(-a)*r;

    % hard constraint: keep biases bounded (guaranteed "in-bin" reconstruction)
    theta(idx_max+1 + ib_next) = sat(theta(idx_max+1 + ib_next), -theta_clip, theta_clip);
    theta(idx_max+1 + ib)      = sat(theta(idx_max+1 + ib),      -theta_clip, theta_clip);
end

%% -------------------- Costs (just for sanity) --------------------
J = @(x,u) sum(x(1:end-1).^2 + 0.1*u.^2);  % simple regulation cost
J1 = J(x1,u1);
J2 = J(x2,u2);
J3 = J(x3,u3);

fprintf('\nCost (lower is better):\n');
fprintf('  Midpoint decode:     J = %.4f\n', J1);
fprintf('  Interval observer:   J = %.4f\n', J2);
fprintf('  Learned bin-bias:    J = %.4f\n', J3);

%% ======================= Plots =======================
figure('Name','State trajectories');
plot(t, x1, 'LineWidth', 1.5); hold on;
plot(t, x2, 'LineWidth', 1.5);
plot(t, x3, 'LineWidth', 1.5);
grid on; xlabel('k'); ylabel('x_k');
title('Closed-loop state with quantized ADC');
legend('Midpoint', 'Interval observer', 'Learned bin-bias', 'Location','best');

figure('Name','Control input');
stairs(0:N-1, u1, 'LineWidth', 1.2); hold on;
stairs(0:N-1, u2, 'LineWidth', 1.2);
stairs(0:N-1, u3, 'LineWidth', 1.2);
grid on; xlabel('k'); ylabel('u_k');
title('Control input');
legend('Midpoint', 'Interval observer', 'Learned bin-bias', 'Location','best');

figure('Name','Quantized measurements vs estimates (first 80 steps)');
Kshow = 80;
kk = 1:Kshow;

subplot(3,1,1);
stairs(kk, xq1(kk), 'LineWidth', 1.0); hold on;
plot(kk, xhat1(kk), '.', 'MarkerSize', 9);
grid on; ylabel('x_q, \hat{x}');
title('Case 1: Midpoint decode (\hat{x}=x_q)');
legend('x_q','\hat{x}','Location','best');

subplot(3,1,2);
stairs(kk, xq2(kk), 'LineWidth', 1.0); hold on;
plot(kk, xhat2(kk), '.', 'MarkerSize', 9);
grid on; ylabel('x_q, \hat{x}');
title('Case 2: Interval observer');
legend('x_q','\hat{x}','Location','best');

subplot(3,1,3);
stairs(kk, xq3(kk), 'LineWidth', 1.0); hold on;
plot(kk, xhat3(kk), '.', 'MarkerSize', 9);
grid on; xlabel('k'); ylabel('x_q, \hat{x}');
title('Case 3: Learned bounded bin-bias');
legend('x_q','\hat{x}','Location','best');

figure('Name','Learned bias table');
ibins = (-idx_max:idx_max);
plot(ibins, theta(:), 'LineWidth', 1.4);
grid on; xlabel('bin index i'); ylabel('\theta_i');
title('Learned per-bin bias (bounded)');

%% ======================= Local helpers =======================
function y = sat(x, xmin, xmax)
    y = min(max(x, xmin), xmax);
end

function i = sat_int(i, imin, imax)
    i = min(max(i, imin), imax);
end

function I = intersect_interval(I1, I2)
    lo = max(I1(1), I2(1));
    hi = min(I1(2), I2(2));
    if lo <= hi
        I = [lo, hi];
    else
        I = []; % empty intersection
    end
end

function [lo2, hi2] = affine_interval(a, c, lo, hi)
    % maps x in [lo,hi] through x -> a*x + c
    if a >= 0
        lo2 = a*lo + c;
        hi2 = a*hi + c;
    else
        lo2 = a*hi + c;
        hi2 = a*lo + c;
    end
end

function set_theta_impl(~,~) %#ok<DEFNU>
    % placeholder to keep older MATLAB happy if needed
end
