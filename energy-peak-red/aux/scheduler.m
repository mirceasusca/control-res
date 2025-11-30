%% demo_peak_shaving_with_MILP.m
% Demonstration of peak reduction using MILP on top of a forecasted baseline.
% Assumes you already have t (hours) and L (baseline load, e.g. L_fore_smart).
% If not, this script can generate a simple example baseline.

% clear; 
clc; close all;

%% ------------------------------------------------------------------------
% 1. Baseline input (t, L)
% -------------------------------------------------------------------------
try
    % If t and L already exist in the base workspace, grab them
    t = evalin('base', 't');
    L = evalin('base', 'L_fore_smart');
    fprintf('Using baseline from workspace (L_fore_smart).\n');
catch
    % Fall back to generating a simple example baseline
    fprintf('No baseline found in workspace. Generating example baseline.\n');
    T = 96;
    t = linspace(0,24,T);    % hours

    morning_peak = 1.5 * exp(-(t - 8).^2 / (2*1.2^2));
    evening_peak = 2.5 * exp(-(t - 19).^2 / (2*1.5^2));
    night_background = 0.5 + 0.1*sin(0.5*t);
    noise = 0.15 * randn(size(t));

    L_true = morning_peak + evening_peak + night_background + noise;
    L = max(L_true, 0.2);    % ensure no negative values
end

T = numel(t);
dt = t(2) - t(1);           % assume uniform grid [hours]

%% ------------------------------------------------------------------------
% 2. Configurable consumers (controllable appliances)
% -------------------------------------------------------------------------
rng(1);  % for reproducibility

numConsumers = 5;   % <<< CONFIGURABLE number of consumers

% Rated powers (kW) for each consumer (random between 1.5 and 3.0 kW)
A_kW = 1.5 + (3.0 - 1.5)*rand(numConsumers,1);

% Required durations (in hours), random between 1 and 3 hours
dur_h = 1 + 2*rand(numConsumers,1);

% Convert durations to time slots (round up to integer)
d_slots = ceil(dur_h / dt);

fprintf('\nConsumers:\n');
for i = 1:numConsumers
    fprintf('  Consumer %d: A = %.2f kW, duration = %.2f h (%d slots)\n', ...
        i, A_kW(i), dur_h(i), d_slots(i));
end

%% ------------------------------------------------------------------------
% 3. "Unscheduled" reference case:
%    All consumers start around the global peak of the baseline
% -------------------------------------------------------------------------
[~, idx_peak_baseline] = max(L);

x_unsched = zeros(numConsumers, T);  % ON/OFF schedule (unscheduled reference)

for i = 1:numConsumers
    d = d_slots(i);
    % Start as close as possible to the baseline peak
    start_idx = idx_peak_baseline;
    
    % If it doesn't fit, shift earlier
    if start_idx + d - 1 > T
        start_idx = T - d + 1;
    end
    
    x_unsched(i, start_idx:start_idx+d-1) = 1;
end

total_load_unsched = L(:)' + sum(A_kW .* x_unsched, 1);
P_unsched = max(total_load_unsched);

%% ------------------------------------------------------------------------
% 4. MILP-optimized schedule
% -------------------------------------------------------------------------
[x_opt, y_opt, P_opt] = schedule_peak_shaving(L(:), A_kW, d_slots);

total_load_sched = L(:)' + sum(A_kW .* x_opt, 1);
P_sched = max(total_load_sched);

%% ------------------------------------------------------------------------
% 5. Print peak comparison
% -------------------------------------------------------------------------
P_baseline = max(L);

fprintf('\nPeak comparison:\n');
fprintf('  Baseline peak (no controllable loads)  : %.3f kW\n', P_baseline);
fprintf('  Unscheduled peak (all at baseline peak): %.3f kW\n', P_unsched);
fprintf('  MILP-scheduled peak                    : %.3f kW\n', P_sched);

%% ------------------------------------------------------------------------
% 6. Plot results
% -------------------------------------------------------------------------
figure;
subplot(2,1,1);
plot(t, L, 'k-', 'LineWidth', 2); hold on;
plot(t, total_load_unsched, 'r--', 'LineWidth', 1.5);
plot(t, total_load_sched, 'b-.', 'LineWidth', 1.8);
grid on;
xlabel('Time [hours]');
ylabel('Load [kW]');
title('Baseline vs. Unscheduled vs. MILP-Scheduled Total Load');
legend('Baseline (forecast)', ...
       sprintf('Unscheduled (peak = %.2f)', P_unsched), ...
       sprintf('MILP scheduled (peak = %.2f)', P_sched), ...
       'Location', 'NorthWest');

subplot(2,1,2);
plot(t, sum(A_kW .* x_unsched,1), 'r--', 'LineWidth', 1.5); hold on;
plot(t, sum(A_kW .* x_opt,1), 'b-.', 'LineWidth', 1.8);
grid on;
xlabel('Time [hours]');
ylabel('Controllable load [kW]');
title('Unscheduled vs. MILP-Scheduled Controllable Loads');
legend('Unscheduled', 'MILP scheduled', 'Location', 'NorthWest');

fprintf('\nDone. The plot shows how the MILP shifts controllable loads\n');
fprintf('away from the baseline peak to reduce the total daily peak.\n');

%% ========================================================================
% Local function: MILP for peak shaving
% ========================================================================
function [x_opt, y_opt, P_opt] = schedule_peak_shaving(L0, A_kW, d_slots)
% schedule_peak_shaving
%   Solve MILP: minimize peak load P given baseline L0 and controllable loads
%
% Inputs:
%   L0      - T x 1 baseline load (column vector)
%   A_kW    - N x 1 vector of appliance powers
%   d_slots - N x 1 vector of durations in time slots
%
% Outputs:
%   x_opt   - N x T ON/OFF schedule (binary)
%   y_opt   - N x T start indicators (binary)
%   P_opt   - optimal peak load (scalar)

    L0 = L0(:);              % T x 1
    [N, ~] = size(A_kW);     % N appliances
    T = numel(L0);
    A_kW = A_kW(:);

    % Indexing:
    %   y_{i,k}: i=1..N, k=1..T       -> indices 1..(N*T)
    %   x_{i,k}: i=1..N, k=1..T       -> indices (N*T+1)..(2*N*T)
    %   P:                              index 2*N*T + 1

    nY   = N*T;
    nX   = N*T;
    idxP = nY + nX + 1;
    nVar = idxP;

    % Objective: minimize P
    f = zeros(nVar,1);
    f(idxP) = 1;

    % Integer/binary variables: all x and y
    intcon = 1:(nY+nX);

    % Bounds
    lb = zeros(nVar,1);
    ub = ones(nVar,1);
    ub(idxP) = Inf;     % peak can be >1

    % -----------------------------
    % Equality constraints Aeq z = beq
    % 1) sum_k y_{i,k} = 1   (start once)
    % 2) sum_k x_{i,k} = d_i (duration)
    % 3) link x and y: x_{i,k} = sum_{tau=max(1,k-d_i+1)..k} y_{i,tau}
    % -----------------------------
    nEq_start = N;
    nEq_dur   = N;
    nEq_link  = N*T;
    nEq       = nEq_start + nEq_dur + nEq_link;

    Aeq = sparse(nEq, nVar);
    beq = zeros(nEq,1);
    row = 0;

    % 1) start-once constraints
    for i = 1:N
        row = row + 1;
        for k = 1:T
            idxY = (i-1)*T + k;
            Aeq(row, idxY) = 1;
        end
        beq(row) = 1;
    end

    % 2) duration constraints
    for i = 1:N
        row = row + 1;
        for k = 1:T
            idxX = nY + (i-1)*T + k;
            Aeq(row, idxX) = 1;
        end
        beq(row) = d_slots(i);
    end

    % 3) link constraints
    for i = 1:N
        di = d_slots(i);
        for k = 1:T
            row = row + 1;
            idxX = nY + (i-1)*T + k;
            Aeq(row, idxX) = 1;

            tau_min = max(1, k-di+1);
            tau_max = k;
            for tau = tau_min:tau_max
                idxY = (i-1)*T + tau;
                Aeq(row, idxY) = Aeq(row, idxY) - 1;
            end
            beq(row) = 0;
        end
    end

    % -----------------------------
    % Inequality constraints Aineq z <= bineq
    %   L0(k) + sum_i A_i x_{i,k} <= P
    % => sum_i A_i x_{i,k} - P <= -L0(k)
    % -----------------------------
    Aineq = sparse(T, nVar);
    bineq = zeros(T,1);

    for k = 1:T
        for i = 1:N
            idxX = nY + (i-1)*T + k;
            Aineq(k, idxX) = Aineq(k, idxX) + A_kW(i);
        end
        Aineq(k, idxP) = -1;
        bineq(k) = -L0(k);
    end

    % -----------------------------
    % Solve MILP
    % -----------------------------
    options = optimoptions('intlinprog', ...
        'Display', 'off', ...
        'Heuristics', 'advanced', ...
        'CutGeneration', 'intermediate');

    [z_opt, fval, exitflag] = intlinprog(f, intcon, ...
        Aineq, bineq, Aeq, beq, lb, ub, options);

    if exitflag <= 0
        warning('intlinprog did not converge to an optimal solution. Exitflag = %d', exitflag);
    end

    % Extract solution
    z_opt(isnan(z_opt)) = 0;
    y_vec = z_opt(1:nY);
    x_vec = z_opt(nY+1:nY+nX);
    P_opt = z_opt(idxP);

    % Reshape into N x T
    y_opt = reshape(y_vec, [T, N])';
    x_opt = reshape(x_vec, [T, N])';

    % Numerical safety
    y_opt = round(y_opt);
    x_opt = round(x_opt);
end