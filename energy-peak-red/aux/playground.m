%% Peak Shaving with Forecasting – Scalable Example
% Requires Optimization Toolbox (intlinprog)

clear; clc; close all;
rng(0);  % For reproducibility

%% 1. Time discretization
T  = 48;          % number of time slots (e.g. 48 x 30 min = 24h)
dt = 24 / T;      % hours per slot
t_hours = (0:T-1) * dt;

%% 2. True baseline load (kW) for the day (synthetic but realistic)
% Morning bump + evening peak + noise
morning = 0.8 + 0.6 * exp(-((t_hours - 8) .^ 2) / (2 * 2^2));
evening = 1.2 + 1.0 * exp(-((t_hours - 19) .^ 2) / (2 * 2^2));
noise   = 0.2 * randn(1, T);
L_true  = max(0.3, morning + evening + noise);   % keep positive
L_true  = L_true(:);  % column vector T x 1

%% 3. Define controllable appliances
% N appliances, with rated power A_i (kW) and durations in hours
A_kW   = [2.5; 1.5; 2.0; 3.0];     % rated powers
dur_h  = [ 2 ; 1.5; 1 ;  3 ];      % durations in hours
N      = numel(A_kW);

d_slots = round(dur_h / dt);       % duration in time slots (integer)

% (Optional) You can scale N and T easily, just adjust A_kW, dur_h, T.

%% 4. Build forecasting scenarios
% S0: Naive flat forecast (no real information)
L_fore_S0 = mean(L_true) * ones(T, 1);

% S1: Basic forecast (noisy)
L_fore_S1 = L_true + 0.6 * std(L_true) * randn(T, 1);

% S2: Advanced forecast (less noisy, "better" model)
L_fore_S2 = L_true + 0.25 * std(L_true) * randn(T, 1);

% S*: Oracle (perfect forecast = true baseline)
L_fore_Sstar = L_true;

% Make sure all forecasts are non-negative
L_fore_S1 = max(0.01, L_fore_S1);
L_fore_S2 = max(0.01, L_fore_S2);

%% 5. Run optimization for each scenario
scenarios = {'Naive (flat)', 'Basic forecast', 'Advanced forecast', 'Oracle'};
L_fore_all = {L_fore_S0, L_fore_S1, L_fore_S2, L_fore_Sstar};

results = struct();

for s = 1:numel(scenarios)
    name = scenarios{s};
    L0   = L_fore_all{s};
    
    fprintf('Solving MILP for scenario: %s\n', name);
    
    [x_opt, y_opt, P_opt] = schedule_peak_shaving(L0, A_kW, d_slots);
    
    % Realized total load using TRUE baseline
    total_load_real = L_true + sum( (A_kW .* x_opt), 1 )';
    P_real = max(total_load_real);
    
    % Forecast error (MAE)
    mae = mean(abs(L0 - L_true));
    
    results(s).name            = name;
    results(s).L0              = L0;
    results(s).x_opt           = x_opt;
    results(s).y_opt           = y_opt;
    results(s).P_opt_forecast  = P_opt;
    results(s).P_real          = P_real;
    results(s).MAE             = mae;
    results(s).total_load_real = total_load_real;
end

%% 6. Display numeric results
fprintf('\n=== Summary over scenarios ===\n');
fprintf('%-18s | %-8s | %-10s | %-10s\n', ...
    'Scenario', 'MAE', 'P_real (kW)', 'P_gap_to_Oracle');
fprintf(repmat('-', 1, 60)); fprintf('\n');

P_oracle = results(strcmp({results.name}, 'Oracle')).P_real;

for s = 1:numel(results)
    gap = results(s).P_real - P_oracle;
    fprintf('%-18s | %8.3f | %10.3f | %10.3f\n', ...
        results(s).name, results(s).MAE, results(s).P_real, gap);
end

%% 7. Plot true baseline and realized total loads
figure;
subplot(2,1,1);
plot(t_hours, L_true, 'LineWidth', 1.5); hold on;
title('True baseline load');
xlabel('Time [hours]');
ylabel('kW');
grid on;

subplot(2,1,2);
plot(t_hours, L_true, 'k--', 'LineWidth', 1.0); hold on;
for s = 1:numel(results)
    plot(t_hours, results(s).total_load_real, 'LineWidth', 1.2);
end
xlabel('Time [hours]');
ylabel('kW');
title('Realized total load (baseline + scheduled appliances)');
legend(['True baseline', scenarios], 'Location', 'NorthWest');
grid on;

%% 8. (Optional) Plot MAE vs realized peak for all scenarios
figure;
for s = 1:numel(results)
    plot(results(s).MAE, results(s).P_real, 'o', 'MarkerSize', 8); hold on;
    text(results(s).MAE * 1.01, results(s).P_real, results(s).name);
end
xlabel('Forecast MAE [kW]');
ylabel('Realized peak [kW]');
title('Effect of forecast quality on realized peak');
grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function: schedule_peak_shaving
% Build and solve the MILP:
%   - Variables: y_{i,t} (start), x_{i,t} (on/off), P (peak)
%   - Objective: minimize P
%   - Constraints:
%       * Each appliance starts once
%       * Each appliance runs for d_i slots
%       * x_{i,t} = sum_{tau=max(1, t-d_i+1)..t} y_{i,tau}
%       * L0(t) + sum_i A_i x_{i,t} <= P   for all t
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x_opt, y_opt, P_opt] = schedule_peak_shaving(L0, A_kW, d_slots)

    L0 = L0(:);              % T x 1
    [N, ~] = size(A_kW);     % N appliances
    T = numel(L0);

    A_kW = A_kW(:);          % ensure column

    % Indexing helpers
    % y_{i,t}: i=1..N, t=1..T  -> index 1..(N*T)
    % x_{i,t}: i=1..N, t=1..T  -> index (N*T+1)..(2*N*T)
    nY = N * T;
    nX = N * T;
    idxP = nY + nX + 1;      % Peak variable

    nVars = nY + nX + 1;

    % Objective: min P
    f = zeros(nVars, 1);
    f(idxP) = 1;

    % Integer constraints: y and x are binary
    intcon = 1:(nY + nX);

    % Bounds
    lb = zeros(nVars, 1);
    ub = ones(nVars, 1);
    ub(idxP) = Inf;          % peak can be larger than 1

    % === Equality constraints Aeq * z = beq ===
    % 1) Start once: sum_t y_{i,t} = 1
    % 2) Duration:  sum_t x_{i,t} = d_i
    % 3) Link: x_{i,t} - sum_{tau=max(1, t-d_i+1)..t} y_{i,tau} = 0

    % Count rows
    nEq_start   = N;
    nEq_dur     = N;
    nEq_link    = N * T;
    nEq         = nEq_start + nEq_dur + nEq_link;

    Aeq = sparse(nEq, nVars);
    beq = zeros(nEq, 1);

    row = 0;

    % 1) Start once
    for i = 1:N
        row = row + 1;
        for t = 1:T
            idxY = (i-1)*T + t;
            Aeq(row, idxY) = 1;
        end
        beq(row) = 1;
    end

    % 2) Duration constraints: sum_t x_{i,t} = d_i
    for i = 1:N
        row = row + 1;
        for t = 1:T
            idxX = nY + (i-1)*T + t;
            Aeq(row, idxX) = 1;
        end
        beq(row) = d_slots(i);
    end

    % 3) Link constraints
    for i = 1:N
        di = d_slots(i);
        for t = 1:T
            row = row + 1;
            % x_{i,t}
            idxX = nY + (i-1)*T + t;
            Aeq(row, idxX) = 1;

            % - sum over y_{i,tau}, tau = max(1, t-di+1)..t
            tau_min = max(1, t - di + 1);
            tau_max = t;
            for tau = tau_min:tau_max
                idxY = (i-1)*T + tau;
                Aeq(row, idxY) = Aeq(row, idxY) - 1;
            end
            beq(row) = 0;
        end
    end

    % === Inequality constraints Aineq * z <= bineq ===
    % Peak constraints: L0(t) + sum_i A_i x_{i,t} <= P
    % => sum_i A_i x_{i,t} - P <= -L0(t)

    Aineq = sparse(T, nVars);
    bineq = zeros(T, 1);

    for t = 1:T
        for i = 1:N
            idxX = nY + (i-1)*T + t;
            Aineq(t, idxX) = Aineq(t, idxX) + A_kW(i);
        end
        Aineq(t, idxP) = -1;
        bineq(t)       = -L0(t);
    end

    % Solve MILP
    options = optimoptions('intlinprog', ...
        'Display', 'off', ...
        'Heuristics', 'advanced', ...
        'CutGeneration', 'intermediate');

    [z_opt, fval, exitflag] = intlinprog(f, intcon, Aineq, bineq, Aeq, beq, lb, ub, options);

    if exitflag <= 0
        warning('intlinprog did not converge to an optimal solution. Exitflag = %d', exitflag);
    end

    % Extract solutions
    z_opt(isnan(z_opt)) = 0;  % safety
    y_vec = z_opt(1:nY);
    x_vec = z_opt(nY+1:nY+nX);
    P_opt = z_opt(idxP);

    % Reshape x and y into N x T matrices
    y_opt = reshape(y_vec, [T, N])';  % N x T
    x_opt = reshape(x_vec, [T, N])';  % N x T

    % Round for safety (numerical tolerances)
    y_opt = round(y_opt);
    x_opt = round(x_opt);
end