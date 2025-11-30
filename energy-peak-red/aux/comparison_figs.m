function demo_forecast_milp_full
%% DEMO: Forecasting + MILP Peak Shaving + Random Benchmarks
clear; clc; close all;
rng(0);

%% ------------------------------------------------------------------------
% 1) ONE-DAY BASELINE: Ground truth + naive + smart forecast
% -------------------------------------------------------------------------
T = 96;                          % 96 x 15min = 24h
t = linspace(0,24,T);            % hours

% Ground truth (synthetic but realistic)
morning_peak     = 1.5 * exp(-(t - 8).^2  / (2*1.2^2));
evening_peak     = 2.5 * exp(-(t - 19).^2 / (2*1.5^2));
night_background = 0.5 + 0.1*sin(0.5*t);
noise            = 0.15 * randn(size(t));

L_true = morning_peak + evening_peak + night_background + noise;
L_true = max(L_true,0.2);        % no negatives

% Naive forecast: flat + big noise
avg_load      = mean(L_true);
L_fore_naive  = avg_load + 0.4*randn(size(t));
L_fore_naive  = max(L_fore_naive,0.2);

% Smart forecast: aligned peaks + smoother background
smart_morning     = 1.3 * exp(-(t - 8).^2  / (2*1.3^2));
smart_evening     = 2.0 * exp(-(t - 19).^2 / (2*1.8^2));
smart_background  = 0.55 + 0.08*sin(0.4*t);
L_fore_smart      = smart_morning + smart_evening + smart_background ...
                    + 0.07*randn(size(t));
L_fore_smart      = max(L_fore_smart,0.2);

MAE_naive = mean(abs(L_fore_naive - L_true));
MAE_smart = mean(abs(L_fore_smart - L_true));

fprintf('Forecast example (one day):\n');
fprintf('  MAE naive  = %.3f\n', MAE_naive);
fprintf('  MAE smart  = %.3f\n', MAE_smart);

% ---- Figure 1: forecasting example ----
figure(1);
plot(t, L_true, 'LineWidth', 2); hold on;
plot(t, L_fore_naive, '--', 'LineWidth', 1.5);
plot(t, L_fore_smart, '-.', 'LineWidth', 1.5);
grid on;
xlabel('Time [hours]');
ylabel('Load [kW]');
title('Ground Truth vs. Naive vs. Smart Forecast (Example Day)');
legend('Ground truth', ...
       sprintf('Naive (MAE=%.2f)', MAE_naive), ...
       sprintf('Smart (MAE=%.2f)', MAE_smart), ...
       'Location', 'NorthWest');
% print -dpdf fig1_forecast_example.pdf

%% ------------------------------------------------------------------------
% 2) MILP SCHEDULING on top of the SMART forecast baseline
% -------------------------------------------------------------------------
L_baseline = L_fore_smart(:);  % use smart forecast as baseline for scheduling
T = numel(L_baseline);

% Configurable controllable consumers
numConsumers = 5;          % change as you wish
dt           = t(2)-t(1);  % hours per slot

A_kW  = 1.5 + (3.0 - 1.5)*rand(numConsumers,1);   % 1.5–3.0 kW
dur_h = 1   + 2*rand(numConsumers,1);             % 1–3 hours

d_slots = ceil(dur_h / dt);

fprintf('\nConsumers (controllable loads):\n');
for i = 1:numConsumers
    fprintf('  #%d: A = %.2f kW, duration = %.2f h (%d slots)\n', ...
        i, A_kW(i), dur_h(i), d_slots(i));
end

% --- Reference "unscheduled" case: all start at baseline peak ---
[~, idx_peak_base] = max(L_baseline);
x_unsched = zeros(numConsumers, T);
for i = 1:numConsumers
    d = d_slots(i);
    start_idx = idx_peak_base;
    if start_idx + d - 1 > T
        start_idx = T - d + 1;
    end
    x_unsched(i, start_idx:start_idx+d-1) = 1;
end
total_load_unsched = L_baseline' + sum(A_kW .* x_unsched,1);
P_unsched          = max(total_load_unsched);

% --- MILP schedule ---
[x_milp, y_milp, P_milp] = schedule_peak_shaving(L_baseline, A_kW, d_slots);
total_load_milp         = L_baseline' + sum(A_kW .* x_milp,1);
P_milp_real             = max(total_load_milp);

P_base = max(L_baseline);

fprintf('\nPeak comparison (single day):\n');
fprintf('  Baseline-only peak        : %.3f kW\n', P_base);
fprintf('  Unscheduled peak          : %.3f kW\n', P_unsched);
fprintf('  MILP scheduled peak       : %.3f kW\n', P_milp_real);

% ---- Figure 2: load flattening by MILP ----
figure(2);
subplot(2,1,1);
plot(t, L_baseline, 'LineWidth', 2); hold on;
plot(t, total_load_unsched, '--', 'LineWidth', 1.5);
plot(t, total_load_milp, '-.', 'LineWidth', 1.8);
grid on;
xlabel('Time [hours]');
ylabel('Total load [kW]');
title('Baseline vs. Unscheduled vs. MILP-Scheduled Total Load');
legend(sprintf('Baseline (peak=%.2f)',P_base), ...
       sprintf('Unscheduled (peak=%.2f)',P_unsched), ...
       sprintf('MILP scheduled (peak=%.2f)',P_milp_real), ...
       'Location','NorthWest');

subplot(2,1,2);
plot(t, sum(A_kW .* x_unsched,1), '--', 'LineWidth', 1.5); hold on;
plot(t, sum(A_kW .* x_milp,1), '-.', 'LineWidth', 1.8);
grid on;
xlabel('Time [hours]');
ylabel('Controllable load [kW]');
title('Controllable Consumption: Unscheduled vs. MILP-Scheduled');
legend('Unscheduled','MILP scheduled','Location','NorthWest');
% print -dpdf fig2_milp_flattening.pdf

%% ------------------------------------------------------------------------
% 3) RANDOM SCHEDULING vs MILP (boxplot of peaks)
% -------------------------------------------------------------------------
numRandomTests = 50;
P_rand_all   = zeros(numRandomTests,1);
P_rand_pairs = zeros(numRandomTests,1);

for trial = 1:numRandomTests
    % --- Random-all: each appliance random start ---
    x_rand = zeros(numConsumers,T);
    for i = 1:numConsumers
        d = d_slots(i);
        start_idx = randi([1, T-d+1]);
        x_rand(i, start_idx:start_idx+d-1) = 1;
    end
    total_rand = L_baseline' + sum(A_kW .* x_rand,1);
    P_rand_all(trial) = max(total_rand);

    % --- Random-pairs: appliances shuffled then random by pairs ---
    x_pair = zeros(numConsumers,T);
    perm   = randperm(numConsumers);
    for idx = 1:2:numConsumers
        i = perm(idx);
        d = d_slots(i);
        s = randi([1, T-d+1]);
        x_pair(i, s:s+d-1) = 1;
        if idx+1 <= numConsumers
            j = perm(idx+1);
            d2 = d_slots(j);
            s2 = randi([1, T-d2+1]);
            x_pair(j, s2:s2+d2-1) = 1;
        end
    end
    total_pair = L_baseline' + sum(A_kW .* x_pair,1);
    P_rand_pairs(trial) = max(total_pair);
end

fprintf('\nRandom scheduling statistics (peaks):\n');
fprintf('  Random-all   mean = %.3f, min = %.3f\n', ...
    mean(P_rand_all), min(P_rand_all));
fprintf('  Random-pairs mean = %.3f, min = %.3f\n', ...
    mean(P_rand_pairs), min(P_rand_pairs));

% ---- Figure 3: boxplot of peaks ----
figure(3);
boxplot([P_rand_all P_rand_pairs], ...
        'Labels',{'Random all','Random pairs'});
hold on; grid on;
yline(P_milp_real,'r-','LineWidth',2);
yline(P_unsched,'k--','LineWidth',1.5);
yline(P_base,'b:','LineWidth',1.5);
ylabel('Peak load [kW]');
title('Peak Comparison: Random Schedules vs. MILP');
legend(sprintf('MILP peak = %.2f',P_milp_real), ...
       sprintf('Unscheduled peak = %.2f',P_unsched), ...
       sprintf('Baseline peak = %.2f',P_base), ...
       'Location','NorthEast');
hold off;
% print -dpdf fig3_random_vs_milp.pdf

end

% ======================================================================
% Local MILP solver (same as before, just packaged)
% ======================================================================
function [x_opt, y_opt, P_opt] = schedule_peak_shaving(L0, A_kW, d_slots)

    L0 = L0(:);
    [N,~] = size(A_kW);
    T = numel(L0);
    A_kW = A_kW(:);

    nY = N*T;
    nX = N*T;
    idxP = nY + nX + 1;
    nVar = idxP;

    f = zeros(nVar,1);    % objective: min P
    f(idxP) = 1;

    intcon = 1:(nY+nX);
    lb = zeros(nVar,1);
    ub = ones(nVar,1);
    ub(idxP) = Inf;

    % === Equality constraints ===
    nEq_start = N;
    nEq_dur   = N;
    nEq_link  = N*T;
    nEq       = nEq_start + nEq_dur + nEq_link;
    Aeq = sparse(nEq, nVar);
    beq = zeros(nEq,1);
    row = 0;

    % start once
    for i = 1:N
        row = row + 1;
        for k = 1:T
            idxY = (i-1)*T + k;
            Aeq(row, idxY) = 1;
        end
        beq(row) = 1;
    end

    % duration
    for i = 1:N
        row = row + 1;
        for k = 1:T
            idxX = nY + (i-1)*T + k;
            Aeq(row, idxX) = 1;
        end
        beq(row) = d_slots(i);
    end

    % link x,y
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

    % === Inequality constraints ===
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

    options = optimoptions('intlinprog', ...
        'Display','off', ...
        'Heuristics','advanced', ...
        'CutGeneration','intermediate');

    [z_opt,~,exitflag] = intlinprog(f,intcon,Aineq,bineq,Aeq,beq,lb,ub,options);

    if exitflag <= 0
        warning('intlinprog did not converge, exitflag=%d',exitflag);
    end

    z_opt(isnan(z_opt)) = 0;
    y_vec = z_opt(1:nY);
    x_vec = z_opt(nY+1:nY+nX);
    P_opt = z_opt(idxP);

    y_opt = reshape(y_vec,[T,N])';
    x_opt = reshape(x_vec,[T,N])';
    y_opt = round(y_opt);
    x_opt = round(x_opt);
end