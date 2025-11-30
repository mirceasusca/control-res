%% ============================================================
%  RESTARTED SCRIPT: FORECAST + MULTI-MODEL APPLIANCES + MILP
%  ============================================================
% clear; 
clc; close all;
load('forecasting_pred.mat')

% --------------------------------------------------------------
% ASSUMED AVAILABLE VARIABLES:
%   L_fore_naive
%   L_fore_smart
%   L_true
%   t               % hours, length T
%
% If not available, uncomment to generate a synthetic test:
%
%{
T = 96;
t = linspace(0,24,T);
morning = 1.5*exp(-(t-8).^2/(2*1^2));
evening = 2.7*exp(-(t-19).^2/(2*1.3^2));
L_true = max(morning + evening + 0.5 + 0.1*randn(size(t)),0.2);
L_fore_naive = mean(L_true)*ones(size(t));
L_fore_smart = L_true + 0.15*randn(size(t));  % better forecast
%}
% --------------------------------------------------------------

t_grid = t * 3600;
dt = diff(t_grid);
N = length(t_grid);
Nint = N-1;

%% ============================================================
%  CREATE APPLIANCES (REASONABLE POWER LEVELS)
% =============================================================

Appliances = {};
idx = 0;

% peak forecast is roughly 3 → appliances must be smaller (0.3–1.2kW each)

% Square loads (small heaters)
for n = 1:2
    params.power    = 0.8;
    params.duration = 1.0 * 3600;
    Appliances{end+1} = createAppliance("square",params);
end

% Ramp-type loads (small HVAC)
for n = 1:1
    params.A_min = 0.2;
    params.A_nom = 1.0;
    params.ramp_up = 0.2*3600;
    params.ramp_down = 0.2*3600;
    params.duration = 1.5 * 3600;
    Appliances{end+1} = createAppliance("ramp",params);
end

% Piecewise-cycle loads (dishwasher / washing machine)
for n = 1:3
    params.stages = [0.3, 0.9, 1.2];
    params.stage_durs = [0.3, 0.7, 0.4]*3600;
    Appliances{end+1} = createAppliance("piecewise",params);
end

% First-order dynamic load (freezer/compressor)
for n = 1:1
    params.Ci       = 1.0;          % not used in this simplified model
params.alpha    = 1/600;        % ~10 minutes time constant
params.u_nom    = 0.8;          % kW steady-state draw
params.duration = 2*3600;       % 2 hours
Appliances{end+1} = createAppliance("first_order", params);
end

M = length(Appliances);
fprintf("Created %d appliances.\n", M);

%% ============================================================
%  DEFINE OPTIONAL PRECEDENCE / SUCCESSION CONSTRAINTS
% =============================================================

% Example: appliance 3 must start after appliance 1 finishes,
% but no later than +1.5 hours.
Succession = {};
Succession{1} = struct("i",1,"j",3,"gmin", 1.0*3600, "gmax", 1.5*3600);
% You can add more constraints:
% Succession{end+1} = struct("i",2,"j",4,"gmin",3600,"gmax",7200);

%% ============================================================
%  SAMPLE PROFILES p_{i,τ,k} FOR MILP
% =============================================================

fprintf("Sampling appliance profiles...\n");

p = cell(M,1);

for i = 1:M
    Ai = Appliances{i};

    dt_internal = 60;        % 1-min internal resolution
    tau_int = 0:dt_internal:Ai.duration;

    % --------------- intrinsic profile p_i(τ) -----------------
    switch Ai.type
        case "square"
            p_int = Ai.power * ones(size(tau_int));

        case "ramp"
            p_int = zeros(size(tau_int));
            for kk = 1:length(tau_int)
                tau = tau_int(kk);
                if tau < Ai.ramp_up
                    p_int(kk) = Ai.A_min + (Ai.A_nom-Ai.A_min)*(tau/Ai.ramp_up);
                elseif tau < (Ai.duration - Ai.ramp_down)
                    p_int(kk) = Ai.A_nom;
                else
                    rem = tau - (Ai.duration-Ai.ramp_down);
                    p_int(kk) = Ai.A_nom - (Ai.A_nom-Ai.A_min)*(rem/Ai.ramp_down);
                end
            end

        case "piecewise"
            p_int = zeros(size(tau_int));
            edges = [0, cumsum(Ai.stage_durs)];
            for s = 1:length(Ai.stages)
                idx = tau_int >= edges(s) & tau_int < edges(s+1);
                p_int(idx) = Ai.stages(s);
            end

        case "first_order"
            % dP/dt = -alpha * P + alpha * u_nom  →  P_ss = u_nom
            P = zeros(size(tau_int));
            for kk = 2:length(tau_int)
                dtt = tau_int(kk) - tau_int(kk-1);
                P(kk) = exp(-Ai.alpha * dtt) * P(kk-1) + ...
                        (1 - exp(-Ai.alpha * dtt)) * Ai.u_nom;
            end
            p_int = P;
    end

    % --------------- project onto non-uniform grid --------------
    p_mat = zeros(N,N);

    for tau_idx = 1:Nint
        elapsed = 0;
        for k_idx = tau_idx:Nint
            elapsed = elapsed + dt(k_idx);
            if elapsed > tau_int(end)
                val = 0;
            else
                val = interp1(tau_int,p_int,elapsed,'linear',0);
            end
            p_mat(tau_idx,k_idx) = val;
        end
    end

    p{i} = p_mat;
end

%% ============================================================
%  MILP FORMULATION
% =============================================================

num_y = M*Nint;
idxP = num_y + 1;

f = zeros(num_y+1,1);
f(idxP) = 1;

A = []; b = [];
Aeq = []; beq = [];

% ---------- Start exactly once ----------
for i=1:M
    row = zeros(1,num_y+1);
    base = (i-1)*Nint;
    row(base + (1:Nint)) = 1;
    Aeq = [Aeq; row];
    beq = [beq; 1];
end

% ---------- Succession constraints ----------
for s = 1:length(Succession)
    sc = Succession{s};
    i = sc.i; j = sc.j;

    row = zeros(1,num_y+1);
    S_i = zeros(1,num_y+1);
    S_j = zeros(1,num_y+1);

    base_i = (i-1)*Nint;
    base_j = (j-1)*Nint;

    % compute S_i and S_j
    S_i(base_i + (1:Nint)) = t_grid(1:Nint);
    S_j(base_j + (1:Nint)) = t_grid(1:Nint);

    % S_j - S_i >= gmin  →  (S_j - S_i) >= gmin
    A = [A; -(S_j - S_i)];
    b = [b; -sc.gmin];

    % S_j - S_i <= gmax
    A = [A; (S_j - S_i)];
    b = [b; sc.gmax];
end

% ---------- Peak constraints ----------
epsilon = 0.1;  % small uncertainty margin

for k = 1:Nint
    row = zeros(1,num_y+1);
    rhs = L_fore_smart(k) + epsilon;

    offset = 0;
    for i = 1:M
        pm = p{i};
        for tau = 1:Nint
            row(offset+tau) = pm(tau,k);
        end
        offset = offset + Nint;
    end

    row(idxP) = -1;
    A = [A; row];
    b = [b; rhs];
end

% ---------- Solve MILP ----------
lb = zeros(num_y+1,1);
ub = ones(num_y+1,1);
ub(idxP) = Inf;

intcon = 1:num_y;
opts = optimoptions("intlinprog","Display","off","MaxTime",20);

fprintf("Solving MILP...\n");
sol = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,opts);
y_milp = sol(1:num_y);
P_milp = sol(end);

%% ============================================================
%  RANDOM FEASIBLE SCHEDULE (RESPECTING SUCCESSION)
% ============================================================

y_rand = zeros(num_y,1);

% first schedule all appliances ignoring succession
for i = 1:M
    Ai = Appliances{i};
    feasible = [];

    for k=1:Nint
        elapsed = 0;
        for kk=k:Nint
            elapsed = elapsed + dt(kk);
            if elapsed >= Ai.duration
                feasible(end+1) = k;
                break;
            end
        end
    end

    start_k = feasible(randi(length(feasible)));
    base=(i-1)*Nint;
    y_rand(base + start_k) = 1;
end

% now apply succession constraints by shifting start times forward
for s = 1:length(Succession)
    sc = Succession{s};
    i = sc.i; j = sc.j;

    base_i = (i-1)*Nint;
    base_j = (j-1)*Nint;

    ki = find(y_rand(base_i + (1:Nint))==1);
    kj = find(y_rand(base_j + (1:Nint))==1);

    % compute actual times
    Si = t_grid(ki);
    Sj = t_grid(kj);

    if Sj - Si < sc.gmin  % too early
        target_t = Si + sc.gmin;
        [~,new_kj] = min(abs(t_grid - target_t));
        new_kj = min(new_kj, Nint);

        y_rand(base_j + (1:Nint)) = 0;
        y_rand(base_j + new_kj) = 1;
    end

    if Sj - Si > sc.gmax  % too late
        target_t = Si + sc.gmax;
        [~,new_kj] = min(abs(t_grid - target_t));
        new_kj = max(1,new_kj);

        y_rand(base_j + (1:Nint)) = 0;
        y_rand(base_j + new_kj) = 1;
    end
end

%% ============================================================
%  COMPUTE TOTAL LOADS FOR BOTH SCHEDULES
% ============================================================

computeLoad = @(y) ...
    reshape(cell2mat(arrayfun(@(i) computeApplianceLoad(p{i}, y((i-1)*Nint+1 : i*Nint)), ...
           1:M, 'UniformOutput', false)), N, []);

function L = computeApplianceLoad(p_i, y_i)
    L = zeros(size(p_i,1),1);
    starts = find(y_i>0.5);
    for s = starts
        L = L + p_i(s,:)';
    end
end

load_milp = sum(computeLoad(y_milp),2);
load_rand = sum(computeLoad(y_rand),2);

%% ============================================================
%  PLOTTING
% ============================================================

figure; hold on; grid on;
plot(t, L_true, "k", "LineWidth", 2);
plot(t, L_fore_smart, "--k", "LineWidth", 1.2);
plot(t, L_fore_smart + load_milp', "r", "LineWidth", 2);
plot(t, L_fore_smart + load_rand', "b", "LineWidth", 1.7);
xlabel("Time [h]");
ylabel("Load [kW]");
legend("True Load","Smart Forecast", ...
       "MILP optimal schedule", "Random feasible schedule");
title("Optimal Appliance Scheduling vs Random Feasible Scheduling");