%% --- NON-UNIFORM FORECAST GRID (AS BEFORE) ------------------

T = 96;
t = linspace(0,24,T);
t_grid = t*3600;
dt = diff(t_grid);

% ground truth + smart forecast
morning_peak = 1.5 * exp(-(t - 8).^2 / (2*1.2^2));
evening_peak = 2.5 * exp(-(t - 19).^2 / (2*1.5^2));
noise = 0.15*randn(size(t));
L_true = max(morning_peak + evening_peak + 0.5 + noise, 0.2);

smart_morn = 1.3*exp(-(t-8).^2/(2*1.3^2));
smart_even = 2.1*exp(-(t-19).^2/(2*1.8^2));
L_fore_smart = max(smart_morn + smart_even + 0.55 + 0.07*randn(size(t)),0.2);


%% ============================================================
%  CREATE MULTIPLE APPLIANCE TYPES (N1, N2, ...)
%  ============================================================

Appliances = {};

% --- How many of each type?
N_square = 2;
N_ramp   = 1;
N_piece  = 2;
N_prob   = 1;
N_dyn    = 1;
N_conv   = 1;

% -------------------------------------------------------------
% TYPE 1: Square Loads
for n = 1:N_square
    params.power = 1.5;
    params.duration = 2*3600;
    Appliances{end+1} = createAppliance("square",params);
end

% -------------------------------------------------------------
% TYPE 2: Ramp Loads
for n = 1:N_ramp
    params.A_min     = 0.5;
    params.A_nom     = 2.5;
    params.ramp_up   = 0.5*3600;
    params.ramp_down = 0.5*3600;
    params.duration  = 3*3600;
    Appliances{end+1} = createAppliance("ramp",params);
end

% -------------------------------------------------------------
% TYPE 3: Piecewise cycles
for n = 1:N_piece
    params.stages     = [0.6, 1.5, 2.8];
    params.stage_durs = [0.5, 1.0, 0.5]*3600;
    Appliances{end+1} = createAppliance("piecewise",params);
end

% -------------------------------------------------------------
% TYPE 4: Probabilistic-duration loads
for n = 1:N_prob
    params.T_opts = [1*3600, 2*3600, 3*3600];
    params.prob   = [0.4, 0.4, 0.2];
    params.power  = 1.8;
    Appliances{end+1} = createAppliance("probabilistic",params);
end

% -------------------------------------------------------------
% TYPE 5: First-order dynamic loads
for n = 1:N_dyn
    params.Ci = 1.5;
    params.alpha = 0.0005;
    params.u_nom = 3.0;
    params.duration = 3*3600;
    Appliances{end+1} = createAppliance("first_order",params);
end

% -------------------------------------------------------------
% TYPE 6: Convolution-based devices
for n = 1:N_conv
    params.duration = 3*3600;
    tau = linspace(0,params.duration,100);
    params.kernel = exp(-tau/(0.5*3600)) * 2.0;
    Appliances{end+1} = createAppliance("convolution",params);
end

M = length(Appliances);
fprintf("Total appliances: %d\n", M);


%% ============================================================
%  BUILD INTRINSIC PROFILES p_i(τ) → NON-UNIFORM GRID p_{i,τ,k}
%  ============================================================

N = length(t_grid);
p = cell(M,1);

for i = 1:M

    Ai = Appliances{i};

    % internal profile sampling
    dt_internal = 60;
    tau_int = 0:dt_internal:Ai.duration;

    % intrinsic profile p_i(τ)
    switch Ai.type
        case "square"
            p_int = Ai.power * ones(size(tau_int));

        case "ramp"
            p_int = zeros(size(tau_int));
            for jj = 1:length(tau_int)
                tau = tau_int(jj);
                if tau < Ai.ramp_up
                    p_int(jj) = Ai.A_min + (Ai.A_nom - Ai.A_min)*(tau/Ai.ramp_up);
                elseif tau < (Ai.duration - Ai.ramp_down)
                    p_int(jj) = Ai.A_nom;
                else
                    rem = tau - (Ai.duration - Ai.ramp_down);
                    p_int(jj) = Ai.A_nom - (Ai.A_nom - Ai.A_min)*(rem/Ai.ramp_down);
                end
            end

        case "piecewise"
            p_int = zeros(size(tau_int));
            edges = [0, cumsum(Ai.stage_durs)];
            for s = 1:numel(Ai.stages)
                idx = (tau_int >= edges(s) & tau_int < edges(s+1));
                p_int(idx) = Ai.stages(s);
            end

        case "probabilistic"
            Texp = sum(Ai.T_opts .* Ai.prob);
            tau_int = linspace(0,Texp,length(tau_int));
            p_int = Ai.power * ones(size(tau_int));

        case "first_order"
            P = zeros(size(tau_int));
            for jj = 2:length(tau_int)
                dtt = tau_int(jj) - tau_int(jj-1);
                P(jj) = exp(-Ai.alpha*dtt)*P(jj-1) + ...
                        (1-exp(-Ai.alpha*dtt))*(Ai.u_nom/Ai.alpha);
            end
            p_int = P;

        case "convolution"
            tt_kern = linspace(0,Ai.duration,length(Ai.kernel));
            p_int = interp1(tt_kern,Ai.kernel,tau_int,'linear',0);
    end

    % project p_i(τ) onto (τ,k) pairs
    p_mat = zeros(N,N);
    for tau_idx = 1:(N-1)
        elapsed = 0;
        for k_idx = tau_idx:(N-1)
            elapsed = elapsed + dt(k_idx);
            if elapsed > max(tau_int)
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
%  ============================================================

num_y = M*(N-1);
idxP  = num_y + 1;

f = zeros(num_y+1,1);
f(idxP) = 1; % minimize P

A = []; b = [];
Aeq = []; beq = [];

% single start
for i=1:M
    row = zeros(1,num_y+1);
    base = (i-1)*(N-1);
    row(base + (1:(N-1))) = 1;
    Aeq = [Aeq; row];
    beq = [beq; 1];
end

% peak constraints
for k = 1:(N-1)
    row = zeros(1,num_y+1);
    rhs = L_fore_smart(k) + 0.1; % uncertainty
    cnt = 1;

    for i=1:M
        pm = p{i};
        for tau=1:(N-1)
            row(cnt) = pm(tau,k);
            cnt = cnt + 1;
        end
    end

    row(idxP) = -1;
    A = [A; row];
    b = [b; rhs];
end

lb = zeros(num_y+1,1);
ub = ones(num_y+1,1);
ub(end) = Inf;

intcon = 1:num_y;

opts = optimoptions('intlinprog','Display','iter','MaxTime',20);

sol = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,opts);

y_sol = sol(1:num_y);
P_star = sol(end);

fprintf("Optimal predicted peak = %.3f kW\n", P_star);

%% ------------------------------------------------------------
%  Random Scheduling for Comparison
% ------------------------------------------------------------

y_rand = randomSchedule(Appliances, t_grid);

% Compute load for random schedule
load_random = zeros(N,1);

offset = 0;
for i=1:M
    pm = p{i};
    for tau = 1:(N-1)
        if y_rand(offset + tau) > 0.5
            load_random = load_random + pm(tau,:)';
        end
    end
    offset = offset + (N-1);
end

% Total random load
L_rand_total = L_fore_smart(:) + load_random;


%% ============================================================
%  PLOT RESULTS
%  ============================================================

figure; hold on; grid on;
plot(t, L_true, 'k', 'LineWidth', 2);
plot(t, L_fore_smart, '--k', 'LineWidth', 1.5);
plot(t, L_fore_smart + load_add', 'r', 'LineWidth', 2);   % MILP
plot(t, L_rand_total, 'b', 'LineWidth', 1.7);             % random
xlabel("Time [h]");
ylabel("Load [kW]");
legend("Ground Truth", "Smart Forecast", ...
       "Optimal MILP Schedule", "Random Schedule");
title("Optimal vs Random Scheduling Comparison");