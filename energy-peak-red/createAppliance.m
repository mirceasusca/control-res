function A = createAppliance(type, params)
% createAppliance  Constructs a unified appliance description
%
% TYPE is a string: 
%   "square", "ramp", "piecewise", "probabilistic", "first_order", "convolution"
%
% PARAMS is a struct with fields depending on TYPE.

A.type = string(type);

switch A.type

    case "square"
        A.power    = params.power;       % kW
        A.duration = params.duration;    % s

    case "ramp"
        A.A_min     = params.A_min;      % kW
        A.A_nom     = params.A_nom;      % kW
        A.ramp_up   = params.ramp_up;    % s
        A.ramp_down = params.ramp_down;  % s
        A.duration  = params.duration;   % s

    case "piecewise"
        A.stages     = params.stages;      % [kW]
        A.stage_durs = params.stage_durs;  % [s]
        A.duration   = sum(params.stage_durs);

    case "probabilistic"
        A.T_opts   = params.T_opts;   % [s]
        A.prob     = params.prob;     % probabilities
        A.power    = params.power;    % kW
        % use expected duration in the scheduler
        A.duration = sum(params.T_opts .* params.prob);

    case "first_order"
        % dP/dt = -alpha P + alpha u_nom  → steady state P_ss = u_nom
        A.Ci       = params.Ci;        % (kept for future extension)
        A.alpha    = params.alpha;     % 1/s
        A.u_nom    = params.u_nom;     % kW
        A.duration = params.duration;  % s

    case "convolution"
        A.kernel   = params.kernel;    % kW, impulse/step response samples
        A.duration = params.duration;  % s

    otherwise
        error("Unknown appliance type: %s", type);
end