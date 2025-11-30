function A = createAppliance(type, params)
% createAppliance  Constructs a unified appliance description
%
% TYPE is a string: 
%   "square", "ramp", "piecewise", "probabilistic", "first_order", "convolution"
%
% PARAMS is a struct with fields depending on TYPE.

A.type = type;

switch type

    case "square"
        A.power    = params.power;
        A.duration = params.duration;

    case "ramp"
        A.A_min     = params.A_min;
        A.A_nom     = params.A_nom;
        A.ramp_up   = params.ramp_up;
        A.ramp_down = params.ramp_down;
        A.duration  = params.duration;

    case "piecewise"
        A.stages     = params.stages;      % vector of powers
        A.stage_durs = params.stage_durs;  % vector of durations
        A.duration   = sum(params.stage_durs);

    case "probabilistic"
        A.T_opts   = params.T_opts;   % [T1, T2, ...]
        A.prob     = params.prob;     % probabilities
        A.power    = params.power;    % constant-power example
        A.duration = sum(params.T_opts .* params.prob); % expected

    case "first_order"
        A.Ci       = params.Ci;
        A.alpha    = params.alpha;
        A.u_nom    = params.u_nom;
        A.duration = params.duration;

    case "convolution"
        A.kernel   = params.kernel;       % impulse response
        A.duration = params.duration;

    otherwise
        error("Unknown appliance type: %s", type);
end