function y_rand = randomSchedule(Appliances, t_grid)
% randomSchedule: Assigns each appliance a random feasible start time
%   respecting only duration (not precedence). Precedence will be adjusted
%   later if needed.
%
% INPUT:
%   Appliances : cell array of appliance structs (each with .duration)
%   t_grid     : time grid (seconds), length N
%
% OUTPUT:
%   y_rand : binary vector of length M*(N-1), same layout as MILP y.

M = numel(Appliances);
N = numel(t_grid);
Nint = N-1;
num_y = M * Nint;

y_rand = zeros(num_y,1);
dt = diff(t_grid);

offset = 0;
for i = 1:M
    Ai = Appliances{i};
    dur = Ai.duration;

    feasible_starts = [];
    for k = 1:Nint
        elapsed = 0;
        for kk = k:Nint
            elapsed = elapsed + dt(kk);
            if elapsed >= dur
                feasible_starts(end+1) = k; %#ok<AGROW>
                break;
            end
        end
    end

    if isempty(feasible_starts)
        error("No feasible start times for appliance %d.", i);
    end

    start_idx = feasible_starts(randi(numel(feasible_starts)));
    base = offset;
    y_rand(base + start_idx) = 1;

    offset = offset + Nint;
end