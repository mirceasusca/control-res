function y_rand = randomSchedule(Appliances, t_grid)
% randomSchedule: Assigns each appliance a random feasible start time
%
% INPUT:
%   Appliances : struct array (each with duration field)
%   t_grid     : time grid (seconds)
%
% OUTPUT:
%   y_rand : binary vector of size M*(N-1), same shape as MILP y variables

M = length(Appliances);
N = length(t_grid);
num_y = M*(N-1);
y_rand = zeros(num_y,1);

dt = diff(t_grid);

offset = 0;
for i = 1:M
    Ai = Appliances{i};

    % total operation duration
    dur = Ai.duration;

    % compute which start indices are feasible
    feasible_starts = [];
    for k = 1:(N-1)
        elapsed = 0;
        for kk = k:(N-1)
            elapsed = elapsed + dt(kk);
            if elapsed >= dur
                feasible_starts = [feasible_starts, k];
                break;
            end
        end
    end

    % pick a random feasible start
    if isempty(feasible_starts)
        error("No feasible start times for appliance %d.", i);
    end
    start_idx = feasible_starts(randi(length(feasible_starts)));

    % encode in y_rand
    base = offset;
    y_rand(base + start_idx) = 1;

    offset = offset + (N-1);
end