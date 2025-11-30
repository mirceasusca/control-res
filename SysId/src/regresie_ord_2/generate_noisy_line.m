function [x, y] = generate_noisy_line(num_points, slope, intercept, noise_level)
    % Generate x coordinates
    x = linspace(0, 10, num_points);
    
    % Generate y coordinates based on the line equation y = mx + b
    y_true = slope * x + intercept;
    
    % Add random noise to the y coordinates
    noise = noise_level * randn(size(y_true));
    y = y_true + noise;
end