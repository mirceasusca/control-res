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