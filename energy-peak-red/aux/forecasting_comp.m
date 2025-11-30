% %% One-Day Consumption Forecasting Example
% clear; clc; close all;
% 
% %% Time axis (96 intervals = 15 min resolution)
% T = 96;
% t = linspace(0,24,T);    % hours
% 
% %% ------------------------------------------------------------
% % 1. Ground truth load (created to look realistic)
% % ------------------------------------------------------------
% morning_peak = 1.5 * exp(-(t - 8).^2 / (2*1.2^2));
% evening_peak = 2.5 * exp(-(t - 19).^2 / (2*1.5^2));
% night_background = 0.5 + 0.1*sin(0.5*t);
% noise = 0.15 * randn(size(t));
% 
% L_true = morning_peak + evening_peak + night_background + noise;
% L_true = max(L_true,0.2);  % ensure no negative values
% 
% %% ------------------------------------------------------------
% % 2. Very naive forecast (baseline average + random noise)
% % ------------------------------------------------------------
% avg_load = mean(L_true);           % constant prediction
% naive_noise = 0.4 * randn(size(t)); % high variance noise
% L_fore_naive = avg_load + naive_noise;
% L_fore_naive = max(L_fore_naive,0.2);
% 
% %% ------------------------------------------------------------
% % 3. A slightly smarter forecast
% %    Using: daily pattern (~ sinusoids representing peaks)
% %    but still not perfect
% % ------------------------------------------------------------
% L_fore_smart = ...
%     0.6 * sin(pi*(t-6)/12).^2 + ...    % morning-ish pattern
%     1.1 * sin(pi*(t-17)/8 ).^2 + ...   % evening-ish pattern
%     0.5 + 0.05*t;                      % small upward drift
% 
% % Add small noise
% L_fore_smart = L_fore_smart + 0.1*randn(size(t));
% L_fore_smart = max(L_fore_smart, 0.2);
% 
% %% ------------------------------------------------------------
% % 4. Compute forecast errors
% % ------------------------------------------------------------
% MAE_naive = mean(abs(L_fore_naive - L_true));
% MAE_smart = mean(abs(L_fore_smart - L_true));
% 
% fprintf('Naive forecast MAE  = %.3f\n', MAE_naive);
% fprintf('Smart forecast MAE  = %.3f\n', MAE_smart);
% 
% %% ------------------------------------------------------------
% % 5. Plot comparison
% % ------------------------------------------------------------
% figure;
% plot(t, L_true, 'k-', 'LineWidth', 2); hold on;
% plot(t, L_fore_naive, 'r--', 'LineWidth', 1.5);
% plot(t, L_fore_smart, 'b-.', 'LineWidth', 1.5);
% 
% grid on;
% xlabel('Time [hours]');
% ylabel('Load [kW]');
% title('One-Day Baseline Consumption: Ground Truth vs. Forecasts');
% legend('Ground truth', ...
%        sprintf('Naive forecast (MAE=%.2f)', MAE_naive), ...
%        sprintf('Smart forecast (MAE=%.2f)', MAE_smart), ...
%        'Location', 'NorthWest');

%% One-Day Consumption Forecasting Example (Improved Smart Forecast)
clear; clc; close all;

%% Time axis (96 intervals = 15 min resolution)
T = 96;
t = linspace(0,24,T);    % hours

%% ------------------------------------------------------------
% 1. Ground truth load (morning + evening peaks)
% ------------------------------------------------------------
morning_peak = 1.5 * exp(-(t - 8).^2 / (2*1.2^2));
evening_peak = 2.5 * exp(-(t - 19).^2 / (2*1.5^2));
night_background = 0.5 + 0.1*sin(0.5*t);
noise = 0.15 * randn(size(t));

L_true = morning_peak + evening_peak + night_background + noise;
L_true = max(L_true,0.2);  % ensure no negative values


%% ------------------------------------------------------------
% 2. Very naive forecast (flat + big noise)
% ------------------------------------------------------------
avg_load = mean(L_true);
L_fore_naive = avg_load + 0.4*randn(size(t));
L_fore_naive = max(L_fore_naive, 0.2);


%% ------------------------------------------------------------
% 3. Improved smart forecast aligned with true peak times
% ------------------------------------------------------------
% Morning peak around 08:00
% smart_morning = 1.3 * exp(-(t - 8).^2 / (2*1.3^2));
smart_morning = 1.3 * exp(-(t - 7).^2 / (2*1.3^2));

% Evening peak around 19:00
smart_evening = 2.0 * exp(-(t - 18.5).^2 / (2*1.8^2));

% Better background estimate
smart_background = 0.55 + 0.08*sin(0.4*t);

% Add modest noise
L_fore_smart = smart_morning + smart_evening + smart_background ...
                + 0.07*randn(size(t));
L_fore_smart = max(L_fore_smart,0.2);


%% ------------------------------------------------------------
% 4. Forecast errors
% ------------------------------------------------------------
MAE_naive = mean(abs(L_fore_naive - L_true));
MAE_smart = mean(abs(L_fore_smart - L_true));

fprintf('Naive forecast MAE  = %.3f\n', MAE_naive);
fprintf('Smart forecast MAE  = %.3f\n', MAE_smart);


%% ------------------------------------------------------------
% 5. Plot comparison
% ------------------------------------------------------------
figure;
plot(t, L_true, 'k-', 'LineWidth', 2); hold on;
plot(t, L_fore_naive, 'r--', 'LineWidth', 1.5);
plot(t, L_fore_smart, 'b-.', 'LineWidth', 1.8);

grid on;
xlabel('Time [hours]');
ylabel('Load [kW]');
title('Ground Truth vs. Naive vs. Improved Smart Forecast');
legend('Ground truth', ...
       sprintf('Naive forecast (MAE=%.2f)', MAE_naive), ...
       sprintf('Smart forecast (MAE=%.2f)', MAE_smart), ...
       'Location', 'NorthWest');

%%
