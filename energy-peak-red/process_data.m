%% actual_pred_DAY_2010_11_18
load('actual_pred_DAY_2010_11_18.mat')
actual_6_22 = actual_pred_DAY_2010_11_18(:,1);
predicted_6_22 = actual_pred_DAY_2010_11_18(:,2);
% rng(2);
actual = [0.3*rand(6,1);actual_6_22;0.3*rand(2,1)];
predicted = [0.3*rand(6,1);predicted_6_22;0.3*rand(2,1)];
t_hourly = 0:24;

%% actual_pred_DAY_2010_11_23
load('actual_pred_DAY_2010_11_23.mat')
actual_6_22 = actual_pred_DAY_2010_11_23(:,1);
predicted_6_22 = actual_pred_DAY_2010_11_23(:,2);
actual = [0.3*rand(6,1);actual_6_22;0.3*rand(2,1)];
predicted = [0.3*rand(6,1);predicted_6_22;0.3*rand(2,1)];
t_hourly = 0:24;

%% actual_pred_DAY_2010_11_25
load('actual_pred_DAY_2010_11_25.mat')
actual_6_22 = actual_pred_DAY_2010_11_25(:,1);
predicted_6_22 = actual_pred_DAY_2010_11_25(:,2);
actual = [0.3*rand(6,1);actual_6_22;0.3*rand(2,1)];
predicted = [0.3*rand(6,1);predicted_6_22;0.3*rand(2,1)];
t_hourly = 0:24;

%%

figure
plot(t_hourly,actual,t_hourly,predicted)
legend('actual','predicted')

% interpolate data
t = 0:1/4:24;
L_fore_smart = interp1(t_hourly,predicted,t);
L_true = interp1(t_hourly,actual,t);

figure
plot(t,L_true,t,L_fore_smart)
legend('actual','predicted')