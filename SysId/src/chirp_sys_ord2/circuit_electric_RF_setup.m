%%
% Nume si prenume: TODO
%

% clearvars
% clc

%% Magic numbers (replace with received numbers)
m = 1; % functionality not implemented yet
n = 1; % functionality not implemented yet

%% Experiment setup (fixed, do not modify)
Ts = 1e-5; % fundamental step size
Tfin = 13e-3; % simulation duration

umin = -12; umax = 12; % input saturation
ymin = -12; ymax = 12; % output saturation

whtn_pow_in = 1e-8/8; % input white noise power and sampling time
whtn_Ts_in = Ts;
whtn_seed_in = 23341;
q_in = (umax-umin)/pow2(10); % input quantizer (DAC)

whtn_pow_out = 1e-8/8; % output white noise power and sampling time
whtn_Ts_out = Ts;
whtn_seed_out = 23342;
q_out = (ymax-ymin)/pow2(10); % output quantizer (ADC)

%% Process data (fixed, do not modify)
wn = 3.1416e+04;
zeta = 0.35;
K = 1.02;

x0 = [1.05,-1000]; % initial conditions

%% Input setup
u0 = 0;
ust = 1;
t1 = 10;

%% Data acquisition (use t, u, y to perform system identification)
out = sim("circuit_electric_RF_model.slx");

t = out.tout;
u = out.u;
y = out.y;

plot(t,u,t,y)

%% System identification
