%%
% Nume si prenume: TODO
%

clearvars
clc

%% Magic numbers (replace with received numbers)
m = 1; % functionality not implemented yet
n = 1; % functionality not implemented yet

%% Experiment setup (fixed, do not modify)
Ts = 1e-2; % fundamental step size
Tfin = 20; % simulation duration

umin = -15; umax = 15; % input saturation
ymin = -15; ymax = 15; % output saturation

whtn_pow_in = 1e-5/2; % input white noise power and sampling time
whtn_Ts_in = Ts;
whtn_seed_in = 23341;
q_in = (umax-umin)/pow2(10); % input quantizer (DAC)

whtn_pow_out = 1e-6; % output white noise power and sampling time
whtn_Ts_out = Ts;
whtn_seed_out = 23342;
q_out = (ymax-ymin)/pow2(10); % output quantizer (ADC)

%% Process data (fixed, do not modify)
time_constant = 0.5;
static_gain = 1.76;

x0 = -0.5; % initial conditions

%% Input setup
u0 = 0;
ust = 1;
t1 = 10;

%% Data acquisition (use t, u, y to perform system identification)
out = sim("basic_example_R2023b.slx");

t = out.tout;
u = out.u;
y = out.y;

plot(t,u,t,y)

%% System identification
