%%
% Nume si prenume: TODO
%

clearvars
clc

%% Magic numbers (replace with received numbers)
m = 0;
n = 4;

%% Process data (fixed, do not modify)
a1 = 2*(0.2+m/20)*(1000+n*200);
a0 = (1000+n*200)^2;
b0 = (2.2+m+n)*(1000+n*200)^2;

x0 = [-1e-5*m; (-1)^m*1e-7*2*n]; % initial conditions

%% Experiment setup (fixed, do not modify)
Ts = 20/a1/1e4; % fundamental step size
Tfin = 30/a1; % simulation duration

umin = -15; umax = 15; % input saturation
ymin = -24; ymax = 24; % output saturation

whtn_pow_in = 1e-7/2; % input white noise power and sampling time
whtn_Ts_in = Ts;
whtn_seed_in = 23341;
q_in = (umax-umin)/pow2(10); % input quantizer (DAC)

whtn_pow_out = 1e-6/2; % output white noise power and sampling time
whtn_Ts_out = Ts*5;
whtn_seed_out = 23342;
q_out = (ymax-ymin)/pow2(10); % output quantizer (ADC)

%% Input setup
u0 = 0;
ust = 1;
t1 = 10/a1; % Recomandat 

%% Data acquisition (use t, u, y to perform system identification)
out = sim("proiect_P1_R2024b.slx");

t = out.tout;
u = out.u;
y = out.y;

plot(t,u,t,y)

%% System identification
