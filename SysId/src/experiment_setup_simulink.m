clearvars
clc

%% Magic numbers
m = 1; 
n = 1;

%% Experiment Setup
Ts = 1e-2;
Tfin = 20;

whtn_pow_in = 1e-5; % input white noise power and sampling time
whtn_Ts_in = Ts;
whtn_seed_in = 23341;
q_in = 1/pow2(6); % input quantizer (DAC)

whtn_pow_out = 1e-6; % output white noise power and sampling time
whtn_Ts_out = Ts;
whtn_seed_out = 23342;
q_out = 1/pow2(6); % output quantizer (ADC)

x0 = -0.5;

u0 = 0;
ust = 2;

%% Process data
time_constant = 0.5;
static_gain = 1.76;