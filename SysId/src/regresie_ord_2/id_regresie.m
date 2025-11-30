clearvars
clc

%% Magic numbers (change with received numbers)
m = 1;
n = 1;

%% Experiment setup
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

%% Process data
x0 = [0,0];

time_constant1 = 2;
time_constant2 = 0.5;
static_gain = 1.6;
tau_m = 0.0;

%% Input setup
u0 = 0;
ust = 1;

%% Data acquisition
out = sim("blank_example_setup.slx");

t = out.tout;
u = out.u;
y = out.y;

plot(t,u,t,y)
shg

%% System identification
% dat = iddata(y,u,t(2)-t(1));
% nk = delayest(dat,2,1,20,50)
% nk*Ts

i1 = 891;
i2 = 987;
i3 = 1902;
i4 = 1989;

u0 = mean(u(i1:i2));
ust = mean(u(i3:i4));
y0 = mean(y(i1:i2));
yst = mean(y(i3:i4));

K = (yst-y0)/(ust-u0);

i5 = 1155;
i6 = 1549;

xk = log(yst-y(i5:i6));
tk = t(i5:i6);

A = [sum(tk.^2),sum(tk);
    sum(tk),length(tk)];
b = [sum(xk.*tk);sum(xk)];
x = A\b

T1 = -1/x(1)

%%
i7 = 1032;
i8 = 1054;

tk = t(i7:i8);
xk = log(y(i7:i8)-yst+(yst-y0)*exp(-(tk-t(1050))/T1));
% xk = log(y(i7:i8)-yst);
plot(tk,xk)
% plot(tk,y(i7:i8))
shg

A = [sum(tk.^2),sum(tk);
    sum(tk),length(tk)];
b = [sum(xk.*tk);sum(xk)];
x = A\b

T2 = -1/x(1);

%%
A = [0,1;
    -1/T1/T2,-(1/T1+1/T2)];
B = [0;K/T1/T2];
C = [1,0];
D = 0;

u_new = u;
% u_new = [u(1)*ones(1,30)';u(1:end-30)];
ysim = lsim(ss(A,B,C,D),u_new,t,[y(1),0]);

plot(t,u,t,y,t,ysim)


