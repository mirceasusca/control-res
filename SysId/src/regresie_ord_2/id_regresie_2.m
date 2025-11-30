% MASTER TAPES
clearvars
clc

%% Magic numbers (change with received numbers)
m = 1;
n = 1;

%% Experiment setup
Ts = 1e-2; % fundamental step size
Tfin = 40; % simulation duration

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

time_constant1 = 4;
time_constant2 = 1;
static_gain = 1.6;
% tau_m = 0.0;

%% Data acquisition
out = sim("schema_regresie.slx");

t = out.tout;
u = out.u;
y = out.y;

figure
plot(t,u,t,y)
shg

%% System identification
% dat = iddata(y,u,t(2)-t(1));
% nk = delayest(dat,2,1,20,50)
% nk*Ts

i1 = 1767;
i2 = 1974;
i3 = 3718;
i4 = 3949;

u0 = mean(u(i1:i2));
ust = mean(u(i3:i4));
y0 = mean(y(i1:i2));
yst = mean(y(i3:i4));

K = (yst-y0)/(ust-u0);

%%
i5 = 2095;
i6 = 3002;

xk = log(yst-y(i5:i6));
tk = t(i5:i6);
plot(tk,xk)

A = [sum(tk.^2),sum(tk);
    sum(tk),length(tk)];
b = [sum(xk.*tk);sum(xk)];
x = A\b

T1 = -1/x(1)

%%
% i7 = 2001;
% i8 = 3600;
% % i7 = 3500;
% % i8 = 3600;
% i9 = 2001; % step trigger
% 
% tk = t(i7:i8);
% T2_test = 0.5;
% % xk = log(y(i7:i8)-yst+(yst-y0)*T1/(T1-T2_test)*exp(-(tk-t(i9))/T1));
% % xk = log(y(i7:i8)-yst+(yst-y0)*time_constant1/(time_constant1-time_constant2)*exp(-(tk-t(i9))/time_constant1));
% % xk = log(y(i7:i8)-yst+(yst-y0)*exp(-(tk-t(i9))/time_constant1));
% % xk = log(y(i7:i8)-yst+(yst-y0)*exp(-(tk-t(i9))/T1));
% % xk = (yst - y(i7:i8));
% % +(yst-y0)*exp(-(tk-t(i9))/T1)
% xk = log(yst-y(i7:i8)+(yst-y0)*T1/(T1-T2_test)*exp(-(tk-t(i9))/T1));
% % xk = y(i7:i8)-yst+(yst-y0)*T1/(T1-T2)*exp(-(tk-t(i9))/T1);
% % xk = log(y(i7:i8)-yst);
% figure
% plot(tk,xk)
% % plot(tk,y(i7:i8))
% % shg
% 
% A = [sum(tk.^2),sum(tk);
%     sum(tk),length(tk)];
% b = [sum(xk.*tk);sum(xk)];
% x = A\b
% hold on
% plot(tk,x(1)*tk+x(2))
% 
% T2 = -1/x(1)
% % T2 = 0.5;
% 
% %%
% A = [0,1;
%     -1/T1/T2,-(1/T1+1/T2)];
% B = [0;K/T1/T2];
% C = [1,0];
% D = 0;
% 
% u_new = u;
% % u_new = [u(1)*ones(1,30)';u(1:end-30)];
% ysim = lsim(ss(A,B,C,D),u_new,t,[y(1),0]);
% 
% plot(t,u,t,y,t,ysim)
% 

%%
i7 = 2001;
i8 = 2156;

% Tinfl = t(i8)-t(i7)
Tinfl = 1.85
T1
T2v = linspace(0.1,2,100);

F = @(T2var)T1*T2var*log(T2var)-T2var*(Tinfl+T1*log(T1))+T1*Tinfl;
% T2sol = double(solve(T1*T2var*log(T2var)-T2var*(Tinfl+T1*log(T1))+T1*Tinfl == 0, T2var))
T2sol = fzero(F,1)

yv = T1*T2v.*log(T2v)-T2v*(Tinfl+T1*log(T1))+T1*Tinfl;
figure
subplot(211)
plot(T2v,yv), hold on
yline(0)
xline(T2sol)

% T2 = 0.7641;
T2 = T2sol;
A = [0,1;
    -1/T1/T2,-(1/T1+1/T2)];
B = [0;K/T1/T2];
C = [1,0];
D = 0;

u_new = u;
% u_new = [u(1)*ones(1,30)';u(1:end-30)];
ysim = lsim(ss(A,B,C,D),u_new,t,[y(1),0]);

subplot(212)
plot(t,u,t,y,t,ysim)
