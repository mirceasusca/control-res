%% MASTER TAPES!
%%
% Nume si prenume: TODO
%

% clearvars
% clc

%% Magic numbers (replace with received numbers)
m = 1; % functionality not implemented yet
n = 1; % functionality not implemented yet

%% Experiment setup (fixed, do not modify)
Ts = 1e-1; % fundamental step size
Tfin = 1500; % simulation duration

umin = -15; umax = 15; % input saturation
ymin = -15; ymax = 15; % output saturation

whtn_pow_in = 1e-6; % input white noise power and sampling time
whtn_Ts_in = Ts;
whtn_seed_in = 23341;
q_in = (umax-umin)/pow2(12); % input quantizer (DAC)

whtn_pow_out = 1e-6; % output white noise power and sampling time
whtn_Ts_out = Ts;
whtn_seed_out = 23342;
q_out = (ymax-ymin)/pow2(12); % output quantizer (ADC)

%% Process data (fixed, do not modify)
T1 = 20;
T2 = 1.5;
K = 1.2;

H = tf(K,conv([T1,1],[T2,1]))
% figure
% bode(H)

x0 = [7,0.1]; % initial conditions
% x0 = [0,0]; % initial conditions

% Input setup
u0 = 0;
ust = 1;
t1 = 10;

% probabil bune
% f1 = 1/40/pi/5;
% f2 = 1/4/pi*5;

f1 = 1/40/pi/5;
f2 = 1/4/pi*5;

%% Data acquisition (use t, u, y to perform system identification)
out = sim("sistem_termic_RF_model.slx");

t = out.tout;
u = out.u;
y = out.y;

% semilogx(t,20*log10(abs(u-5)),t,20*log10(abs(y-6)))
plot(t,u,t,y)

%% System identification
Tu = 2*(160.7-86.8);
wu = 2*pi/Tu
DT = 102.8-86.8
phi = -wu*DT

Tu = 2*(322.7-290)
Ty = 2*(335.5-304.1)
wu = 2*pi/Tu
wy = 2*pi/Ty
DT = 304.1-290
phi = -wu*DT

Tu = 2*(379.5-351.9)
Ty = 2*(390.7-364.5)
wu = 2*pi/Tu
wy = 2*pi/Ty
DT = 364.5-351.9
phi = -wu*DT

Tu = 2*(430.4-405.1)
Ty = 2*(439.9-416.5)
wu = 2*pi/Tu
wy = 2*pi/Ty
DT = 416.5-405.1
phi = -wu*DT

Tu = 2*(622.9-605.4)
Ty = 2*(630.6-614.2)
wu = 2*pi/Tu
wy = 2*pi/Ty
DT = 614.2-605.4
phi = -wu*DT

uM = 6.19629;
um = 3.79395;
yM = 6.45742;
ym = 5.6543;
Mag = (yM-ym)/(uM-um)
Phi = 1.5798

wn = (wu+wy)/2;

%%
Im = Mag;

T1v = roots([wn^2*Im,-wn*K,Im])
T2v = 1/wn^2./T1v(2)

%% lsim
% figure,plot(t,u,t,y)

T1h = 17.7056;
T2h = 1.6406;
Kh = K;
H = tf(Kh,conv([T1h,1],[T2h,1]));
zpk(H)

ysim = lsim(H,u,t);
hold on
% plot(t,ysim)

A = [0,1;
    -1/T1h/T2h,-(1/T1h+1/T2h)];
B = [0;Kh/T1h/T2h];
C = [1,0]; 
D = 0;
sys = ss(A,B,C,D);
ysim2 = lsim(sys,u,t,[y(1),0.1]);
figure
plot(t,u,t,y,t,ysim2,'linewidth',1.5)

J1 = 1/length(u)*norm(y-ysim);
J2 = 1/length(u)*norm(y-ysim2);

empn1 = norm(y-ysim)/norm(y-mean(y))
empn2 = norm(y-ysim2)/norm(y-mean(y))