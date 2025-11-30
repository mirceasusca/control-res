
ysim = lsim(h,u,t);
hold on
plot(t,ysim)

ysim2 = lsim(tf(static_gain,[time_constant,1]),u,t);
plot(t,ysim2)

r1 = xcorr(y,ysim);
r2 = xcorr(y,ysim2);

%% System identification
