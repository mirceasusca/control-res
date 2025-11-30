%% MASTER TAPES
zeta = 0.2;
wn = 20;
K = 1.4; 

H = tf(K*wn^2,[1,2*zeta*wn,wn^2]);

[y,t] = step(H);
u = ones(length(y),1);

figure,plot(t,u,t,y-K), hold on
plot(tk,exp(xk))
% plot(t,u,t,y-K)
% plot(t,abs(y-K))

%%

% idx = [15,29,43,56,70,84,97];
idx = [15,29];

xk = log(abs(y(idx)-K));
tk = t(idx);

figure
plot(tk,xk)

A = [sum(tk.^2),sum(tk);
    sum(tk),length(tk)];
b = [sum(xk.*tk);sum(xk)];
x = A\b
hold on
plot(tk,x(1)*tk+x(2))

Re_poli = x(1)
Re = Re_poli;

%%
Tosc = 2*(tk(2)-tk(1))
wosc = 2*pi/Tosc;

zeta_est = -Re/sqrt(wosc^2+Re^2);
wn_est = -Re/zeta_est;

%% 
Hest = tf(K*wn_est^2,[1,2*zeta_est*wn_est,wn_est^2]);

ysim = lsim(Hest,u,t);
figure
plot(t,u,t,y,t,ysim)

