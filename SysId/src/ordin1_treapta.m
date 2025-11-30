close all 
% Load data from file
t = treapta0(:,1);
u = treapta0(:,2);
y = 2*treapta0(:,3);
% plot data
% plot(t,[u, y]); grid ; shg
plot(t,[u,y],'LineWidth',2); grid minor; shg
% extract the values
% indeces for stationary values
i1 = 404; % start index for initial value
i2 = 501; % stop index for initial value
j1 = 707; % start index for stationary value
j2 = 825; % stop index for stationary value
u0 = mean(u(i1:i2));
y0 = mean(y(i1:i2));
ust = mean(u(j1:j2));
yst = mean(y(j1:j2));
hold on
plot(t, ust*ones(1,length(t)),'--r','LineWidth',2);
plot(t, u0*ones(1,length(t)),'--r','LineWidth',2);
% plot(t, yst*ones(1,length(t)),'--g');
% plot(t, y0*ones(1,length(t)),'--g');
% legend('u','y','u_{st}','u_0','y_{st}','y_0');
legend('u','u_{st}','u_0');
xlabel('Timp[s]'); ylabel('u');
%%
K = 2.4; 
T = 3e-3; 

A = [-1/T]; B = [K/T]; C = 1; D = 0;

y1 = lsim(A,B,C,D,u-ust,t,0) + K*ust + rand(length(t),1)-0.5;
figure
plot(t,[u,y1],'LineWidth',2); grid minor; shg
hold on
% plot(t, ust*ones(1,length(t)),'--r','LineWidth',2);
% plot(t, u0*ones(1,length(t)),'--r','LineWidth',2);
% plot(t, K*ust*ones(1,length(t)),'--k','LineWidth',2);
% plot(t, K*u0*ones(1,length(t)),'--k','LineWidth',2);
% legend('u','y','u_{st}','u_0','y_{st}','y_0');
legend('u','y');
xlabel('Timp[s]'); ylabel('u/y');


% plot a horizontal line at 63%    
hold on    
y0 = K*u0;
yst = K*ust;
y63=0.63*(yst-y0)+y0;
plot(t,y63*ones(1,length(t)),'r');
xline(t(501),'--','LineWidth',2)
xline(t(560),'--','LineWidth',2)
legend('u','y','y_{63}');

%%
% ysim = lsim(tf(K,[T 1]),u,t);
ysim = lsim(ss(-1/T,K/T,1,0),u,t,yst);
figure;
plot(t,u,t,y1,t,ysim,'LineWidth',2); grid minor; shg
xlabel('Timp[s]'); ylabel('u/y/y_{sim}');
legend('u','y','y_{sim}');
xlim([t(1) t(end)])

%% extract the indices to calculate T
i3 = 502;
i4 = 508;
% gain factor
k = (yst-y0)/(ust-u0); 
% time
T=t(i4)-t(i3);
% construct the transfer function
H=tf(k,[T,1]);
% obtain the simulation results
ysim=lsim(H,u,t);
figure
plot(t,[u,y,ysim]); shg

% simulate from nonzero initial conditions
A = -1/T; B = k/T; C = 1; D = 0;
ysim1 = lsim(A,B,C,D,u,t,y(1));
figure
plot(t,[u,y,ysim1],'Linewidth', 1.5); shg; grid minor

xlim([-0.01 0.01])
legend('u','y','y_{sim}')
xlabel('Time[s]'); ylabel('u/y')
% evaluate the results
J = norm(y-ysim1)
e_MPN = norm(y-ysim1)/norm(y-mean(y))
 